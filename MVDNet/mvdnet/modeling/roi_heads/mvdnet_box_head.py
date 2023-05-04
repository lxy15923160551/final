import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from typing import Dict

from detectron2.layers import Conv2d, Linear, ShapeSpec, get_norm
from detectron2.modeling.roi_heads import ROI_BOX_HEAD_REGISTRY
from ..attention import SelfAttentionBlock, CrossAttentionBlock
from mvdnet.layers import Conv3d


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=True, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


@ROI_BOX_HEAD_REGISTRY.register()
class MVDNetBoxHead(nn.Module):
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        num_fc     = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim     = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm       = cfg.MODEL.ROI_BOX_HEAD.NORM
        self.history_on = cfg.INPUT.HISTORY_ON
        self.num_history = cfg.INPUT.NUM_HISTORY+1
        self.pooler_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        assert num_fc > 0

        for f in input_shape.keys():
            if f.startswith("radar"):
                self.radar_key = f
                self.radar_output_size = input_shape[f].channels * input_shape[f].height * input_shape[f].width
                self.radar_input_channels = input_shape[f].channels
            elif f.startswith("lidar"):
                self.lidar_key = f
                self.lidar_output_size = input_shape[f].channels * input_shape[f].height * input_shape[f].width
                self.lidar_input_channels = input_shape[f].channels

        assert(self.lidar_output_size >= self.radar_output_size)
        if self.lidar_output_size != self.radar_output_size:
            self.match_conv = Conv2d(
                in_channels = self.lidar_input_channels,
                out_channels = self.radar_input_channels,
                kernel_size = 3,
                padding = 1,
                bias = False,
                norm = nn.BatchNorm2d(self.radar_input_channels),
                activation = F.leaky_relu_
            )
        else:
            self.match_conv = None
        self.radar_self_attention = SelfAttentionBlock(self.radar_output_size)
        self.lidar_self_attention = SelfAttentionBlock(self.radar_output_size)
        self.radar_cross_attention = CrossAttentionBlock(self.radar_output_size)
        self.lidar_cross_attention = CrossAttentionBlock(self.radar_output_size)

    
        if self.history_on:
            self.conv1 = ConvLSTM(64, 64, (3,3), 2)
            self.conv2 = ConvLSTM(64, 64, (3,3), 2)
            
            self.tnn1 = Conv3d(
                in_channels = self.radar_input_channels*2,
                out_channels = self.radar_input_channels,
                kernel_size = [3, 3, 3],
                padding = [1, 1, 1],
                bias=False,
                norm=nn.BatchNorm3d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
            self.tnn2 = Conv3d(
                in_channels = self.radar_input_channels,
                out_channels = self.radar_input_channels,
                kernel_size = [3, 3, 3],
                padding = [1, 1, 1],
                bias=False,
                norm=nn.BatchNorm3d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
            self.tnn3 = Conv3d(
                in_channels = self.radar_input_channels,
                out_channels = self.radar_input_channels,
                kernel_size = [self.num_history, 3, 3],
                padding = [0, 1, 1],
                bias=False,
                norm=nn.BatchNorm3d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
            
            self.tnns = [self.tnn1, self.tnn2, self.tnn3]
            
        else:
            self.tnn = Conv2d(
                in_channels = self.radar_input_channels*2,
                out_channels = self.radar_input_channels,
                kernel_size = 3,
                padding = 1,
                bias=False,
                norm=nn.BatchNorm2d(self.radar_input_channels),
                activation=F.leaky_relu_
            )
        self._output_size = self.radar_output_size
        
        self.fcs = []
        for k in range(num_fc):
            fc = Linear(self._output_size, fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)
        if self.match_conv is not None:
            weight_init.c2_msra_fill(self.match_conv)
        if self.history_on:
            for layer in self.tnns:
                weight_init.c2_msra_fill(layer)
            #pass
        else:
            weight_init.c2_msra_fill(self.tnn)

    def forward(self, x):
        radar_features = x[self.radar_key]
        lidar_features = x[self.lidar_key]

        if self.history_on:
            fusion_feature = []
            for radar_x, lidar_x in zip(radar_features, lidar_features):
                if self.match_conv is not None:
                    lidar_x = self.match_conv(lidar_x)
                radar_x = torch.flatten(radar_x, start_dim=1)
                lidar_x = torch.flatten(lidar_x, start_dim=1)
                radar_x = self.radar_self_attention(radar_x)
                lidar_x = self.lidar_self_attention(lidar_x)
                radar_y = self.radar_cross_attention([radar_x, lidar_x])
                lidar_y = self.lidar_cross_attention([lidar_x, radar_x])
                radar_y = radar_y.reshape(-1, self.radar_input_channels,
                    self.pooler_size, self.pooler_size)
                lidar_y = lidar_y.reshape(-1, self.radar_input_channels,
                    self.pooler_size, self.pooler_size)
                feature_x = torch.cat([radar_y, lidar_y], dim=1)
                fusion_feature.append(feature_x)
            fusion_feature = torch.stack(fusion_feature).permute(1,2,0,3,4).contiguous()
            #print("1:",fusion_feature.shape)
            for layer in self.tnns:
                #print("1:",fusion_feature.shape)
                fusion_feature = layer(fusion_feature)
                #print("o:",fusion_feature.shape)
            #if fusion_feature.shape[1] == 32:
            #    fusion_feature = self.conv1(fusion_feature)
            #else:
                #fusion_feature = self.conv2(fusion_feature)
            fusion_feature = torch.flatten(fusion_feature, start_dim=1)
        else:
            if self.match_conv is not None:
                lidar_features = self.match_conv(lidar_features)
            radar_x = torch.flatten(radar_features, start_dim=1)
            lidar_x = torch.flatten(lidar_features, start_dim=1)
            radar_x = self.radar_self_attention(radar_x)
            lidar_x = self.lidar_self_attention(lidar_x)
            radar_y = self.radar_cross_attention([radar_x, lidar_x])
            lidar_y = self.lidar_cross_attention([lidar_x, radar_x])
            radar_y = radar_y.reshape(-1, self.radar_input_channels,
                self.pooler_size, self.pooler_size)
            lidar_y = lidar_y.reshape(-1, self.radar_input_channels,
                self.pooler_size, self.pooler_size)
            feature_x = torch.cat([radar_y, lidar_y], dim=1)
            #print("2:",feature_x.shape)
            feature_x = self.tnn(feature_x)
            fusion_feature = torch.flatten(feature_x, start_dim=1)
        
        for layer in self.fcs:
            fusion_feature = F.leaky_relu_(layer(fusion_feature))
        return fusion_feature

    @property
    def output_size(self):
        return self._output_size
