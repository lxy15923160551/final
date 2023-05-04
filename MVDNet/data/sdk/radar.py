# Adapted from https://github.com/ori-mrg/robotcar-dataset-sdk/blob/master/python/radar.py
# Licensed under the Apache License

from typing import AnyStr, Tuple
import numpy as np
import cv2


def load_radar(example_path: AnyStr) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Decode a single Oxford Radar RobotCar Dataset radar example
    Args:
        example_path (AnyStr): Oxford Radar RobotCar Dataset Example png
    Returns:
        timestamps (np.ndarray): Timestamp for each azimuth in int64 (UNIX time)
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        valid (np.ndarray) Mask of whether azimuth data is an original sensor reading or interpolated from adjacent
            azimuths
        fft_data (np.ndarray): Radar power readings along each azimuth
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
    """
    # Hard coded configuration to simplify parsing code
    radar_resolution = np.array([0.0432], np.float32)
    encoder_size = 5600

    raw_example_data = cv2.imread(example_path, cv2.IMREAD_GRAYSCALE)
    timestamps = raw_example_data[:, :8].copy().view(np.int64)
    azimuths = (raw_example_data[:, 8:10].copy().view(np.uint16) / float(encoder_size) * 2 * np.pi).astype(np.float32)
    valid = raw_example_data[:, 10:11] == 255
    fft_data = raw_example_data[:, 11:].astype(np.float32)[:, :, np.newaxis] / 255.

    return timestamps, azimuths, valid, fft_data, radar_resolution


def radar_polar_to_cartesian(azimuths: np.ndarray, fft_data: np.ndarray, radar_resolution: float,
                             cart_resolution: float, cart_pixel_width: int, interpolate_crossover: bool,
                             fine_timestamp: np.ndarray, x0: float, y0: float, yaw0: float, x1:float, y1:float, yaw1:float) -> np.ndarray:
    """Convert a polar radar scan to cartesian.
    Args:
        azimuths (np.ndarray): Rotation for each polar radar azimuth (radians)
        fft_data (np.ndarray): Polar radar power readings
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        cart_resolution (float): Cartesian resolution (metres per pixel)
        cart_pixel_size (int): Width and height of the returned square cartesian output (pixels). Please see the Notes
            below for a full explanation of how this is used.
        interpolate_crossover (bool, optional): If true interpolates between the end and start  azimuth of the scan. In
            practice a scan before / after should be used but this prevents nan regions in the return cartesian form.

    Returns:
        np.ndarray: Cartesian radar power readings
    Notes:
        After using the warping grid the output radar cartesian is defined as as follows where
        X and Y are the `real` world locations of the pixels in metres:
         If 'cart_pixel_width' is odd:
                        +------ Y = -1 * cart_resolution (m)
                        |+----- Y =  0 (m) at centre pixel
                        ||+---- Y =  1 * cart_resolution (m)
                        |||+--- Y =  2 * cart_resolution (m)
                        |||| +- Y =  cart_pixel_width // 2 * cart_resolution (m) (at last pixel)
                        |||| +-----------+
                        vvvv             v
         +---------------+---------------+
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+ <-- X = 0 (m) at centre pixel
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         |               |               |
         +---------------+---------------+
         <------------------------------->
             cart_pixel_width (pixels)
         If 'cart_pixel_width' is even:
                        +------ Y = -0.5 * cart_resolution (m)
                        |+----- Y =  0.5 * cart_resolution (m)
                        ||+---- Y =  1.5 * cart_resolution (m)
                        |||+--- Y =  2.5 * cart_resolution (m)
                        |||| +- Y =  (cart_pixel_width / 2 - 0.5) * cart_resolution (m) (at last pixel)
                        |||| +----------+
                        vvvv            v
         +------------------------------+
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         |                              |
         +------------------------------+
         <------------------------------>
             cart_pixel_width (pixels)
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    coords = np.linspace(-cart_min_range, cart_min_range, cart_pixel_width, dtype=np.float32)

    # Compensate radar self motion
    Y, X = np.meshgrid(coords, -coords)
    angle_resolution = 360.0 / len(fine_timestamp)
    sample_angle = ((np.arctan2(Y-y0, X-x0) - yaw0) % (2 * np.pi)) / np.pi * 180
    sample_time = np.floor(sample_angle / angle_resolution).astype(int)
    sample_time[sample_time >= len(fine_timestamp)] = len(fine_timestamp) - 1
    sample_x1 = fine_timestamp[sample_time] * x1
    sample_y1 = fine_timestamp[sample_time] * y1
    sample_yaw1 = fine_timestamp[sample_time] * yaw1

    sample_range = np.sqrt((X - x0 - sample_x1) ** 2 + (Y - y0 - sample_y1) ** 2)
    sample_angle = (np.arctan2(Y - y0 - sample_y1, X - x0 - sample_x1) - yaw0 - sample_yaw1) % (2 * np.pi)

    # Interpolate Radar Data Coordinates
    azimuth_step = azimuths[1] - azimuths[0]
    sample_u = (sample_range - radar_resolution / 2) / radar_resolution
    sample_v = (sample_angle - azimuths[0]) / azimuth_step

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    if interpolate_crossover:
        fft_data = np.concatenate((fft_data[-1:], fft_data, fft_data[:1]), 0)
        sample_v = sample_v + 1

    polar_to_cart_warp = np.stack((sample_u, sample_v), -1).astype(np.float32)
    cart_img = np.expand_dims(cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    # cart_img = cv2.remap(fft_data, polar_to_cart_warp, None, cv2.INTER_LINEAR)
    return cart_img
