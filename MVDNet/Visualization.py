import math
import os

import cv2
import numpy as np
import json

input_dir = r'radar'
save_dir = r'after_images1'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# 定义边界框颜色
color = (255, 0, 0)
with open("coco_instances_results.json", 'r', encoding="utf-8") as f:
    json_data = json.load(f)
with open('eval.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split()
        img_id = line[0].strip()
        img_name = line[1] + '.jpg'
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        detections = []
        for img_coco_info in json_data:
            if int(img_id) == img_coco_info['image_id']:
                detections.append(img_coco_info)
        print(len(detections), img_id, img_name)
        if len(detections) == 0:
            continue
        for detection in detections:
            bbox = detection['bbox']  # 获取边界框坐标信息
            x, y, w, h, angle = bbox  # 分别获取左上角坐标、宽度和高度, 旋转角度
            x_center, y_center = x + w / 2, y + h / 2  # 计算中心点坐标
            rad = math.radians(angle)  # 将角度转换为弧度
            cos = math.cos(rad)
            sin = math.sin(rad)
            # 计算四个角点的坐标
            top_left = (int(x_center - w / 2), int(y_center - h / 2))
            top_right = (int(x_center + w / 2), int(y_center - h / 2))
            bottom_left = (int(x_center - w / 2), int(y_center + h / 2))
            bottom_right = (int(x_center + w / 2), int(y_center + h / 2))
            corners = [top_left, top_right, bottom_right, bottom_left]
            rotated_corners = []
            for corner in corners:
                x, y = corner
                x -= x_center
                y -= y_center
                new_x = x * cos - y * sin
                new_y = x * sin + y * cos
                rotated_corners.append((int(new_x + x_center), int(new_y + y_center)))
            # 在图像上画出旋转后的边界框
            for i in range(4):
                cv2.line(img, rotated_corners[i], rotated_corners[(i + 1) % 4], color, 2)

        save_path = os.path.join(save_dir, img_name)
        cv2.imencode('.jpg', img)[1].tofile(save_path)

        print(f'已保存{save_path}')
