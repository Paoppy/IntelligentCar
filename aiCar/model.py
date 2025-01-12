import cv2
import numpy as np
import torch
from ais_bench.infer.interface import InferSession
from det_utils import letterbox, scale_coords, nms

from collections import deque

# 全局变量用于保存车道线历史
left_lane_history = deque(maxlen=5)
right_lane_history = deque(maxlen=5)


def preprocess_image(image, cfg, bgr2rgb=True):
    """图片预处理"""
    img, scale_ratio, pad_size = letterbox(image, new_shape=cfg['input_shape'])
    if bgr2rgb:
        img = img[:, :, ::-1]
    img = img.transpose(2, 0, 1)  # HWC2CHW
    img = np.ascontiguousarray(img, dtype=np.float16)
    return img, scale_ratio, pad_size


def draw_bbox(bbox, img0, color, wt, names):
    """在图片上画预测框，同时记录目标名称"""
    detected_objects = []  # 用于存储检测到的目标名称
    for idx, class_id in enumerate(bbox[:, 5]):
        if float(bbox[idx][4] < float(0.05)):
            continue
        img0 = cv2.rectangle(img0, (int(bbox[idx][0]), int(bbox[idx][1])), (int(bbox[idx][2]), int(bbox[idx][3])),
                             color, wt)
        img0 = cv2.putText(img0, str(idx) + ' ' + names[int(class_id)], (int(bbox[idx][0]), int(bbox[idx][1] + 16)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        img0 = cv2.putText(img0, '{:.4f}'.format(bbox[idx][4]), (int(bbox[idx][0]), int(bbox[idx][1] + 32)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        detected_objects.append(names[int(class_id)])  # 添加目标名称到列表

    return img0, detected_objects  # 返回图片和检测到的目标列表


def get_labels_from_txt(path):
    """从txt文件获取图片标签"""
    labels_dict = dict()
    with open(path) as f:
        for cat_id, label in enumerate(f.readlines()):
            labels_dict[cat_id] = label.strip()
    return labels_dict


def infer_frame_with_vis(image, model, labels_dict, cfg, bgr2rgb=True):
    # 数据预处理
    img, scale_ratio, pad_size = preprocess_image(image, cfg, bgr2rgb)
    # 模型推理
    output = model.infer([img*0.003])[0]

    output = torch.tensor(output)
    # 非极大值抑制后处理
    boxout = nms(output, conf_thres=cfg["conf_thres"], iou_thres=cfg["iou_thres"])
    pred_all = boxout[0].numpy()
    # 预测坐标转换
    scale_coords(cfg['input_shape'], pred_all[:, :4], image.shape, ratio_pad=(scale_ratio, pad_size))
    # 图片预测结果可视化并获取目标名称列表
    img_vis, detected_objects  = draw_bbox(pred_all, image, (0, 255, 0), 2, labels_dict)
    
    return img_vis, detected_objects  # 返回检测可视化结果


def img2bytes(image):
    """将图片转换为字节码"""
    return bytes(cv2.imencode('.jpg', image)[1])


def find_camera_index():
    max_index_to_check = 10  # Maximum index to check for camera

    for index in range(max_index_to_check):
        cap = cv2.VideoCapture(index)
        if cap.read()[0]:
            cap.release()
            return index

    # If no camera is found
    raise ValueError("No camera found.")


def init_model():
    model_path = 'car.om'
    label_path = './car.txt'
    # 初始化推理模型
    model = InferSession(0, model_path)
    labels_dict = get_labels_from_txt(label_path)
    return model, labels_dict


def detect_lane_lines(image):
    """
    检测车道线并对车道线进行优化和标记为红色。
    """
    global left_lane_history, right_lane_history

    # 转为灰度图像并进行高斯模糊
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 边缘检测
    edges = cv2.Canny(blurred, 30, 150)

    # 定义感兴趣区域（ROI）
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height),
        (width, height),
        (width // 2 + 100, height // 2),
        (width // 2 - 100, height // 2),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # 霍夫变换检测车道线
    lines = cv2.HoughLinesP(
        cropped_edges,
        rho=1,  # 累加器的像素分辨率
        theta=np.pi / 180,  # 累加器的角度分辨率
        threshold=50,  # 投票的最小阈值（50）
        minLineLength=40,  # 线条最小长度（40px）
        maxLineGap=150
    )

    # 初始化左右车道线
    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if x2 != x1 else float('inf')
            if slope < -0.5:  # 左车道线
                left_fit.append((x1, y1, x2, y2))
            elif slope > 0.5:  # 右车道线
                right_fit.append((x1, y1, x2, y2))

    # 平滑车道线
    def smooth_lines(lines, history):
        if lines:
            avg_line = np.mean(lines, axis=0)
            history.append(avg_line)
        elif history:  # 如果没有检测到新车道线且有历史数据，则清空
            history.clear()
        if len(history) > 0:
            return np.mean(history, axis=0)
        return None

    left_lane = smooth_lines(left_fit, left_lane_history)
    right_lane = smooth_lines(right_fit, right_lane_history)

    # 创建叠加图层
    lane_image = np.zeros_like(image)

    # 绘制红色车道线
    def draw_lane_line(lane, image, color):
        if lane is not None:
            cv2.line(image, (int(lane[0]), int(lane[1])),
                     (int(lane[2]), int(lane[3])), color, 10)

    # 绘制车道线
    draw_lane_line(left_lane, lane_image, (0, 0, 255))  # 红色
    draw_lane_line(right_lane, lane_image, (0, 0, 255))  # 红色

    return lane_image, (left_lane, right_lane)