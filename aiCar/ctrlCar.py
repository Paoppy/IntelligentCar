import cv2
import time
import serial
from model import find_camera_index, init_model, detect_lane_lines, infer_frame_with_vis, img2bytes
from queue import Queue
from collections import Counter

# 创建一个线程安全的队列用于共享车道检测结果
lane_data_queue = Queue()
# 创建一个线程安全的队列来传递检测到的目标
detected_objects_queue = Queue()

# Arduino 串口参数
arduino_port = '/dev/ttyAMA0'  # 替换为你的 Arduino 端口
arduino_baudrate = 115200  # 波特率

# 初始化串口连接
try:
    arduino = serial.Serial(arduino_port, arduino_baudrate, timeout=1)
    print(f"Connected to Arduino on {arduino_port}")
except Exception as e:
    print(f"Failed to connect to Arduino: {e}")
    exit()

cfg = {
    'conf_thres': 0.6,  # 模型置信度阈值，阈值越低，得到的预测框越多
    'iou_thres': 0.6,  # IOU阈值，高于这个阈值的重叠预测框会被过滤掉
    'input_shape': [640, 640],  # 模型输入尺寸
}

def send_command(command):
    global arduino
    try:
        if not arduino.is_open:
            print("串口已关闭，尝试重新打开...")
            arduino.open()
        arduino.write((command + "&").encode())  # 确保发送终止符一致
        # print(f"已发送串口命令: {command}")
    except serial.SerialException as e:
        print(f"串口通信失败: {e}")
        # 尝试重新连接串口
        try:
            arduino.close()
            time.sleep(2)  # 延迟确保端口重置
            arduino.open()
            print("串口重新连接成功")
        except Exception as reconnect_error:
            print(f"重新连接串口失败: {reconnect_error}")


def gen_frames():
    # 获取摄像头
    camera_index = find_camera_index()
    cap = cv2.VideoCapture(camera_index)
    model, labels_dict = init_model()

    # 用于记录最近10次检测结果
    recent_detections = []

    while True:
        # 对摄像头每一帧进行推理和可视化
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 320))

        # 车道线检测
        lane_image, lanes = detect_lane_lines(frame)

        # 偏移量计算
        if lanes[0] is not None and lanes[1] is not None:
            lane_center = (lanes[0][0] + lanes[0][2] + lanes[1][0] + lanes[1][2]) // 4
            car_center = frame.shape[1] // 2
            offset = lane_center - car_center

            # 根据偏移量发送命令
            if abs(offset) > 30:
                command = "AR" if offset > 0 else "AL"
            else:
                command = "F"

            if not lane_data_queue.empty():
                lane_data_queue.get()  # 清空队列中的旧指令
            lane_data_queue.put(command)  # 放入新指令
            
        elif lanes[0] is None and lanes[1] is not None:  # 无法检测到左车道线
            command = "AL"  # 向左转
            if not lane_data_queue.empty():
                lane_data_queue.get()  # 清空队列中的旧指令
            lane_data_queue.put(command)
            
        elif lanes[0] is not None and lanes[1] is  None:
            command = "AR"
            if not lane_data_queue.empty():
                lane_data_queue.get()  # 清空队列中的旧指令
            lane_data_queue.put(command)  # 放入新指令

        # 获取检测可视化结果
        image_pred, detected_objects = infer_frame_with_vis(frame, model, labels_dict, cfg)

        # 更新最近10次检测结果
        recent_detections.extend(detected_objects)
        if len(recent_detections) > 10:
            recent_detections = recent_detections[-10:]  # 保留最近10次结果
            most_common_object = Counter(recent_detections).most_common(1)[0][0]
            if not detected_objects_queue.empty():
                detected_objects_queue.get()  # 清空旧数据
            detected_objects_queue.put(most_common_object)  # 存入新结果
            recent_detections = []

        # 将车道线叠加到预测图像上
        combined_image = cv2.addWeighted(image_pred, 0.7, lane_image, 0.3, 0)

        # 将结果显示到画面
        frame_bytes = img2bytes(combined_image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
