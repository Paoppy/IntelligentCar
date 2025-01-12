from flask import Flask, request, jsonify, Response, render_template
from ctrlCar import send_command, gen_frames, detected_objects_queue, lane_data_queue
import threading
import time

app = Flask(__name__)

# Timestamp for the last command sent to the car
last_command_time = time.time()
command_interval = 2.0  # Minimum interval (in seconds) between commands
last_send_time = time.time()  # 上次发送指令的时间戳
send_interval = 3.0  # 指令发送最小间隔（单位：秒）

# Flags to control threads
lane_keeping_active = threading.Event()
detection_monitor_active = threading.Event()

def monitor_detected_objects():
    """后台线程：实时输出 detected_objects_global 内容，并根据检测到的目标发送指令"""
    global last_send_time
    while detection_monitor_active.is_set():  # 检查标志是否为激活状态
        if not detected_objects_queue.empty():  # 如果队列中有数据
            detected_objects = detected_objects_queue.get()
            current_time = time.time()
            # 根据检测结果发送指令，并确保发送间隔大于指定值
            if detected_objects == "left" and current_time - last_send_time >= send_interval:
                send_command("L")
                last_send_time = current_time
                print("目标检测发送指令：L")
            elif detected_objects == "right" and current_time - last_send_time >= send_interval:
                send_command("R")
                last_send_time = current_time
                print("目标检测发送指令：R")

        time.sleep(0.5)  # 每隔 0.5 秒检查一次

def process_lane_keeping():
    """Background thread: Handles lane-keeping logic and sends commands."""
    global last_command_time
    while lane_keeping_active.is_set():  # 检查标志是否为激活状态
        command = None  # 默认命令为 None
        if not lane_data_queue.empty():
            command = lane_data_queue.get()  # 从队列中获取命令

        current_time = time.time()
        # 检查命令是否有效，并确保发送间隔小于指定时间
        if command and (current_time - last_command_time >= command_interval):
            send_command(command)  # 发送指令
            print("车道保持发送指令：" + command)
            last_command_time = current_time

        time.sleep(0.1)  # 缩短检查间隔以更及时地处理指令

def automatic_mode():
    # Start the lane-keeping thread
    lane_keeping_active.set()
    lane_keeping_thread = threading.Thread(target=process_lane_keeping, daemon=True)
    lane_keeping_thread.start()

    # Start the object detection monitoring thread
    detection_monitor_active.set()
    monitor_thread = threading.Thread(target=monitor_detected_objects, daemon=True)
    monitor_thread.start()

@app.route('/')
def home():
    # 渲染模板，并传递需要的数据（如果有的话）
    return render_template('web.html')

@app.route('/video_start')
def video_start():
    # 通过将一帧帧的图像返回，就达到了看视频的目的。multipart/x-mixed-replace是单次的http请求-响应模式，如果网络中断，会导致视频流异常终止，必须重新连接才能恢复
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/button-click', methods=['POST'])
def handle_button_click():
    global lane_keeping_active, detection_monitor_active
    data = request.get_json()
    command = data.get('message')
    if command == "A":
        automatic_mode()
    elif command == "C":
        # Stop both threads
        lane_keeping_active.clear()
        detection_monitor_active.clear()
        print("自动模式已停止")
    else:
        print(f"通过按钮发送指令: {command}")
        send_command(command)
    return jsonify({"status": "success", "message": f"Button {command} clicked!"})

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=5000, debug=True)
