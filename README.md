# 智能小车辅助系统

一次简单的项目实践——智能小车辅助系统，具有简单的车道线检测、车道保持、自动避障等功能。本次实践旨在实现实现小车智能辅助系统，能够在行驶过程中对周围环境进行分析，实现自动避障功能（包括碰撞预警和紧急制动）、车道保持辅助等功能。

<br/>

## 一、系统整体架构

智能系统由多个模块协同组成，架构设计如图 1-1 所示。系统通过摄像头实时采集小车行驶环境中的视频数据，作为车道检测和目标识别的输入。车道检测模块对摄像头获取的图像进行处理，提取车道线信息，通过车道线信息计算出小车保持在车道中心行驶所需的控制信号。同时，目标识别模块对视频画面中的转向标志进行识别，并将识别结果提供给后端的决策线程。

整个系统的核心处理单元是Atlas模块，负责接收车道检测和目标识别的输出数据，综合分析后生成小车的运动控制指令。这些指令通过串口通信传递到Arduino模块。Arduino模块还结合超声波传感器反馈的障碍物信息，进一步优化控制指令，确保小车能够在复杂环境中平稳运行。

<p align="center"><img src="https://github.com/user-attachments/assets/8cb01b02-2843-4153-a48c-08b5cfb5f72b" alt="图 1-1 系统架构图"></p>

<br/>

## 二、模块设计

### 1. 硬件模块

小车硬件模块是智能驾驶辅助系统的基础部分，组成了小车的基本结构和边缘计算平台。其设计结合了Atlas开发板、Arduino微控制器、超声波传感器、电机控制器以及LED指示灯，通过硬件和软件的协同配合，实现了小车的运动控制和障碍物检测功能，具体硬件组成及其作用见下表，接线方式可参考代码 [Arduino.ino](Arduino/Arduino.ino) 中使用的引脚信息。

| 硬件名称 |	数量 | 作用 |
| :--     | :--: | :--  |
| Atlas DK A2	| 1	| 边缘计算，运行整个系统的前后端程序。 |
| Arduino Nano	| 1	| 微控制器，控制电机、超声波传感器和LED灯。 |
| 摄像头	| 1	| 用于视觉感知。 |
| 超声波传感器	| 2	| 前方传感器负责检测障碍，右侧传感器负责避障策略。 |
| 双色LED灯	| 1	| 用于提醒前方是否有障碍。（替代蜂鸣器） |
| 减速直流电机	| 4	| 控制车轮转动。 |

### 2. 前端模块

前端模块结合使用Jinjia2模板引擎和HTML模板，是用户操作和显示的界面。整个布局划分为视频显示占Web界面的2/3，按钮布局占1/3，Web界面样式如图2-1。按钮分为两个部分，顶部的两个按钮负责控制后端多线程的打开和关闭，底部的五个按钮用发送小车需要执行的指令。用户在前端通过按钮发送命令，后端通过API接收到这些命令后触发相应的功能。另外前端负责处理后端提供的视频流接口，供前端实时显示小车的视角。

<p align="center"><img src="https://github.com/user-attachments/assets/f50cfb2a-3d47-4c1b-aa57-117cc19b6af3" alt="图 2-1 Web界面样式"></p>


### 3. 后端模块

后端模块使用Flask框架构建，在整个系统中的作用是作为中间桥梁，用于与前端界面进行交互，以及从算法模块获取实时数据（如摄像头图像、车道检测结果、目标检测结果），根据处理结果，调用send_command函数，向小车发送控制指令，实现实时响应。

通过/video_start路由，将小车摄像头捕获的视频帧实时传输到前端界面。通过API接收前端的按钮点击命令（/api/button-click路由）。根据接收到的来自前端的指令，判断需要向小车发送控制指令或是需要更改多线程的状态。使用多线程同时处理车道保持和目标检测逻辑，确保视频流在执行多种算法的同时能保持流畅显示。


### 4. 通信模块

#### 前后端交互：

前后端交互的实现主要基于Flask框架作为后端和Jinjia2作为前端技术栈。后端Flask提供API接口以及视频流服务，处理与客户端的交互逻辑。前端HTML页面加载时会调用”/video_start”接口，通过Response返回视频流，实现实时视频效果。

前后端交互的核心体现在按钮点击事件上。当用户点击按钮时，调用sendMessage()函数。该函数使用API向后端发送HTTP POST请求，URL为”/api/button-click”，后端对应的路由为”@app.route('/api/button-click', methods=['POST'])”，通过request.get_json()解析前端发送的数据，提取按钮名称`message`。根据`message`值执行不同逻辑：如果是“A”，调用`automatic_mode`函数启动两个线程，分别用于车道保持和目标检测；如果是“C”，通过清除事件标志停止用于车道保持和目标检测的两个后台线程；如果是其他值，直接调用send_command()发送指令到Arduino。

为了确保交互实时性和安全性，后端的线程逻辑采用事件控制和时间间隔限制，根据时间戳限制发送频率，确保车辆指令不会过于频繁或冲突。

#### 串口通信：

串口通信用于后端与硬件通信，使用的波特率为115200，本次实践**未使用**模拟接口。在系统运行时，会尝试连接串口，串口连接成功后，可以通过send_command()向Arduino发送指令。在程序运行过程中串口通信存在通信中断的问题，所以在send_command()中添加判断串口通信状态的代码，当通信中断时，系统会尝试重新连接串口，再发送指令。

### 5. 算法模块

算法模块主要包括**车道线检测**、**车道偏移计算**、**目标检测**、**自动避障**和**多线程通信**五个功能，是智能驾驶辅助系统的核心。

**车道线检测**使用Canny边缘检测算法提取每一帧图像的边缘信息，以突出可能的车道线轮廓，再定义一个四边形区域（ROI），以过滤掉边缘检测图像中非车道部分的边缘信息，从而集中检测车辆前方的车道线，得到一个裁剪后的感兴趣区域。使用霍夫变换检测线段，从感兴趣区域中提取直线段，通过判断线段的斜率（slope），将车道线分为左车道和右车道。最后通过维护车道线的历史记录，对检测到的车道线进行平滑处理。提高了车道检测的稳定性和连续性。

```python
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
```

**车道偏移计算**通过获取车道检测后返回的车道线信息，发出相应的指令。偏移计算分为三个决策条件：如果左右车道线都存在，则按照两条车道线的四个顶点计算车道线的中心位置，通过与小车中心坐标相减得出小车的偏移量，如果偏移量的绝对值超过阈值，则通过指令对小车运动方向进行微调；如果仅检测到右车道线，则小车发送左转的指令；如果仅检测到左车道线，则小车发送右转的指令。第一个决策用于正常的车道保持，适用于车道线清晰容易提取车道线的路段，后面两个决策用于车道线不清晰的路段或弯道，当小车遭遇左转弯道时，左侧的车道线会比右侧的车道线提前从摄像头范围内消失，右侧同理，三个决策组合可以提高小车在特殊路段行驶的稳定性。

```python
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
```

**目标检测**通过加载Yolo目标检测模型，在视频画面中显示目标检测结果，并将目标名称作为发送不同指令的判断条件。训练目标检测模型使用的是左转和右转两种交通标志，数据集图像共566张，标签为“left”和“right”。

**自动避障**属于硬件代码，使用正面的超声波传感器测量小车和前方障碍物的距离，当距离达到阈值后，触发避障策略，小车会根据侧面的超声波传感器与其范围内的障碍物的距离判断小车避障的方向。如果小车右侧存在障碍物，则小车原地左转向后，直行一段距离再原地右转恢复行进方向，不存在障碍物则小车原地右转向后，直行一段距离再原地左转恢复行进方向。

```C
// 测量前方超声波距离
float measureDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  // 计算声波往返时间
  float duration = pulseIn(echoPin, HIGH);

  // 转换为厘米
  float distance = (duration * 0.034) / 2;
  return distance;
}

// 避障策略
void avoidObstacle(float distanceR) {
  stopMotors();
  delay(1000); // 停止一段时间

  if (distanceR > 0 && distanceR < 30) {
    turnLeft();
    delay(400);
    moveForward();
    delay(400);
    turnRight();
    delay(500);
  } else {
    turnRight();
    delay(500);
    moveForward();
    delay(400);
    turnLeft();
    delay(400);
  }
  stopMotors();
}
```

**多线程通信**的设置是为了避免多种算法发送指令时不发生冲突，同时为了避免视频画面卡顿。确保车道保持与目标检测结果能够被其他线程或模块访问，为车道保持和目标检测功能提供了数据实时性和共享性。整个系统包含四个线程，两个线程用于以安全队列的形式保存车道保持与目标检测结果，这两个线程会在程序运行时直接启动；剩余两个线程用于实时监控前两个线程队列中存放的数据，根据数据的内容向Arduino发送通信指令，这两个线程的启动或结束由前端按钮控制，打开线程进入自动驾驶模式，关闭线程则进入手动驾驶模式。


<br/>

## 三、系统外观和结构
智能驾驶辅助系统实现了基本的车道检测、车道保持、前向碰撞预警、自动紧急避障、目标检测等功能，小车外观如图3-1，系统运行时的Web界面和终端如图 3-2。

<p align="center"><img src="https://github.com/user-attachments/assets/73638baa-a497-4fd1-b0d8-fd5c7d68d8d4" alt="图 3-1 小车外观"></p>

<p align="center"><img src="https://github.com/user-attachments/assets/7ed479a8-c993-4bc0-9e6d-608e65dfc0f2" alt="图 3-2 系统运行时的Web界面和终端"></p>

<br/>

## 备注：

<br/>

## 参考文献：
[1] 王超,杨福康.基于自主导航与避障系统的智能小车设计[J].信息技术与信息化,2022,(8):217-220.  
[2] 李兴鑫,王飞,李楠鑫.基于深度学习的智能小车辅助驾驶系统开发设计[C].//2023年上半年度学术研究会.2023:74-89.  
[3] 杨阳.基于多线激光雷达的建图与定位技术研究[D].天津工业大学,2023.10.27357/d.cnki. gtgyu.2023.000424.  
[4] 胡名波.自主巡逻车视觉导航车道保持算法研究[D].中国民航大学,2016.



