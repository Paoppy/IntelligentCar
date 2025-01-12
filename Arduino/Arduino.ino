#define enLPin  5
#define in1Pin  6
#define in2Pin  7
#define in3Pin  9
#define in4Pin  10
#define enRPin  11
#define trigPin 12
#define echoPin 13
#define trigPinR 2
#define echoPinR 3

#define pinMiddle 8
#define pinS 4

// 超声波距离阈值 (厘米)
#define OBSTACLE_DISTANCE 30

#define Speed 150

float distance;
float distanceR;
String command = "";

// 函数声明
void SoftSerialCtrl();
void moveForward();
void moveBackward();
void stopMotors();
void turnLeft();
void turnRight();
float measureDistance();
float measureDistanceR();
void avoidObstacle();
void updateLED(float distance);

void setup() {
  Serial.begin(115200);
  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);
  pinMode(in3Pin, OUTPUT);
  pinMode(in4Pin, OUTPUT);
  pinMode(enLPin, OUTPUT);
  pinMode(enRPin, OUTPUT);
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(trigPinR, OUTPUT);
  pinMode(echoPinR, INPUT);
  pinMode(pinMiddle, OUTPUT);
  pinMode(pinS, OUTPUT);
}

void loop() {
  SoftSerialCtrl();
  distance = measureDistance();
  updateLED(distance);
}

void SoftSerialCtrl() {
  if (Serial.available() > 0) {
    command = Serial.readStringUntil('&'); // 读取到&为止
    command.trim(); // 清除多余空白符
    if (command == "L") {
      turnLeft();
      delay(400);
      stopMotors();
    } else if (command == "R") {
      turnRight();
      delay(400);
      stopMotors();
    }else if (command == "F") {
      moveForward();
      delay(400);
      stopMotors();
    } else if (command == "B") {
      moveBackward();
      delay(600);
      stopMotors();
    } else if (command == "AL") {
      moveForward();
      delay(200);
      turnLeft();
      delay(200);
      stopMotors();
    } else if (command == "AR") {
      moveForward();
      delay(300);
      turnRight();
      delay(300);
      stopMotors();
    } else if (command == "S") {
      stopMotors();
    }
    command = ""; // 清空命令缓存
  }
}

// 电机控制函数
void moveForward() {
  digitalWrite(enLPin, Speed);
  digitalWrite(in1Pin, LOW);
  digitalWrite(in2Pin, HIGH);

  digitalWrite(enRPin, Speed);
  digitalWrite(in3Pin, HIGH);
  digitalWrite(in4Pin, LOW);
}

void moveBackward() {
  digitalWrite(enLPin, Speed);
  digitalWrite(in1Pin, HIGH);
  digitalWrite(in2Pin, LOW);

  digitalWrite(enRPin, Speed);
  digitalWrite(in3Pin, LOW);
  digitalWrite(in4Pin, HIGH);
}

void stopMotors() {
  digitalWrite(enLPin, LOW);
  digitalWrite(enRPin, LOW);
}

void turnLeft() {
  digitalWrite(enLPin, Speed);
  digitalWrite(in1Pin, HIGH);
  digitalWrite(in2Pin, LOW);

  digitalWrite(enRPin, Speed);
  digitalWrite(in3Pin, HIGH);
  digitalWrite(in4Pin, LOW);
}

void turnRight() {
  digitalWrite(enLPin, Speed);
  digitalWrite(in1Pin, LOW);
  digitalWrite(in2Pin, HIGH);

  digitalWrite(enRPin, Speed);
  digitalWrite(in3Pin, LOW);
  digitalWrite(in4Pin, HIGH);
}

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

// 测量右侧超声波距离
float measureDistanceR() {
  digitalWrite(trigPinR, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPinR, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPinR, LOW);

  // 计算声波往返时间
  float duration = pulseIn(echoPinR, HIGH);

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

void updateLED(float distance) {
  if (distance > 0 && distance < OBSTACLE_DISTANCE) {
    // 障碍物在阈值以内，亮红灯
    digitalWrite(pinMiddle, HIGH); // 红灯亮
    digitalWrite(pinS, LOW);       // 绿灯灭
    // 测量右侧距离
    distanceR = measureDistanceR();
    avoidObstacle(distanceR);
  } else {
    // 障碍物在阈值以外，亮绿灯
    digitalWrite(pinMiddle, LOW);  // 红灯灭
    digitalWrite(pinS, HIGH);      // 绿灯亮
  }
}
