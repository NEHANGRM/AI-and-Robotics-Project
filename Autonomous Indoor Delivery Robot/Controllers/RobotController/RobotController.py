from controller import Robot
import cv2
import numpy as np

# -------- ROBOT INIT --------
TIME_STEP = 64
MAX_SPEED = 19.1
GRIPPER_MOTOR_MAX_SPEED = 0.5

robot = Robot()

# Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

arm_motor = robot.getDevice("horizontal_motor")
arm_motor.setVelocity(1.0)

left_finger = robot.getDevice("finger_motor::left")
right_finger = robot.getDevice("finger_motor::right")
left_finger.setVelocity(GRIPPER_MOTOR_MAX_SPEED)
right_finger.setVelocity(GRIPPER_MOTOR_MAX_SPEED)
left_finger.setAvailableTorque(0.5)
right_finger.setAvailableTorque(0.5)

camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

front_range = robot.getDevice("distance sensor")
front_range.enable(TIME_STEP)

oa_sensors = []
for i in range(9):
    s = robot.getDevice(f"ds{i}")
    s.enable(TIME_STEP)
    oa_sensors.append(s)

# -------- BASIC MOVEMENTS --------
def moveForward(speed=0.5):  
    left_motor.setVelocity(speed*MAX_SPEED)
    right_motor.setVelocity(speed*MAX_SPEED)

def moveBackward(speed=0.3): 
    left_motor.setVelocity(-speed*MAX_SPEED)
    right_motor.setVelocity(-speed*MAX_SPEED)

def stop():                  
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

def turnLeft(speed=0.4):      
    left_motor.setVelocity(-speed*MAX_SPEED)
    right_motor.setVelocity(speed*MAX_SPEED)

def turnRight(speed=0.4):     
    left_motor.setVelocity(speed*MAX_SPEED)
    right_motor.setVelocity(-speed*MAX_SPEED)

def extendArm(position=0.10): 
    arm_motor.setPosition(position)

def retractArm(position=-0.10): 
    arm_motor.setPosition(position)

def openGripper():            
    left_finger.setPosition(-0.5)
    right_finger.setPosition(-0.5)

def waitSteps(n):
    for _ in range(n):
        if robot.step(TIME_STEP) == -1: break

def closeGripper():
    grip_target = 0.3
    left_finger.setPosition(grip_target)
    right_finger.setPosition(grip_target)
    waitSteps(40)

def pickup(extend_pos=-3.3, retract_pos=-0.5):
    openGripper(); waitSteps(20)
    extendArm(extend_pos); waitSteps(50)
    closeGripper(); waitSteps(30)
    retractArm(retract_pos); waitSteps(40)

def putdown(extend_pos=-3.3, retract_pos=-0.5):
    extendArm(extend_pos); waitSteps(30)
    openGripper(); waitSteps(20)
    retractArm(retract_pos); waitSteps(40)

# -------- STATE MACHINE --------
STATE = "SEARCH_OBJECT"
PREV_STATE = None
carrying_object = False
carried_color = None
CLOSE_THRESHOLD = 180000
BOX_HALF_WIDTH = 80
PICKUP_DISTANCE_MIN = 730
PICKUP_DISTANCE_MAX = 760

OA_LINGER_STEPS = 20
oa_cooldown = 0
oa_counter = 0
WALL_THRESHOLD = 10           # timesteps before switching to wall follow
CLEAR_EXIT_THRESHOLD = 10     # clear timesteps before exiting OA

clear_counter = 0
wall_lost = False
wall_side = "LEFT"
lost_count = 0

# desired distance from wall
TARGET_DIST = 500

def change_state(new_state):
    global STATE
    if STATE != new_state:
        print(f"[STATE] {STATE} -> {new_state}")
        STATE = new_state

def wall_follow(l1,l2):
    print(l1," ",l2)
    if l1>1000 and l2<50:
        left_motor.setVelocity(MAX_SPEED*0.5)
        right_motor.setVelocity(MAX_SPEED*0.1)
    elif l1>500 or l2>500:
        left_motor.setVelocity(MAX_SPEED*0.5)
        right_motor.setVelocity(-MAX_SPEED*0.5)
    elif l1<30 or l1<30:
        left_motor.setVelocity(-MAX_SPEED*0.5)
        right_motor.setVelocity(MAX_SPEED*0.5)
    else:
        left_motor.setVelocity(MAX_SPEED*0.5)
        right_motor.setVelocity(MAX_SPEED*0.5)


while robot.step(TIME_STEP) != -1:
    distance = front_range.getValue()
    oa_vals = [s.getValue() for s in oa_sensors]
    L2,L3,F4,F5,R6,R7 = oa_vals[1],oa_vals[2],oa_vals[3],oa_vals[4],oa_vals[5],oa_vals[6]
    front_blocked = F4 > 500 or F5 > 500
    left_blocked  = L2 > 500 or L3 > 500
    right_blocked = R6 > 500 or R7 > 500

    # -------- Camera Processing --------
    image = camera.getImage()
    w = camera.getWidth()
    h = camera.getHeight()
    img_array = np.frombuffer(image, np.uint8).reshape((h, w, 4))
    bgr_image = img_array[:, :, :3]
    bgr_image = cv2.flip(bgr_image, 1)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    cx_center = w // 2
    box_left = cx_center - BOX_HALF_WIDTH
    box_right = cx_center + BOX_HALF_WIDTH

    def detect(mask, min_area=200):
        c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if c:
            cnt = max(c, key=cv2.contourArea)
            if cv2.contourArea(cnt) > min_area:
                return cv2.boundingRect(cnt), cv2.contourArea(cnt)
        return None,0

    green,_ = detect(cv2.inRange(hsv, (40,40,40),(80,255,255)))
    blue,_  = detect(cv2.inRange(hsv, (100,150,50),(130,255,255)))
    red,area_red = detect(cv2.inRange(hsv, (0,150,150),(10,255,255)),300)
    yellow,area_yellow = detect(cv2.inRange(hsv, (20,150,150),(30,255,255)),300)

    if not carrying_object:
        color_found = bool(green or blue)
    else:
        if carried_color=="GREEN": color_found = bool(red)
        elif carried_color=="BLUE": color_found = bool(yellow)
        else: color_found = False

    # -------- Obstacle / Wall detection --------
    if front_blocked or left_blocked or right_blocked:
        if STATE not in ["OBSTACLE_AVOIDANCE","WALL_FOLLOW"]:
            PREV_STATE = STATE
            change_state("OBSTACLE_AVOIDANCE")
            oa_counter = 0
            clear_counter = 0
        if STATE == "OBSTACLE_AVOIDANCE":
            oa_counter += 1
            if oa_counter >= WALL_THRESHOLD:
                change_state("WALL_FOLLOW")
    else:
        oa_counter = 0

    # -------- STATE MACHINE --------
    if STATE == "OBSTACLE_AVOIDANCE":
        if not front_blocked and not left_blocked and not right_blocked:
            clear_counter += 1
            if clear_counter >= CLEAR_EXIT_THRESHOLD:
                print("[OA] Path clear -> exiting to search state.")
                change_state(PREV_STATE)
                continue
        else:
            clear_counter = 0

        left_speed = MAX_SPEED*0.5
        right_speed = MAX_SPEED*0.5
        if front_blocked or left_blocked: right_speed = -MAX_SPEED*0.3
        elif right_blocked: left_speed = -MAX_SPEED*0.4
        left_motor.setVelocity(left_speed)
        right_motor.setVelocity(right_speed)

    elif STATE == "WALL_FOLLOW":
        if color_found:
            print("[WALL] Target colour detected. Exiting wall follow.")
            change_state(PREV_STATE)
            continue
        wall_follow(L2,L3)

    elif STATE == "SEARCH_OBJECT":
        if not carrying_object:
            target,color=None,None
            if green: target,color=green,"GREEN"
            elif blue: target,color=blue,"BLUE"
            if target:
                x,y,wid,hei = target
                cx = x + wid//2
                area = wid*hei
                if cx < box_left: turnRight(0.2)
                elif cx > box_right: turnLeft(0.2)
                else:
                    if area < 8000: moveForward(0.3)
                    else:
                        if distance < PICKUP_DISTANCE_MIN: moveBackward(0.2)
                        elif distance > PICKUP_DISTANCE_MAX: moveForward(0.2)
                        else:
                            stop(); pickup()
                            carrying_object=True
                            carried_color=color
                            change_state("SEARCH_BIN")
            else: turnLeft(0.2)

    elif STATE == "SEARCH_BIN":
        target,area = None,0
        if carried_color=="GREEN" and red: target,area=red,area_red
        elif carried_color=="BLUE" and yellow: target,area=yellow,area_yellow
        if target:
            x,y,wid,hei = target
            cx = x + wid//2
            if cx < box_left: turnRight(0.2)
            elif cx > box_right: turnLeft(0.2)
            else:
                if area < CLOSE_THRESHOLD: moveForward(0.3)
                else:
                    stop(); putdown()
                    carrying_object=False
                    carried_color=None
                    change_state("SEARCH_OBJECT")
        else: turnLeft(0.2)

    cv2.imshow("Robot Camera", cv2.resize(bgr_image,(640,480)))
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cv2.destroyAllWindows()
