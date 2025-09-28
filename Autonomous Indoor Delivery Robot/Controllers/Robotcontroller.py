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

# Camera
camera = robot.getDevice("camera")
camera.enable(TIME_STEP)

# Distance Sensor
ds = robot.getDevice("distance sensor")
ds.enable(TIME_STEP)

# -------- BASIC MOVEMENTS --------
def moveForward(speed=0.5):
    left_motor.setVelocity(speed * MAX_SPEED)
    right_motor.setVelocity(speed * MAX_SPEED)

def moveBackward(speed=0.3):
    left_motor.setVelocity(-speed * MAX_SPEED)
    right_motor.setVelocity(-speed * MAX_SPEED)

def stop():
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)

def turnLeft(speed=0.3):
    left_motor.setVelocity(-speed * MAX_SPEED)
    right_motor.setVelocity(speed * MAX_SPEED)

def turnRight(speed=0.3):
    left_motor.setVelocity(speed * MAX_SPEED)
    right_motor.setVelocity(-speed * MAX_SPEED)

def extendArm(position=0.10):
    arm_motor.setPosition(position)

def retractArm(position=-0.10):
    arm_motor.setPosition(position)

def openGripper():
    left_finger.setPosition(-0.5)
    right_finger.setPosition(-0.5)

def closeGripper():
    """Adaptive close that maintains grip force"""
    print("Closing gripper with maintained contact")
    grip_target = 0.42
    left_finger.setPosition(grip_target)
    right_finger.setPosition(grip_target)
    waitSteps(40)
    print("Gripper holding object")

# -------- HIGH LEVEL ACTIONS --------
def pickup(extend_pos=-3.3, retract_pos=-0.5):
    openGripper()
    waitSteps(20)
    extendArm(extend_pos)
    waitSteps(50)
    closeGripper()
    waitSteps(30)
    retractArm(retract_pos)
    waitSteps(40)

def putdown(extend_pos=-3.3, retract_pos=-0.5):
    extendArm(extend_pos)
    waitSteps(30)
    openGripper()
    waitSteps(20)
    retractArm(retract_pos)
    waitSteps(40)

# -------- HELPER --------
def waitSteps(n):
    for _ in range(n):
        if robot.step(TIME_STEP) == -1:
            break
        image = camera.getImage()
        width = camera.getWidth()
        height = camera.getHeight()
        img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
        bgr_image = img_array[:, :, :3]
        bgr_image = cv2.flip(bgr_image, 1)
        cv2.imshow("Robot Camera", cv2.resize(bgr_image, (640, 480)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# -------- STATE MACHINE --------
STATE = "SEARCH_OBJECT"
carrying_object = False
carried_color = None
CLOSE_THRESHOLD = 80000

# Central box parameters
BOX_HALF_WIDTH = 80

# Distance thresholds (tune after testing!)
PICKUP_DISTANCE_MIN = 730   # too close → back up
PICKUP_DISTANCE_MAX = 760   # too far → move forward

while robot.step(TIME_STEP) != -1:
    # Get distance
    distance = ds.getValue()

    # Get camera frame
    image = camera.getImage()
    width = camera.getWidth()
    height = camera.getHeight()
    img_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    bgr_image = img_array[:, :, :3]
    bgr_image = cv2.flip(bgr_image, 1)
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)

    # Central rectangle (dead zone)
    cx_center = width // 2
    box_left = cx_center - BOX_HALF_WIDTH
    box_right = cx_center + BOX_HALF_WIDTH
    cv2.rectangle(bgr_image, (box_left, 0), (box_right, height), (200, 200, 200), 2)

    # --- Object detection ---
    target_green, target_blue = None, None
    # Green
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_green:
        cnt = max(contours_green, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 200:
            target_green = cv2.boundingRect(cnt)
            x, y, w, h = target_green
            cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Blue
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_blue:
        cnt = max(contours_blue, key=cv2.contourArea)
        if cv2.contourArea(cnt) > 200:
            target_blue = cv2.boundingRect(cnt)
            x, y, w, h = target_blue
            cv2.rectangle(bgr_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # --- Bin detection ---
    target_red, target_yellow = None, None
    curr_area_red, curr_area_yellow = 0, 0

    # Red bin
    lower_red = np.array([0, 150, 150])
    upper_red = np.array([10, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_red:
        cnt = max(contours_red, key=cv2.contourArea)
        curr_area_red = cv2.contourArea(cnt)
        if curr_area_red > 300:
            target_red = cv2.boundingRect(cnt)

    # Yellow bin
    lower_yellow = np.array([20, 150, 150])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    contours_yellow, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours_yellow:
        cnt = max(contours_yellow, key=cv2.contourArea)
        curr_area_yellow = cv2.contourArea(cnt)
        if curr_area_yellow > 300:
            target_yellow = cv2.boundingRect(cnt)

    # --- STATE MACHINE ---
    if STATE == "SEARCH_OBJECT":
        if not carrying_object:
            target = None
            color = None
            if target_green:
                target = target_green
                color = "GREEN"
            elif target_blue:
                target = target_blue
                color = "BLUE"

            if target:
                x, y, w, h = target
                cx = x + w // 2
                area = w * h

                # --- keep centroid in central box ---
                if cx < box_left:
                    turnRight(0.2)
                elif cx > box_right:
                    turnLeft(0.2)
                else:
                    if area < 8000:
                        moveForward(0.3)
                    else:
                        # NEW CONDITION: check distance range
                        if distance < PICKUP_DISTANCE_MIN:
                            print("Too close! Backing up...")
                            moveBackward(0.2)
                        elif distance > PICKUP_DISTANCE_MAX:
                            print("Too far! Moving closer...")
                            moveForward(0.2)
                        else:
                            print("Perfect distance. Picking up.")
                            stop()
                            pickup()
                            carrying_object = True
                            carried_color = color
                            STATE = "SEARCH_BIN"
            else:
                turnLeft(0.2)

    elif STATE == "SEARCH_BIN":
        target = None
        curr_area = 0
        if carried_color == "GREEN" and target_red:
            target = target_red
            curr_area = curr_area_red
        elif carried_color == "BLUE" and target_yellow:
            target = target_yellow
            curr_area = curr_area_yellow

        if target:
            x, y, w, h = target
            cx = x + w // 2

            # Keep bin centroid in central box
            if cx < box_left:
                turnRight(0.2)
            elif cx > box_right:
                turnLeft(0.2)
            else:
                # move forward based on size
                if curr_area < CLOSE_THRESHOLD:
                    moveForward(0.3)
                else:
                    stop()
                    putdown()
                    carrying_object = False
                    carried_color = None
                    STATE = "SEARCH_OBJECT"
        else:
            turnLeft(0.2)

    # Show feed
    cv2.imshow("Robot Camera", cv2.resize(bgr_image, (640, 480)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()