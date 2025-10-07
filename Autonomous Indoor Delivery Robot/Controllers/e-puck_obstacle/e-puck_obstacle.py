from controller import Robot

# --- constants ---
TIME_STEP = 64        # must match world basicTimeStep
MAX_SPEED = 6.28      # e-puck motor max velocity
THRESHOLD = 80.0      # raw sensor value to detect obstacle (≈0–4096)

# --- init robot ---
robot = Robot()

# Motors
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")
left_motor.setPosition(float("inf"))
right_motor.setPosition(float("inf"))
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# Distance sensors
ps_names = ["ps0","ps1","ps2","ps3","ps4","ps5","ps6","ps7"]
ps = []
for name in ps_names:
    s = robot.getDevice(name)
    s.enable(TIME_STEP)
    ps.append(s)

# --- main loop ---
while robot.step(TIME_STEP) != -1:
    # read sensors
    values = [s.getValue() for s in ps]

    # front sensors (ps0 right-front, ps7 left-front)
    left_obst  = values[7] > THRESHOLD
    right_obst = values[0] > THRESHOLD

    # default: go straight
    left_speed  = MAX_SPEED
    right_speed = MAX_SPEED

    # turn away if obstacle detected
    if left_obst:
        # obstacle on left → turn right
        left_speed  = MAX_SPEED
        right_speed = -MAX_SPEED * 0.5
    elif right_obst:
        # obstacle on right → turn left
        left_speed  = -MAX_SPEED * 0.5
        right_speed = MAX_SPEED

    left_motor.setVelocity(left_speed)
    right_motor.setVelocity(right_speed)
