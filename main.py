from djitellopy import Tello
import cv2 as cv
import json

RED = (0, 0, 255)  # too close
GREEN = (0, 255, 0)  # good distance
BLUE = (255, 0, 0)  # too far
YELLOW = (0, 255, 255)


def loadConfigData():
    with open('resources/config.json') as f:
        return json.load(f)


def getForBackVelocity(data, area):
    if area > data['minDistance']:
        for_back_velocity = -data['forBackSpeed']
        color = RED
    elif area < data['maxDistance']:
        for_back_velocity = data['forBackSpeed']
        color = BLUE
    else:
        for_back_velocity = 0
        color = GREEN
    return for_back_velocity, color


def getUpDownYawVelocity(data, centerPoint, frameCenterPoint):
    up_down_velocity, yaw_velocity = 0, 0
    safeArea = data['minSides']
    diffX = centerPoint[0] - frameCenterPoint[0]
    diffY = centerPoint[1] - frameCenterPoint[1]
    if abs(diffY) > safeArea:
        if diffY > 0:
            # go down
            up_down_velocity = -data['upDownSpeed'] // 2
        else:
            # go up
            up_down_velocity = data['upDownSpeed'] // 2
    if abs(diffX) > safeArea:
        if diffX > 0:
            # go right
            yaw_velocity = data['yawSpeed']
        else:
            # go left
            yaw_velocity = -data['yawSpeed']

    return up_down_velocity, yaw_velocity


if __name__ == '__main__':
    # connect to drone
    drone = Tello()
    drone.connect()
    print('Battery = ', drone.get_battery(), '%')

    # reset all parameters and speed
    drone.for_back_velocity = 0
    drone.left_right_velocity = 0
    drone.up_down_velocity = 0
    drone.yaw_velocity = 0
    drone.speed = 0

    # start video stream
    drone.streamoff()
    drone.streamon()

    drone.takeoff()

    # load data and parameters
    data = loadConfigData()

    # load cascade file
    face_cascade = cv.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
    cap = cv.VideoCapture(0)
    left_right_velocity = 0
    while True:
        # direction parameters
        for_back_velocity = 0
        up_down_velocity = 0
        yaw_velocity = 0

        img = drone.get_frame_read().frame

        # change image to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # extract all faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 11)

        # draw safe area circle
        height, width, channels = img.shape
        frameCenterPoint = ((width // 2), (height // 2))
        img = cv.circle(img, frameCenterPoint, data['minSides'], YELLOW, 2)

        # Draw red rectangle around the faces
        for (x, y, w, h) in faces:
            area = w * h
            centerPoint = (x + (w // 2), y + (h // 2))

            for_back_velocity, color = getForBackVelocity(data, area)
            up_down_velocity, yaw_velocity = getUpDownYawVelocity(data, centerPoint, frameCenterPoint)

            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, 'face', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            img = cv.circle(img, centerPoint, 2, GREEN, -1)
        drone.send_rc_control(left_right_velocity, for_back_velocity, up_down_velocity, yaw_velocity)

        # Display the output
        cv.imshow("Drone Camera", img)

        # press 'q' to quit
        if cv.waitKey(1) & 0xFF == ord('q'):
            drone.land()
            break
