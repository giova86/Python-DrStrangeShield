# libraries
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, draw_landmarks, draw_landmarks_custom, draw_limit_rh, draw_limit_lh, check_detection, get_center_lh,get_center_rh

mp_holistic = mp.solutions.holistic
#mp_drawing = mp.solutions.drawing_utils

# cap = cv2.VideoCapture('video.mp4')
cap = cv2.VideoCapture(1)
shield = cv2.VideoCapture("shield.mp4")

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        ret_shield, frame_shield = shield.read()
        if ret_shield:
            pass
        else:
            shield = cv2.VideoCapture("shield.mp4")
            ret_shield, frame_shield = shield.read()

        frame_shield = cv2.resize(frame_shield, (320,180))

        # make detection
        image, results = mediapipe_detection(frame, holistic)



        hsv = cv2.cvtColor(frame_shield, cv2.COLOR_BGR2HSV)
        black_screen = np.array([0,0,0])
        mask = cv2.inRange(frame_shield,black_screen,black_screen)
        res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
        res = frame_shield - res
        alpha=1


        xMinL, xMaxL, yMinL, yMaxL = get_center_lh(frame, results)
        if xMinL:
            xc_lh = (xMaxL+xMinL)/2
            yc_lh = (yMaxL+yMinL)/2
            xc_lh = int(width*xc_lh)
            yc_lh = int(height*yc_lh)

            l_width_shield = int(width*(xMaxL-xMinL)/2*2.5)
            l_height_shield = int(height*(yMaxL-yMinL)/2*2.5)

            res2 = cv2.resize(res, (l_width_shield*2, l_height_shield*2))

            start_h = 0
            start_w = 0
            stop_h = l_height_shield*2
            stop_w = l_width_shield*2

            f_start_h = yc_lh-l_height_shield
            f_stop_h = yc_lh+l_height_shield
            f_start_w = xc_lh-l_width_shield
            f_stop_w = xc_lh+l_width_shield

            if yc_lh-l_height_shield < 0:
                start_h = -yc_lh+l_height_shield
                f_start_h = 0
            if yc_lh+l_height_shield > height:
                stop_h = l_height_shield + height - yc_lh
                f_stop_h = height
            if xc_lh-l_width_shield < 0:
                start_w = -xc_lh+l_width_shield
                f_start_w = 0
            if xc_lh+l_width_shield > width:
                stop_w = l_width_shield + width - xc_lh
                f_stop_w = width

            res2 = res2[start_h:stop_h, start_w:stop_w,:]
            frame_shield =cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res2, 1,1, frame)
            frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

        xMinR, xMaxR, yMinR, yMaxR = get_center_rh(frame, results)
        if xMinR:
            xc_rh = (xMaxR+xMinR)/2
            yc_rh = (yMaxR+yMinR)/2
            xc_rh = int(width*xc_rh)
            yc_rh = int(height*yc_rh)

            r_width_shield = int(width*(xMaxR-xMinR)/2*2.5)
            r_height_shield = int(height*(yMaxR-yMinR)/2*2.5)

            res3 = cv2.resize(res, (r_width_shield*2, r_height_shield*2))

            start_h = 0
            start_w = 0
            stop_h = r_height_shield*2
            stop_w = r_width_shield*2

            f_start_h = yc_rh-r_height_shield
            f_stop_h = yc_rh+r_height_shield
            f_start_w = xc_rh-r_width_shield
            f_stop_w = xc_rh+r_width_shield

            if yc_rh-r_height_shield < 0:
                start_h = -yc_rh+r_height_shield
                f_start_h = 0
            if yc_rh+r_height_shield > height:
                stop_h = r_height_shield + height - yc_rh
                f_stop_h = height
            if xc_rh-r_width_shield < 0:
                start_w = -xc_rh+r_width_shield
                f_start_w = 0
            if xc_rh+r_width_shield > width:
                stop_w = r_width_shield + width - xc_rh
                f_stop_w = width

            res3 = res3[start_h:stop_h, start_w:stop_w,:]
            frame_shield =cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res3, 1,1, frame)
            frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield



        cv2.imshow('Dr. Strange shields', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
