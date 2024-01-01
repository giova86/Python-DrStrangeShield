# libraries
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, get_center_lh,get_center_rh, points_detection, points_detection_hands
from argparse import ArgumentParser
import pickle
from datetime import datetime, timedelta
import time


# - INPUT PARAMETERS ------------------------------- #
parser = ArgumentParser()
parser.add_argument("-m", "--model", dest="ML_model", default='models/model_svm.sav',
                    help="PATH of model FILE.", metavar="FILE")
parser.add_argument("-t", "--threshold", dest="threshold_prediction", default=0.9, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-dc", "--det_conf", dest="min_detection_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-tc", "--trk_conf", dest="min_tracking_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-c", "--camera_id", dest="camera", default=1, type=int,
                    help="ID of the camera. An integer between 0 and N. Default is 1")
parser.add_argument("-s", "--shield", dest="shield_video", default='effects/shield.mp4',
                    help="PATH of the video FILE.", metavar="FILE")

args = parser.parse_args()
# -------------------------------------------------- #

current_directory = os.path.dirname(os.path.realpath(__file__)).replace("\\", "/")

# --- start camera ---
print(f'Setting Camera: {args.camera}')
cap = cv2.VideoCapture(args.camera)
print('-- DONE')
time.sleep(5)

# --- load svm model ---
print('Loading Model')
model = pickle.load(open(current_directory + '/' + args.ML_model, 'rb'))
labels = np.array(model.classes_)  # put the entire alphabet in the future

KEY_1 = False
KEY_2 = False
KEY_3 = False
SHIELDS = False

scale = 1.5
print('-- DONE')

mp_holistic = mp.solutions.holistic
#mp_drawing = mp.solutions.drawing_utils

# --- load shield video ---
shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)

# --- get width and height from camera
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

black_screen = np.array([0,0,0])

with mp_holistic.Holistic(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5,
                          model_complexity=0) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        print(SHIELDS, '-', KEY_1, KEY_2, KEY_3)

        ret_shield, frame_shield = shield.read()
        if not ret_shield:
            shield = cv2.VideoCapture(current_directory + '/' + args.shield_video)
            ret_shield, frame_shield = shield.read()

        # ret_shield_effect, frame_shield_effect = shield_effect.read()
        # if ret_shield_effect:
        #     frame_shield_effect = cv2.resize(frame_shield_effect, (1280,720), interpolation = cv2.INTER_AREA)
        #     pass
        # else:
        #     shield_effect = cv2.VideoCapture('shield_effect.mp4')
        #     ret_shield_effect, frame_shield_effect = shield.read()
        #     frame_shield_effect = cv2.resize(frame_shield_effect, (1280,720), interpolation = cv2.INTER_AREA)

        # make detection
        frame, results = mediapipe_detection(frame, holistic)
        xMinL, xMaxL, yMinL, yMaxL = get_center_lh(frame, results)
        xMinR, xMaxR, yMinR, yMaxR = get_center_rh(frame, results)

        #hsv = cv2.cvtColor(frame_shield, cv2.COLOR_BGR2HSV)
        # black_screen = np.array([0,0,0])
        mask = cv2.inRange(frame_shield,black_screen,black_screen)
        res = cv2.bitwise_and(frame_shield, frame_shield, mask=mask)
        res = frame_shield - res
        alpha=1

        if SHIELDS and xMinL:

            xc_lh = (xMaxL+xMinL)/2
            yc_lh = (yMaxL+yMinL)/2
            xc_lh = int(width*xc_lh)
            yc_lh = int(height*yc_lh)

            l_width_shield = int(width*(xMaxL-xMinL)/2*3.5*scale)
            l_height_shield = int(height*(yMaxL-yMinL)/2*2*scale)

            res2 = cv2.resize(res, (l_width_shield*2, l_height_shield*2))
            # res_effect = cv2.resize(res_effect, (l_width_shield*2, l_height_shield*2))

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
            # res_effect = res_effect[start_h:stop_h, start_w:stop_w,:]

            frame_shield =cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res2, 1,1, frame)
            frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

            # frame_shield =cv2.addWeighted(frame[f_start_h:f_stop_h,f_start_w:f_stop_w], alpha, res_effect, 1,1, frame)
            # frame[f_start_h:f_stop_h,f_start_w:f_stop_w] = frame_shield

        if SHIELDS and xMinR:

            xc_rh = (xMaxR+xMinR)/2
            yc_rh = (yMaxR+yMinR)/2
            xc_rh = int(width*xc_rh)
            yc_rh = int(height*yc_rh)

            r_width_shield = int(width*(xMaxR-xMinR)/2*3.5*scale)
            r_height_shield = int(height*(yMaxR-yMinR)/2*2*scale)

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

        if xMinL and xMinR and SHIELDS:
            prediction = model.predict(np.array([points_detection_hands(results)]))[0]
            pred_prob = np.max(model.predict_proba(np.array([points_detection_hands(results)])))

            if (prediction == 'key_4') and (pred_prob > 0.85):
                KEY_1 = False
                KEY_2 = False
                KEY_3 = False
                SHIELDS = False


        elif xMinL and xMinR and (not SHIELDS):
            prediction = model.predict(np.array([points_detection_hands(results)]))[0]
            pred_prob = np.max(model.predict_proba(np.array([points_detection_hands(results)])))

            if (prediction == 'key_1') and (pred_prob > 0.85):
                t1 = datetime.now()
                KEY_1 = True
            elif (prediction == 'key_2') and (pred_prob > 0.85) and KEY_1:
                t2 = datetime.now()
                if t1 + timedelta(seconds=2) > t2:
                    KEY_2 = True
                else:
                    KEY_1 = False
                    KEY_2 = False

            elif (prediction == 'key_3') and (pred_prob > 0.85) and KEY_1 and KEY_2:
                t3 =datetime.now()
                if t2 + timedelta(seconds=2) > t3:
                    KEY_3 = True
                    SHIELDS = True
                else:
                    KEY_1 = False
                    KEY_2 = False


        cv2.imshow('Dr. Strange shields', frame)
        # cam.send(frame)
        # cam.sleep_until_next_frame()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
