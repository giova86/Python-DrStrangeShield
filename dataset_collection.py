# LIS create dataset

# libraries and packages
import cv2
import time
import mediapipe as mp
import numpy as np
import os
from utils import mediapipe_detection, draw_landmarks, draw_landmarks_custom, points_detection, points_detection_hands
import pandas as pd
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("-s", "--sample", dest="sample", default=100,
                    help="number of samples for each letter. default value is 100")
parser.add_argument("-dc", "--det_conf", dest="min_detection_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-tc", "--trk_conf", dest="min_tracking_confidence", default=0.5, type=float,
                    help="Threshold for prediction. A number between 0 and 1. default is 0.5")
parser.add_argument("-o", "--output", dest="output_file", default='hand_position',
                    help="Name of the saved model. default is 'data_rh'")
args = parser.parse_args()

# -- INPUT ----------------------------------------------
labels = np.array(['key_1', 'key_2', 'key_3', 'key_4'])
no_sequences = args.sample + 1
# -------------------------------------------------------


# data collection
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

data = []
cap = cv2.VideoCapture(2)
with mp_holistic.Holistic(min_detection_confidence=args.min_detection_confidence, min_tracking_confidence=args.min_tracking_confidence) as holistic:

    for label in labels:
        id=0
        while id < no_sequences:
#        for id in range(no_sequences):
            ret, frame = cap.read()

            # make detection
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks_custom(frame, results)

            if id == 0:
                cv2.imshow('LIS', frame)
                cv2.waitKey(2500)
                cv2.putText(frame, f'START COLLECTING OF LETTER {label.capitalize()} in:', (15,32), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 4, cv2.LINE_AA)
                cv2.imshow('LIS', frame)
                cv2.waitKey(2000)
                cv2.putText(frame, f'START COLLECTING OF LETTER {label.capitalize()} in: 3', (15,32), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 4, cv2.LINE_AA)
                cv2.imshow('LIS', frame)
                cv2.waitKey(1200)
                cv2.putText(frame, f'START COLLECTING OF LETTER {label.capitalize()} in: 3 2', (15,32), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 4, cv2.LINE_AA)
                cv2.imshow('LIS', frame)
                cv2.waitKey(1200)
                cv2.putText(frame, f'START COLLECTING OF LETTER {label.capitalize()} in: 3 2 1', (15,32), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 4, cv2.LINE_AA)
                cv2.imshow('LIS', frame)
                cv2.waitKey(1000)
                id = id + 1

            else:
                if results.right_hand_landmarks and results.left_hand_landmarks:
                    data.append(points_detection_hands(results))
                    cv2.putText(frame, 'Hand detected', (15,72), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 4, cv2.LINE_AA)
                    cv2.putText(frame, f'Collecting frame {id} for {label.capitalize()}', (15,32), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,255,0), 4, cv2.LINE_AA)
                    id = id + 1
                else:
                    cv2.putText(frame, f'Collecting frame {id} for {label.capitalize()}', (15,32), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,0,255), 4, cv2.LINE_AA)
                    cv2.putText(frame, 'Hand not detected', (15,72), cv2.FONT_HERSHEY_SIMPLEX, 1 ,(0,0,255), 4, cv2.LINE_AA)

            cv2.waitKey(100)
            cv2.imshow('LIS', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

df = pd.DataFrame(np.array(data))
y=[]
for i in labels:
    y = np.concatenate([y, [i] * (no_sequences-1)])
df['y'] = y
pd.DataFrame(df).to_csv(f'data/{args.output_file}.csv')
print(df['y'].value_counts())
