# methods
import cv2
import time
import mediapipe as mp
import numpy as np
import os
import numpy as np
import pandas as pd

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_landmarks_custom(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(255,255,255),thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(255,255,255),thickness=1, circle_radius=1),
                             )
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,110,10),thickness=2, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121),thickness=2, circle_radius=1),
                             )
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(0,0,255),thickness=3, circle_radius=5),
                             mp_drawing.DrawingSpec(color=(0,0,255),thickness=3, circle_radius=5),
                             )
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(255,0,0),thickness=3, circle_radius=5),
                             mp_drawing.DrawingSpec(color=(255,0,0),thickness=3, circle_radius=5),
                             )
    #cv2.rectangle(image, start_point, end_point, color, thickness)

def draw_limit_rh(image, results):
    if results.right_hand_landmarks:
        xMax = max([i.x for i in results.right_hand_landmarks.landmark])
        xMin = min([i.x for i in results.right_hand_landmarks.landmark])
        yMax = max([i.y for i in results.right_hand_landmarks.landmark])
        yMin = min([i.y for i in results.right_hand_landmarks.landmark])

        xMax=xMax+0.1*(xMax-xMin)
        yMax=yMax+0.1*(yMax-yMin)
        xMin=xMin-0.1*(xMax-xMin)
        yMin=yMin-0.1*(yMax-yMin)

        h,w,_ = image.shape
        cv2.rectangle(image, (int(xMin*w), int(yMin*h)),  (int(xMax*w), int(yMax*h)), (255,0,0), 1)
        cv2.line(image, (int(xMin*w), int(yMin*h)), (int(xMin*w), int(yMin*h)+int((yMax*h-yMin*h)/5)), (255,0,0),8)
        cv2.line(image, (int(xMin*w), int(yMin*h)), (int(xMin*w)+int((xMax*w-xMin*w)/5), int(yMin*h)), (255,0,0),8)
        cv2.line(image, (int(xMax*w), int(yMax*h)), (int(xMax*w), int(yMax*h)-int((yMax*h-yMin*h)/5)), (255,0,0),8)
        cv2.line(image, (int(xMax*w), int(yMax*h)), (int(xMax*w)-int((xMax*w-xMin*w)/5), int(yMax*h)), (255,0,0),8)
        cv2.line(image, (int(xMin*w), int(yMax*h)), (int(xMin*w), int(yMax*h)-int((yMax*h-yMin*h)/5)), (255,0,0),8)
        cv2.line(image, (int(xMin*w), int(yMax*h)), (int(xMin*w)+int((xMax*w-xMin*w)/5), int(yMax*h)), (255,0,0),8)
        cv2.line(image, (int(xMax*w), int(yMin*h)), (int(xMax*w), int(yMin*h)+int((yMax*h-yMin*h)/5)), (255,0,0),8)
        cv2.line(image, (int(xMax*w), int(yMin*h)), (int(xMax*w)-int((xMax*w-xMin*w)/5), int(yMin*h)), (255,0,0),8)
        cv2.putText(image, 'Right Hand',(int(xMin*w), int(yMin*h-(yMax*h-yMin*h)/20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

def get_center_lh(image, results):
    if results.left_hand_landmarks:
        xMax = max([i.x for i in results.left_hand_landmarks.landmark])
        xMin = min([i.x for i in results.left_hand_landmarks.landmark])
        yMax = max([i.y for i in results.left_hand_landmarks.landmark])
        yMin = min([i.y for i in results.left_hand_landmarks.landmark])

        return xMin, xMax, yMin, yMax
    else:
        return None, None, None, None

def get_center_rh(image, results):
    if results.right_hand_landmarks:
        xMax = max([i.x for i in results.right_hand_landmarks.landmark])
        xMin = min([i.x for i in results.right_hand_landmarks.landmark])
        yMax = max([i.y for i in results.right_hand_landmarks.landmark])
        yMin = min([i.y for i in results.right_hand_landmarks.landmark])

        return xMin, xMax, yMin, yMax
    else:
        return None, None, None, None

def draw_limit_lh(image, results):
    if results.left_hand_landmarks:
        xMax = max([i.x for i in results.left_hand_landmarks.landmark])
        xMin = min([i.x for i in results.left_hand_landmarks.landmark])
        yMax = max([i.y for i in results.left_hand_landmarks.landmark])
        yMin = min([i.y for i in results.left_hand_landmarks.landmark])

        xMax=xMax+0.1*(xMax-xMin)
        yMax=yMax+0.1*(yMax-yMin)
        xMin=xMin-0.1*(xMax-xMin)
        yMin=yMin-0.1*(yMax-yMin)

        h,w,_ = image.shape
        cv2.rectangle(image, (int(xMin*w), int(yMin*h)),  (int(xMax*w), int(yMax*h)), (0,0,255), 1)
        cv2.line(image, (int(xMin*w), int(yMin*h)), (int(xMin*w), int(yMin*h)+int((yMax*h-yMin*h)/5)), (0,0,255),8)
        cv2.line(image, (int(xMin*w), int(yMin*h)), (int(xMin*w)+int((xMax*w-xMin*w)/5), int(yMin*h)), (0,0,255),8)
        cv2.line(image, (int(xMax*w), int(yMax*h)), (int(xMax*w), int(yMax*h)-int((yMax*h-yMin*h)/5)), (0,0,255),8)
        cv2.line(image, (int(xMax*w), int(yMax*h)), (int(xMax*w)-int((xMax*w-xMin*w)/5), int(yMax*h)), (0,0,255),8)
        cv2.line(image, (int(xMin*w), int(yMax*h)), (int(xMin*w), int(yMax*h)-int((yMax*h-yMin*h)/5)), (0,0,255),8)
        cv2.line(image, (int(xMin*w), int(yMax*h)), (int(xMin*w)+int((xMax*w-xMin*w)/5), int(yMax*h)), (0,0,255),8)
        cv2.line(image, (int(xMax*w), int(yMin*h)), (int(xMax*w), int(yMin*h)+int((yMax*h-yMin*h)/5)), (0,0,255),8)
        cv2.line(image, (int(xMax*w), int(yMin*h)), (int(xMax*w)-int((xMax*w-xMin*w)/5), int(yMin*h)), (0,0,255),8)
        cv2.putText(image, 'Left Hand',(int(xMin*w), int(yMin*h-(yMax*h-yMin*h)/20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

def check_detection(image, results):
    if results.left_hand_landmarks:
        cv2.putText(image, 'Left Hand: DETECTED',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    else:
        cv2.putText(image, 'Left Hand: NOT DETECTED',(10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    if results.right_hand_landmarks:
        cv2.putText(image, 'Right Hand: DETECTED',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
    else:
        cv2.putText(image, 'Right Hand: NOT DETECTED',(10,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

    if results.face_landmarks:
        cv2.putText(image, 'Face: DETECTED',(10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    else:
        cv2.putText(image, 'Face: NOT DETECTED',(10,110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    if results.face_landmarks:
        cv2.putText(image, 'Pose: DETECTED',(10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,256,121), 2)
    else:
        cv2.putText(image, 'Pose: NOT DETECTED',(10,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (80,256,121), 2)

def points_detection(results):
    xMax = max([i.x for i in results.right_hand_landmarks.landmark])
    xMin = min([i.x for i in results.right_hand_landmarks.landmark])
    yMax = max([i.y for i in results.right_hand_landmarks.landmark])
    yMin = min([i.y for i in results.right_hand_landmarks.landmark])
    rh = np.array([[points.x, points.y, points.z] for points in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    for i in np.arange(0, 63, 3):
        rh[i]=(rh[i]-xMin)/(xMax-xMin)
    for i in np.arange(1, 63, 3):
        rh[i]=(rh[i]-yMin)/(yMax-yMin)

    # lh = np.array([[points.x, points.y, points.z] for points in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # po = np.array([[points.x, points.y, points.z] for points in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    # return np.concatenate([lh, rh, po])

    return rh

def points_detection_hands(results):
    '''
    Questa funzione estrae dai risultati del modello holistic le coordinate di tutti i punti di entrambe le mani.
    Le coordinate (x,y), inzialmente espresse tra 0 e 1 rispetto allo schermo vengono poi riscalate rispetto
    al min/max delle coordinate
    '''
    xMaxR = max([i.x for i in results.right_hand_landmarks.landmark])
    xMinR = min([i.x for i in results.right_hand_landmarks.landmark])
    yMaxR = max([i.y for i in results.right_hand_landmarks.landmark])
    yMinR = min([i.y for i in results.right_hand_landmarks.landmark])

    xMaxL = max([i.x for i in results.left_hand_landmarks.landmark])
    xMinL = min([i.x for i in results.left_hand_landmarks.landmark])
    yMaxL = max([i.y for i in results.left_hand_landmarks.landmark])
    yMinL = min([i.y for i in results.left_hand_landmarks.landmark])

    xMin = min([xMinR, xMinL])
    xMax = max([xMaxR, xMaxL])
    yMin = min([yMinR, yMinL])
    yMax = max([yMaxR, yMaxL])

    rh = np.array([[points.x, points.y, points.z] for points in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[points.x, points.y, points.z] for points in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    for i in np.arange(0, 63, 3):
        rh[i]=(rh[i]-xMin)/(xMax-xMin)
    for i in np.arange(1, 63, 3):
        rh[i]=(rh[i]-yMin)/(yMax-yMin)

    con = np.concatenate((rh, lh))

    # xMax = max([i.x for i in results.left_hand_landmarks.landmark])
    # xMin = min([i.x for i in results.left_hand_landmarks.landmark])
    # yMax = max([i.y for i in results.left_hand_landmarks.landmark])
    # yMin = min([i.y for i in results.left_hand_landmarks.landmark])
    # rh = np.array([[points.x, points.y, points.z] for points in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    # for i in np.arange(0, 63, 3):
    #     rh[i]=(rh[i]-xMin)/(xMax-xMin)
    # for i in np.arange(1, 63, 3):
    #     rh[i]=(rh[i]-yMin)/(yMax-yMin)

    # lh = np.array([[points.x, points.y, points.z] for points in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # po = np.array([[points.x, points.y, points.z] for points in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(99)
    # return np.concatenate([lh, rh, po])

    return con
