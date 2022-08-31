# Dr. Strange Shields
Doctor Strange Shields using Python , Opencv & Mediapipe

<br>
<p align="center">
  <img width="640"  src="./images/example.png">
</p>  <figcaption style="align: right">YouTube URL: https://www.youtube.com/watch?v=pXJt6sXhm_w.</figcaption>
<br>

## Python Version
This code is tested to work with Python 3.9 on M1 architecture. 

## Environment

```
python3.9 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to
Run
```
python shield.py
```
Optional Arguments
```
usage: shield.py [-h] [-m FILE] [-t THRESHOLD_PREDICTION]
                 [-dc MIN_DETECTION_CONFIDENCE] [-tc MIN_TRACKING_CONFIDENCE]
                 [-c CAMERA] [-s FILE]

optional arguments:
  -h, --help            show this help message and exit
  -m FILE, --model FILE
                        PATH of model FILE.
  -t THRESHOLD_PREDICTION, --threshold THRESHOLD_PREDICTION
                        Threshold for prediction. A number between 0 and 1.
                        default is 0.5
  -dc MIN_DETECTION_CONFIDENCE, --det_conf MIN_DETECTION_CONFIDENCE
                        Threshold for prediction. A number between 0 and 1.
                        default is 0.5
  -tc MIN_TRACKING_CONFIDENCE, --trk_conf MIN_TRACKING_CONFIDENCE
                        Threshold for prediction. A number between 0 and 1.
                        default is 0.5
  -c CAMERA, --camera_id CAMERA
                        ID of the camera. An integer between 0 and N. Default
                        is 1
  -s FILE, --shield FILE
                        PATH of the video FILE.
```

## Usage
- In order to activate the shields you have to perform a "magical" sequence of hands position.

<br>
<p align="center">
  <img width="360"  src="./images/position_1.png">
  <img width="360"  src="./images/position_2.png">
  <img width="360"  src="./images/position_3.png">
  <figcaption align = "center">Figure 1: First Position].</figcaption>
</p>
<br>

- In order to deactivate the shields you have to execute a "magical" hands position.

<br>
<p align="center">
  <img width="360"  src="./images/position_4.png">
  <figcaption align = "center">Figure 1: First Position].</figcaption>
</p>
<br>
