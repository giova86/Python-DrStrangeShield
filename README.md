# Dr. Strange Shields
Doctor Strange Shields using Python , Opencv & Mediapipe

## Python Version
This code is tested to work with Python 3.9

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

## Usage
In order to activate the shields you have to perform a "magical" sequence of hands position.

<br>
<p align="center">
  <img width="460"  src="images/position_1.jpg">
  <figcaption align = "center">Figure 1: First Position].</figcaption>
</p>
<br>

