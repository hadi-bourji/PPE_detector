#!/bin/bash

NANO_USERNAME=eurofins
NANO_IP=10.172.0.50
NANO_DIR=/home/eurofins/ppe_violations/nano_camera
LOCAL_DIR=/c/Users/vs2u/Documents/vscode/PPE_Detection/images

scp -r $NANO_USERNAME@$NANO_IP:$NANO_DIR $LOCAL_DIR