
The pipeline I for PPE Detection can be split up into 3 main steps:
1. Data collection/annotation
2. Model Finetuning
3. Deployment

## Data Collection/Annotation

My final dataset was about 1100 images, with ~900 being from the Houston lab security cameras, and the other ~200 being random stock images I found online. All of the security camera footage was downloaded using an app called the NVMS v3 client, which can connect to the security camera footage and allow you to download videos from within the last 24 hours. I extracted frames (every 15-30 seconds) from these videos for my dataset. Unfortunately the Eurofins firewall doesn't allow you to use this software, so if you want to use this to download images it'll probably have to be on your personal laptop. Stock images were downloaded to give more clear examples of hard to see objects like eyewear for the model.

Annotation was done mostly using CVAT, a free image annotation software. CVAT allows you to locally host its app in a docker container, which is important because it keeps all Eurofins images on your local PC. However, docker desktop requires an enterprise license, so to my understanding you can't use it on windows, I personally used it on WSL. 

I used the YOLO format for the dataset, which looks like this:
```
dataset_yolo/
├── data.yaml
├── train.txt
├── images/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── street_12.png
│   │   └── ...
│   └── val/
│       ├── 0101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── 0001.txt
    │   ├── street_12.txt
    │   └── ...
    └── val/
        ├── 0101.txt
        └── ...
```
Each images has a corresponding .txt file which contain the labels. Each label corresponds to a line with 5 numbers: class, center x, center y, width, height. All coordinates are normalized between 0 and 1.

To summarize, my data collection pipeline was NVMS v3 for image collection, upload to CVAT for annotation, then export as YOLO format for use. 

## Model Finetuning

### YOLOX

A lot of constraints on this project are what Eurofins allows us to use. I wanted to use a YOLO model for object detection, but most of the newer YOLO models are owned by [Ultralytics](https://www.ultralytics.com/), which means we'd need an enterprise license to use them. I needed to either find a model with an open source license or implement one from scratch. I used [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) because it was open source and had good performance, so I used their architecture and loss function and implemented the rest myself. It's not necessary to understand the model's architecture, the most important thing to understand is the model's inputs and outputs. The model only accepts images with shape (batch_size, 3, 640, 640), and outputs (batch_size, 8400, 5 + num_classes).  The 8400 is the number of boxes predicted, and each predicted box is shape (obj, cx, cy, w, h, class1,...), where cx, cy, w, h are all in pixel coordinates. 

As for the loss function, YOLOX trains under a combination of three different losses, an objectness loss, a class loss, and a localization loss. The paper will explain it much better than I could, and either way I just took their code (yolox/loss.py)
### Preprocessing

All images have to be downsized and padded to fit 640 x 640 for the model inputs. The only tricky part about this is shifting the labels correctly to fit the new padded/reshaped image, but my dataset code handles this (data_utils/ppe_dataset.py). Another part is data augmentations, torchvision handles moving the bounding boxes through their augmentations, but they didn't have mosaic, an augmentation that YOLOX uses, so I implemented it myself (data_utils/mosaic.py). 

### Post Processing

The model predicts 8400 total boxes, so these have to be heavily filtered to get the final predictions. For every box, the objectness score is multiplied by each class probability to get the "confidence" scores. Then you take whatever class the box is most confident in and label that as the boxes' class (just a max over these obj x class score). Then, you filter by a confidence threshold and Non Maximum Suppression to get the final set of predictions, which is a tensor of shape (num_predictions, 6), where the 6 is (class, x1, y1, x2, y2, confidence). These are what is drawn.  

### Validation

All validation is done using mean average precision (mAP)(data_utils/metrics.py). Average Precision is just the area under the precision recall curve (plotted across different confidence levels) for each class, and then averaged across classes to get a single number. I also implemented this myself, mostly to understand it better, sci-kit learn also has this function. 

### Training Loop

Combining the pieces above, the basic training pipeline is just:
- Create PyTorch dataset given labeled images
- Preprocess and add data augmentations
- Create and reparameterize model to desired number of classes
- Run batched images through model
- Post process outputs, calculate loss
- Calculate mAP on a held out validation set 

The training loop is a pretty standard PyTorch loop. I used torch's automatic mixed precision (AMP) to run training faster, I kept the backbone parameters fixed for the first 10 epochs, and I stopped doing dataset augmentations for the last 10 epochs. 

## Deployment

Deployment was done using TensorRT, an Nvidia library useful for optimizing models on whatever hardware they're on. My code works for TensorRT 10.3, I haven't tested for other versions, but it's just a single script. FP32 and FP16 engine building work easily and provide speedups, but for the largest possible speedup int8 quantization can be done. The TensorRT engine builder expects the linear quantization and dequantization nodes necessary to be put in beforehand, and this can be done via the [TensorRT Model Optimizer](https://github.com/NVIDIA/TensorRT-Model-Optimizer), which has some example scripts you can use for quantization. This may be unnecessary though, FP16 works fine. All engine plans are on the nano already under engines/. 

For the Nano, the user is eurofins, and the password is eurofins. The directory is ~/projects/PPE_Detection, and in this directory you can run `python trtinference.py` to start the model. No virtual environment is needed. Currently the script should start on boot.  
