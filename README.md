# Object_Detection

## Overview
This project detects objects using a pre-trained neural network (SSD MobileNet) and provides real-time visual feedback alongside verbal notifications. The object detection is done using OpenCV's deep learning module (`cv2.dnn`), while object names are announced via a text-to-speech engine (`pyttsx3`).

The application captures live video from a camera, detects objects, and audibly announces the name of any detected objects using a non-blocking, multithreaded approach for speech.

## Features
- **Real-Time Object Detection**: Uses an SSD MobileNet pre-trained model to detect objects from the COCO dataset.
- **Text-to-Speech Integration**: The detected objects are announced using a text-to-speech engine (Pyttsx3) without disrupting the video stream.
- **Multithreaded Speech**: The speech functionality runs asynchronously, ensuring continuous video processing while announcing detected objects.

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- Pyttsx3 (`pyttsx3`)
