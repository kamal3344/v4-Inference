
## Yolov Object Detection Using Darknet Framework and OpenCV

Object detection is a fundamental task in computer vision that involves identifying and localizing objects within an image or video. Over the years, several object detection algorithms have been developed, each with its own strengths and limitations. One such algorithm that has gained significant popularity is YOLOv4 (You Only Look Once), known for its high accuracy and real-time performance.

In this blog post, we will explore the YOLOv4 algorithm and guide you through its implementation using OpenCV. We will cover the architecture, explain the code, and demonstrate how to perform object detection on both images and videos.

## Introduction to YOLOv4

YOLOv4 is the fourth iteration of the YOLO algorithm, which revolutionized object detection by introducing a single-stage, end-to-end approach. Unlike traditional two-stage detectors, YOLOv4 processes the entire image in a single pass, making it highly efficient. It achieves state-of-the-art accuracy by leveraging a combination of advanced techniques, including a powerful backbone network, feature pyramid network, and multiple detection heads.


![App Screenshot](https://miro.medium.com/v2/resize:fit:640/format:webp/1*EM2yNCsxM_F1XrhvZZxdfQ.png)

## understanding the Code

The code provided implements YOLOv4 using opencv. Let’s break it down step by step:

Importing the necessary packages: We start by importing the required packages, including OpenCV, NumPy, time, and argparse. These packages provide the necessary tools for image processing, numerical operations, and command-line argument parsing.

## YOLOv4 Class:

The Yolov4 class encapsulates the functionality of YOLOv4. It initializes the weights and configuration file paths, defines the list of classes, and loads the pre-trained model using cv2.dnn.readNetFromDarknet. It also sets up the necessary parameters for inference.

## Bounding Box Function:

The bounding_box method takes the output of the YOLOv4 model and extracts the bounding box coordinates, confidence scores, and class labels. It applies a confidence threshold and performs non-maximum suppression to filter out weak detection's and overlapping boxes.

## Prediction Function:

The Predictions method takes the filtered bounding box information and overlays the boxes, class labels, and confidence scores on the original image. It also calculates the inference time and displays it on the image.

## Inference Function:

The Inference method performs the actual inference on the input image. It pre-processes the image, sets it as the input to the YOLOv4 model, and retrieves the output predictions. It then calls the bounding_box and predictions functions to process and visualize the results.

## Main Execution:

In the main section of the code, we parse the command-line arguments using argparse. If an image path or video path is provided, the corresponding inference is performed using the Yolov4 class. The results are displayed and optionally saved to a video file.

## Running the Code

To try out the YOLOv4 implementation, follow these steps:

    1. Make sure you have the required dependencies installed, including OpenCV.
    2. Download the YOLOv4 weights here and configuration file here and place them in the same directory as the code.
    3. Open a terminal or command prompt and navigate to the directory containing the code.
    4. To perform object detection on an image, run the command python yolov4.py --image path/to/image.jpg. Replace path/to/image.jpg with the actual path to your image file.
    5. To perform object detection on a video, run the command python yolov4.py --video path/to/video.mp4. Replace path/to/video.mp4 with the actual path to your video file.
    6. for images = python Inference_args.py — weights yolov4.weights — cfg=yolov4.cfg — image=bus.jpg — img_size=320
    7. for videos = python Inference_args.py — weights yolov4.weights — cfg=yolov4.cfg — video=traffic_signs.mp4 — img_size=320

## Resource Utilization

If you are currently running the YOLOv4 inference using OpenCV on a CPU, you may experience high CPU usage, with the CPU utilization reaching above 90%. To improve the performance and achieve better frame rates per second (FPS), it is recommended to utilize GPU acceleration.

### Result:

![App Screenshot](https://miro.medium.com/v2/resize:fit:640/format:webp/1*JhZCXqUmrZlNitoK3__VdQ.png)


## 🛠 Skills

        1.Python 
        2.Machine learning 
        3.Statistics
        4.Mathematics
        5.Numpy 
        6.Neural Networks
        7.Deep Learning Multilayer Perceptron concepts 
        8.Computer Vision
    


## Support

For support, email saikamal9797@gmail.com .


## Acknowledgements

 - [Documentation](https://pjreddie.com/darknet/yolo/)


## 🔗 Links

For my work you can follow:


[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/sai-kamal-korlakunta-a81326163/)

[Medium](https://medium.com/@korlakuntasaikamal10)

