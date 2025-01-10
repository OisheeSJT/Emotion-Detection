# Emotion Detection

This project implements a real-time emotion detection system using a convolutional neural network (CNN) trained to recognize seven emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The system utilizes OpenCV for video capture and face detection.
The data set used here is [FER2013](https://www.kaggle.com/datasets/deadskull7/fer2013). 

## Features
- Real-time emotion detection from webcam feed.
- Supports detection of seven primary emotions.

## Requirements
Install the required libraries 
> pip install -r requirements.txt

## Important 
make sure all the required libraries are installed properly and you have downloaded the [Fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013) dataset. It is highly recommended to use a virtual Environment 
Download all the code (you can ignore Emotion_Detection.h5,  y_test.npy, y_train.npy, x_test.npy, x_train.npy because these files will be created once you run train_model.py and train the model in your local machine )
First Run the **data_processing.py** to prepare and process the dataset then run **train_model.py**. After the completion of the training now you are ready to run **emotion_detection.py** and see the result.
# Emotion-Detection
