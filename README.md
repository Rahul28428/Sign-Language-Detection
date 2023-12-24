# Sign Language Detection using SVM / Random Forest

## Overview
This project enables real-time sign language detection through computer vision and machine learning. It captures hand gestures, processes them with MediaPipe, and employs SVM/Random Forest for classification.

## Project Structure
- `collect_imgs.py`: Collects hand gesture images for dataset creation.
- `create_dataset.py`: Converts image data into hand landmarks using MediaPipe and saves as a pickle file.
- `train_classifier.py`: Trains an SVM classifier using the dataset and performs hyperparameter tuning with GridSearchCV.
- `inference_classifier.py`: Uses the trained model for real-time sign language interpretation through webcam input.

## Requirements
- Python 3.x
- OpenCV
- MediaPipe
- Scikit-learn
- Numpy

## Usage
1. Run `collect_imgs.py` to capture hand gesture images.
2. Execute `create_dataset.py` to convert images into hand landmarks and save the dataset.
3. Train the classifier by running `train_classifier.py`.
4. Use the trained model with real-time webcam input via `inference_classifier.py`.

## Model Selection
Choose between SVM and Random Forest by modifying the classifier in `train_classifier.py`. Update the `labels_dict` in `inference_classifier.py` accordingly.

## Credits
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [Scikit-learn](https://scikit-learn.org/)

Feel free to contribute or report issues!





# Sign Language Detection using LSTM

## Overview

This project focuses on real-time sign language detection using computer vision and machine learning techniques. It captures hand gestures, processes them with MediaPipe, and utilizes a combination of LSTM layers and TensorFlow for accurate action detection from holistic keypoints.

## Key Features

- Holistic keypoints extraction with MediaPipe
- Real-time interpretation of sign language gestures
- LSTM-based action detection for sequence modeling
- Utilizes TensorFlow, MediaPipe, and OpenCV for implementation

## Setup

1. Install the required dependencies:
    ```bash
    pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
    ```

2. Run the main script:
    ```bash
    python sign_language_detection.py
    ```

## Data Collection

The project includes a data collection module for building and training the LSTM model. It captures sequences of hand gestures, extracts keypoints, and saves them for training.

## Training the Model

The LSTM model is trained on the collected data using TensorFlow. The training process includes multiple epochs and utilizes the categorical crossentropy loss function.

```bash
python train_model.py
```

## Evaluation and Testing
The trained model is evaluated using test data, and its performance is assessed using metrics such as multilabel confusion matrix and accuracy score.

```bash
python evaluate_model.py
```

markdown
Copy code
# Sign Language Detection using LSTM

## Overview

This project focuses on real-time sign language detection using computer vision and machine learning techniques. It captures hand gestures, processes them with MediaPipe, and utilizes a combination of LSTM layers and TensorFlow for accurate action detection from holistic keypoints.

## Key Features

- Holistic keypoints extraction with MediaPipe
- Real-time interpretation of sign language gestures
- LSTM-based action detection for sequence modeling
- Utilizes TensorFlow, MediaPipe, and OpenCV for implementation

## Setup

1. Install the required dependencies:
    ```bash
    pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib
    ```

2. Run the main script:
    ```bash
    python sign_language_detection.py
    ```

## Data Collection

The project includes a data collection module for building and training the LSTM model. It captures sequences of hand gestures, extracts keypoints, and saves them for training.

## Training the Model

The LSTM model is trained on the collected data using TensorFlow. The training process includes multiple epochs and utilizes the categorical crossentropy loss function.

```bash
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_3 (LSTM)               (None, 30, 64)            442112    
                                                                 
 lstm_4 (LSTM)               (None, 30, 128)           98816     
                                                                 
 lstm_5 (LSTM)               (None, 64)                49408     
                                                                 
 dense_3 (Dense)             (None, 64)                4160      
                                                                 
 dense_4 (Dense)             (None, 32)                2080      
                                                                 
 dense_5 (Dense)             (None, 3)                 99        
                                                                 
=================================================================
Total params: 596675 (2.28 MB)
Trainable params: 596675 (2.28 MB)
Non-trainable params: 0 (0.00 Byte)
```


## Evaluation and Testing
The trained model is evaluated using test data, and its performance is assessed using metrics such as multilabel confusion matrix and accuracy score.


Real-time Prediction
The real-time sign language detection module captures video frames, extracts holistic keypoints, and predicts gestures using the trained LSTM model. The predictions are displayed on the screen, providing a user-friendly interface for sign language interpretation.






