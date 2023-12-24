# Sign-Language-Detection


## Sign Language Detection using SVM / Random Forest

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


