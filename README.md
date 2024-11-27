# White Blood Cell Classification Using CNNs and Super Resolution (Computer Vision)
This repository contains a Python implementation for classifying white blood cells (WBCs) using Convolutional Neural Networks (CNNs). The goal of this project is to build an accurate model to identify and classify white blood cells in medical images, helping with the diagnosis and monitoring of blood-related diseases.

The project also includes a part where images are compared with and without Super Resolution (SR) to observe the impact of enhancing image quality for classification.

## Requirements
- Python 3.x
- TensorFlow/Keras
- NumPy
- Matplotlib
- OpenCV
- PIL (Python Imaging Library)

## You can install the necessary packages using pip:

`pip install tensorflow numpy matplotlib opencv-python pillow`

## Dataset
The dataset used for training and testing the model consists of images of white blood cells, each labeled with the corresponding class: Neutrophils, Lymphocytes, Monocytes and Eosinophils/Basophils. The dataset was created in collaboration with a company and cannot be shared publicly. However, other blood datasets can be used as alternatives for similar analyses.

## Features:

- Data Preprocessing: The images are resized, normalized, and augmented to improve model generalization.
- CNN Model: The model is built using a Convolutional Neural Network (CNN), which includes layers like convolutional, pooling, and dense layers to classify the images.
- Model Training: The model is trained on the labeled dataset using techniques like dropout to prevent overfitting.
- Evaluation: The model's performance is evaluated using metrics such as accuracy, precision, recall, specificity, and F1-score, along with a confusion matrix.
- Image Prediction: After training, the model is used to predict the class of new white blood cell images.
- Visualizations: Visualizations include confusion matrices and metric plots to assess model performance.
- Super Resolution Comparison: The project also includes a section where images are compared with and without Super Resolution (SR), helping to assess how improving image quality can impact the classification process.

## Usage

1. Open the notebook on Google Colab.
2. Upload your dataset.
3. Install the required libraries:
4. Train the model by running the corresponding cell in the notebook.
5. To make predictions on new images, upload the image and run the prediction cell.
6. After training, you can visualize performance graphs (like the confusion matrix and metrics) directly in the notebook.
7. A section of the notebook also compares the results of images processed with Super Resolution (SR) versus without SR, to assess the impact of SR on classification performance.

## Model Architecture
The CNN architecture used in this project consists of several layers:

- Convolutional Layers: Extract features from the input images.
- Pooling Layers: Reduce the spatial dimensions and help in feature selection.
- Fully Connected Layers: Classify the image based on the extracted features.
- Dropout: Improve model generalization and prevent overfitting.

## Results
The model provides a classification of white blood cells based on the following categories:

- Neutrophils
- Lymphocytes
- Monocytes
- Eosinophils/Basophils
The performance metrics for the model can be viewed in the generated plots, such as confusion matrices, accuracy, precision, recall, specificity, and F1-score.  Additionally, the comparison of images with and without Super Resolution (SR) shows how the enhancement of image quality can affect classification results.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
