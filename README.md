# Medical Image Classification Using Transfer Learning and Grad-CAM

## Overview

This project uses Deep Learning and Transfer Learning techniques to classify chest X-ray images as Pneumonia or Normal. The model is built using ResNet50 and TensorFlow.

Grad-CAM visualization is used to highlight the important regions in the X-ray image that influence the model’s prediction.

## Technologies Used

* Python
* TensorFlow
* ResNet50
* OpenCV
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

## Features

* Pneumonia detection from chest X-ray images
* Transfer Learning using ResNet50
* Grad-CAM heatmap visualization
* Confusion Matrix and Classification Report
* Real-time image prediction support
```md
## Dataset

Chest X-ray dataset used from Kaggle.
```


## Project Structure

```bash
Medical-Image-Classification-Using-Transfer-Learning-and-GradCAM/
│
├── dataset/
├── main.py
├── my_image.jpg
├── requirements.txt
└── README.md
```
````md
## How to Run

```bash
pip install -r requirements.txt
python main.py
````

## Model Performance

* Training Accuracy: 95%
* Validation Accuracy: 92%

```
```



## Output Screenshots

### Confusion Matrix



![Confusion Matrix](./Screenshots/confusion_matrix.png)


### Grad-CAM Visualization

![Grad-CAM Output](./Screenshots/gradcam_output.png)



