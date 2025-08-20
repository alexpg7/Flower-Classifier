# 💮Flower-Classifier ![Static Badge](https://img.shields.io/badge/Udacity-%232B00FF?style=for-the-badge&logo=Udacity&logoColor=white)
![Static Badge](https://img.shields.io/badge/Python-blue?logo=Python&logoColor=white)
![Static Badge](https://img.shields.io/badge/Jupyter_Notebook-orange?logo=Jupyter&logoColor=white)
![Static Badge](https://img.shields.io/badge/TensorFlow-orange?logo=TensorFlow&logoColor=white)
![Static Badge](https://img.shields.io/badge/status-completed-green)

A convolutional neural network to classify 5 different types of flowers. Avoidance of overfitting through data augmentation and dropout.

## 🏁The goal

Classifying images can be a tough task, even more if there are lots of images to classify. It is obvious that at the beggining we will have to do it manually. Eventhough, since it is a repetitive task, we could train a neural network to do the job for us, using all the data that we classified manually.

In these types of tasks, the usual thing is to analyze all the patterns and design an algorithm. To analyze images, figuring out all the hidden patterns would be such a painfull work; instead of that, a neural network can handle the job. The type of neural network that we will use will be a [convolutional](https://en.wikipedia.org/wiki/Convolutional_neural_network) one. The reason why is that it is equipped with tools that make the analysis of spacial patterns in images, such as contrast lines, hue gradients and so on.

The goal of the model will be to take an image of a flower and output a probability distribution corresponding to each type of flower it "thinks" it is.

**EXAMPLE HERE**

## 🔰Setting up the modules

The modules that we will use are the usuals:

```Python
import os # system management
import numpy as np # arrays
import glob # filenames
import shutil # moving files
import matplotlib.pyplot as plt # plotting
import tensorflow as tf # neural networks
from tensorflow.keras.models import Sequential # sequential arquitecture
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D # different layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator # data generator from dataset
```
