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

## 📑Dataset

The dataset will be taken from the official [TensorFlow examples](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz). Using ``tf.keras.utils.get_file`` and extracting the ``.tgz`` file, we end up with a folder with all the already classified images.

```Python
_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"

zip_file = tf.keras.utils.get_file(origin=_URL,
                                   fname="flower_photos.tgz",
                                   extract=True)

base_dir = os.path.join(os.path.dirname(zip_file), 'flower_photos_extracted/flower_photos')
```

Using the ``os`` module, we assign to ``base_dir`` the following path: ``/root/.keras/datasets/flower_photos_extracted/flower_photos``. In future versions of the ``tf.keras`` module this could change; so, please, check where is your ``flower_photos`` directory.

```output
/
└── root/
    └── .keras/
        └── datasets/
            └── flower_photos_extracted/
                └── flower_photos/
                    ├── daisy
                    ├── dandelion
                    ├── roses
                    ├── sunflowers
                    └── tulips
```

As it can be seen, we have 5 types of flowers: roses, daisy, dandelion, sunflowers and tulips.

```Python
classes = ['roses', 'daisy', 'dandelion', 'sunflowers', 'tulips']
```

### 📁Getting the data ready

Now, using these ``classes``, we have to create directories to separate the **training** data from the **validation** data. To do so, we use the ``os``, ``blob`` and ``shutil`` modules; they allow us to move, create and select files/directories.

```Python
for cl in classes:
  # search for the class directory
  img_path = os.path.join(base_dir, cl)
  # pick up the list of all .jpg images inside the dir
  images = glob.glob(img_path + '/*.jpg')
  # show how many are there
  print("{}: {} Images".format(cl, len(images)))
  # choose an 80% of the images to train
  num_train = int(round(len(images)*0.8))
  # classify them
  train, val = images[:num_train], images[num_train:]

  for t in train:
    if not os.path.exists(os.path.join(base_dir, 'train', cl)):
      # create "train/class" folder
      os.makedirs(os.path.join(base_dir, 'train', cl))
    # move all the images there
    shutil.move(t, os.path.join(base_dir, 'train', cl))

  # repeat for validation
  for v in val:
    if not os.path.exists(os.path.join(base_dir, 'val', cl)):
      os.makedirs(os.path.join(base_dir, 'val', cl))
    shutil.move(v, os.path.join(base_dir, 'val', cl))

# set up the paths
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
```
```output
roses: 641 Images
daisy: 633 Images
dandelion: 898 Images
sunflowers: 699 Images
tulips: 799 Images
```

Now, our training and validation datasets are contained in the paths pointed by ``train_dir`` and ``val_dir``.

## 1️⃣First model (overfitted)

To see an example of a simple model, we will train a neural network and will observe how it overfits. This is really common when beggining to work with neural networks. It is a phenomonon that is hard to avoid if you do not know where does it come from and how the network is designed internally.

The first thing to do is setting up an ``image data generator``. This is a ``tensorflow.keras`` object that takes a directory (not loaded in the stack), processes images and makes batches out of it. This is a really useful tool to avoid your RAM overloading. Also, it has built-in tools that preprocess the image randomly, so the model can work out with an alternative version of the image to avoid overfitting (we will see this later).

The colors of the image (given by a ``[R,G,B]`` vector) will be normalized to 1 (that means, dividing them by 255). Finally, since simple neural networks can only work with a fixed image resolution, we will have to rescale them every time we try to feed them into the model (150x150 px).

The data generator will be set up to read from a directory (``train_dir``), using ``.flow_from_directory`` we can specify all the configurations (such as batch_size, order, target size, etc). A very important argument of this method is the ``class_mode='sparse'``. This allows the data batch to include the final outputs depending on the directories where these images come from. Somehow, the directories where these images are contained work as a label, and they are set into the ``train_data_gen`` variable.

```Python
batch_size = 100
IMG_SHAPE = 150

image_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen.flow_from_directory(
                                                batch_size=batch_size,
                                                directory=train_dir,
                                                shuffle=True,
                                                target_size=(IMG_SHAPE,IMG_SHAPE),
                                                class_mode='sparse'
                                                )
```
```output
Found 2935 images belonging to 5 classes.
```

Also, we can create an ``image data generator`` for the validation set:

```Python
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=batch_size,
                                                 directory=val_dir,
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='sparse')
```
```output
Found 735 images belonging to 5 classes.
```

