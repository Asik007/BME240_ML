#!/usr/bin/env python
# coding: utf-8

# ## Imports
# All the necessary libraries
# - tensorflow (tf)
#     - a machine learning framework
# - PIL (Python Image Library) 
#     - to look at images easily
# - Numpy (np)
#     - convert images to matrices and other math functions
# - keras
#     - high level API to interface with tf and perform ML tasks
# - Matplotlib
#     - to plot relevant data and show images
# - Technically Jupyter Notebook
#     - enabled live scripting for easy coding and testing
# 
# 
# data:
# https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images
# 

# In[1]:


from PIL import Image
import tensorflow as tf
from keras import Sequential, layers
import keras
import os
import matplotlib.pyplot as plt
import numpy as np


# ## Preproccessing
# 
# we have to convert all images to 224px x 224px. (not here though)
# using keras we split the data into a training dataset and validating dataset
# **this is important as we can not test on data we trained the model with**
# 
# The bottom code is just to show the images while I was writing the code
# 

# In[2]:


from tensorflow.python.data import AUTOTUNE

img_size = (224,224)


train_ds = keras.utils.image_dataset_from_directory(
    directory='Alzheimer_s Dataset/train',
    labels='inferred',
    # label_mode='categorical',
    batch_size=16,
    image_size=img_size)
validation_ds = keras.utils.image_dataset_from_directory(
    directory='Alzheimer_s Dataset/test',
    labels='inferred',
    # label_mode='categorical',
    batch_size=16,
    image_size=img_size)

# batch_1=train_ds.take(1)


def one_hot_label(image, label):
    label = tf.one_hot(label, 4)
    return image, label

train_ds = train_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)
validation_ds = validation_ds.map(one_hot_label, num_parallel_calls=AUTOTUNE)



class_names = ['MildDementia', 'ModerateDementia', 'NonDementia', 'VeryMildDementia']
train_ds.class_names = class_names
validation_ds.class_names = class_names

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(train_ds.class_names[list(labels[i].numpy()).index(1)])
    plt.axis("off")


# We normalize the data to 0-1 from 0-255

# In[3]:


class_names = train_ds.class_names
print(class_names)

normalization_layer = layers.Rescaling(1./255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


# ## Data augmentation
# 
# This is to expand or dataset and add variance, so the dataset is not over-fitted to our certain set of images. We wrote a function to do this in a separate file for readability. The function takes in a directory with images and outputs the original image and 3 augmented images. These augmented images have been rotated, flipped horizontally and vertically, shift in both the X and Y direction, and a shift transformation.

# In[6]:


from Augment import load_and_augment
from PIL import Image
import os
import matplotlib.pyplot as plt

starting_dir = "Alzheimer_s Dataset/train"
image_labels = ["Original", "Augment 1", "Augment 2", "Augment 3"]
labels = os.listdir(starting_dir)
for label in labels:
    curr_dir = os.path.join(starting_dir, label)
    # print(curr_dir)
    augmented_imgs = load_and_augment(curr_dir, .01)
    plt.figure(figsize=(12, 12))
    plt.subplot(2,2,4)
    print("iter")
    i = 0
    for i, img in enumerate(augmented_imgs):
        # plt.figure(figsize=(12, 12))
        # i += 1
        plt.subplot(2,2,i+1)
        plt.title(image_labels[i])
        plt.imshow(img)
        plt.axis('off')


    plt.tight_layout()
    plt.show()
        
    


# ## Defining the Model
# 
# here we utilize InceptionV3 as a base architecture  as a good starting point.
# 
# then we define a keras model which takes the 1000 classes from mobilenet and cuts it down to 4 which represent our 4 classes
# 
# input: resized images --- MobileNet w imagenet weights --> 1000 classes --- Dense layer --> our 4 classes

# In[4]:


from keras.src.applications import InceptionV3

num_classes = len(class_names)

model_MN = InceptionV3(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

model_MN.trainable = False

model = keras.Sequential([
        model_MN,
       layers. Dropout(0.5),
       layers. GlobalAveragePooling2D(),
       layers. Flatten(),
      layers.  BatchNormalization(),
      layers.  Dense(512, activation='relu'),
      layers.  BatchNormalization(),
      layers.  Dropout(0.5),
      layers.  Dense(256, activation='relu'),
      layers.  BatchNormalization(),
      layers.  Dropout(0.5),
      layers.  Dense(128, activation='relu'),
      layers.  BatchNormalization(),
      layers.  Dropout(0.5),
      layers.  Dense(64, activation='relu'),
      layers.  Dropout(0.5),
       layers. BatchNormalization(),
      layers.  Dense(4, activation='softmax')   
])

model.build(img_size)


# here im just compiling the model
# 
# AUC stands for Area-Under-the-Curve.

# In[5]:


METRICS = [tf.keras.metrics.AUC(name='auc'), "acc"]

model.compile(optimizer='adam',loss=tf.losses.CategoricalCrossentropy(),
              # loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=METRICS)


# In[51]:


model.summary()


# epochs is how many times the model trains on every image in the training set
# 
# and then I fit the network to the training set

# In[6]:


epochs=100
fit_md = model.fit(
  train_ds,
  validation_data=validation_ds,
  epochs=epochs                                                                                             
)


# In[7]:


model.save('final_model.keras')
print("Evaluate on test data")
results = model.evaluate(validation_ds, batch_size=1)
print("test loss, test acc:", results)

#Plotting the trend of the metrics during training

fig, ax = plt.subplots(1, 2, figsize = (30, 5))
ax = ax.ravel()

for i, metric in enumerate(["auc", "loss"]):
    ax[i].plot(fit_md.history[metric])
    ax[i].plot(fit_md.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("Epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])


# In[ ]:




