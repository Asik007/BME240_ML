import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import os
import random

from PIL import Image
import tensorflow as tf

def Augment(in_img: Image, num_img: int):
    # Convert the image to a tf array
    image_array = tf.keras.preprocessing.image.img_to_array(in_img)
    image_array = tf.expand_dims(image_array, 0)  # Add batch dimension

    # Define an augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        # zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )

    # Generate augmented images
    augmented_images = []
    num_augmented_images = num_img  # Number of augmented images to generate
    for i, batch in enumerate(datagen.flow(image_array, batch_size=1)):
        augmented_images.append(tf.squeeze(batch, axis=0))
        if i >= num_augmented_images - 1:
            break

    # augmented_images = [i.numpy() for i in augmented_images]

    return augmented_images


def load_and_augment(data_dir, percent):
    # labels = os.listdir(data_dir)
    #
    # for label in labels:
    returned_img = []
    # curr_dir = os.path.join(data_dir, label)
    print(data_dir)
    files = os.listdir(data_dir)

    # Calculate 40% of the total number of files
    twenty_percent = int(percent * len(files))

    # Choose a random 20% subset of files
    random_files = random.sample(files, twenty_percent)

    for file in random_files:
        curr_img_path = os.path.join(data_dir, file)
        curr_img = Image.open(curr_img_path)
        returned_img.append(curr_img)
        # plt.figure(figsize=(12, 12))
        # plt.subplot(2,2,4)
        # plt.title(f'original')
        # plt.imshow(curr_img)
        # plt.axis('off')

        # Here we make 3 augmented images which inherit the image of the original
        # Assuming Augment function applies some augmentation to the image
        aug_imgs = Augment(curr_img, 3)
        for num, aug_img in enumerate(aug_imgs):
            aug_img_name = f"{os.path.splitext(curr_img_path)[0]}_augment({num}).jpg"
            # np_img = (aug_img[:,:,0]).astype(np.uint8)
            img_8bit = tf.cast(aug_img, tf.uint8)
            returned_img.append(img_8bit)
            # plt.subplot(2, 2, num + 1)
            # plt.title(f'Augmented Image {num + 1}')
            # plt.imshow(img_8bit)
            # plt.axis('off')
            # tf.keras.preprocessing.image.save_img(aug_img_name,img_8bit)
            print(f"Saved augmented image: {aug_img_name}")
    return returned_img