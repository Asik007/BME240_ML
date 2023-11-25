import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
def Augment(in_img: Image, num_img: int, label: str):
    # Convert the image to a tf array
    image_array = tf.keras.preprocessing.image.img_to_array(in_img)
    image_array = tf.expand_dims(image_array, 0)  # Add batch dimension

    # Define an augmentation pipeline
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Generate augmented images
    augmented_images = []
    num_augmented_images = 3  # Number of augmented images to generate
    for i, batch in enumerate(datagen.flow(image_array, batch_size=1)):
        augmented_images.append(tf.squeeze(batch, axis=0))
        if i >= num_augmented_images - 1:
            break

    return augmented_images