import nibabel as nib
import numpy as np
import tensorflow as tf
from PIL import Image

def preprocess_slice(slice_2d, target_size=(128, 128)):
    # Normalleştir
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
    slice_resized = tf.image.resize(slice_2d[..., np.newaxis], target_size)
    return tf.expand_dims(slice_resized, axis=0)  # (1, H, W, 1)

def calculate_tumor_ratio(nifti_path, model):
    img = nib.load(nifti_path)
    volume = img.get_fdata()

    total_liver_pixels = 0
    total_tumor_pixels = 0

    for i in range(volume.shape[2]):
        slice_2d = volume[:, :, i]
        input_tensor = preprocess_slice(slice_2d)

        prediction = model.predict(input_tensor, verbose=0)
        predicted_mask = tf.argmax(prediction[0], axis=-1).numpy()  # (128, 128)

        tumor_pixels = np.sum(predicted_mask == 2)
        liver_pixels = np.sum((predicted_mask == 1) | (predicted_mask == 2))

        total_tumor_pixels += tumor_pixels
        total_liver_pixels += liver_pixels

    if total_liver_pixels > 0:
        tumor_ratio = (total_tumor_pixels / total_liver_pixels) * 100  # Yüzde
    else:
        tumor_ratio = 0.0

    return tumor_ratio
