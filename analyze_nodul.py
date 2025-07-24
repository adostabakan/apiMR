import nibabel as nib
import numpy as np
import tensorflow as tf
from scipy.ndimage import label

def preprocess_slice(slice_2d, target_size=(128, 128)):
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
    slice_resized = tf.image.resize(slice_2d[..., np.newaxis], target_size)
    return tf.expand_dims(slice_resized, axis=0)

def filter_predicted_volume(volume, min_voxel=500):
    filtered_volume = volume.copy()
    for i in range(1, volume.shape[0] - 1):
        current = (volume[i] == 2)
        prev = (volume[i - 1] == 2)
        next_ = (volume[i + 1] == 2)
        if not (np.any(prev) or np.any(next_)):
            filtered_volume[i][current] = 0

    tumor_mask = (filtered_volume == 2).astype(np.uint8)
    structure = np.ones((3, 3, 3))
    labeled_array, num_features = label(tumor_mask, structure=structure)

    for region_idx in range(1, num_features + 1):
        region_voxels = np.sum(labeled_array == region_idx)
        if region_voxels < min_voxel:
            filtered_volume[labeled_array == region_idx] = 0

    return filtered_volume

def count_nodules(volume):
    tumor_mask = (volume == 2).astype(np.uint8)
    structure = np.ones((3, 3, 3))
    _, num_features = label(tumor_mask, structure=structure)
    return num_features

def calculate_nodule_count(nifti_path, model):
    nii = nib.load(nifti_path)
    volume = nii.get_fdata()

    mask_volume = []
    for i in range(volume.shape[2]):
        slice_2d = volume[:, :, i]
        input_tensor = preprocess_slice(slice_2d)
        prediction = model.predict(input_tensor, verbose=0)
        prediction_mask = tf.argmax(prediction[0], axis=-1).numpy()
        mask_volume.append(prediction_mask)

    mask_volume = np.array(mask_volume)  # (Z, H, W)
    mask_volume = np.transpose(mask_volume, (1, 2, 0))  # (H, W, Z)

    filtered = filter_predicted_volume(mask_volume, min_voxel=500)
    nodule_count = count_nodules(filtered)
    return nodule_count
