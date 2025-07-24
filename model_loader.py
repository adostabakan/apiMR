import tensorflow as tf
import os

def load_model(path="MR_model.h5"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model dosyası bulunamadı: {path}")

    model = tf.keras.models.load_model(path, compile=False)
    return model
