import numpy as np
from PIL import Image
import cv2
from skimage import img_as_float

def data_preprocess(img_path, img_shape, batch_size):
    from skimage import io
    from skimage.transform import resize
    img = resize(io.imread(img_path), img_shape)
    # img = img_as_float(Image.open(img_path).convert("L").resize((img_shape[::-1])))
    # img = img_as_float(cv2.resize(cv2.imread(img_path, flags=-1), dsize=img_shape[::-1]))
    input_batch = 255 * np.array(
        np.repeat(np.expand_dims(np.array(img), axis=0), batch_size, axis=0),
        dtype=np.float32,
    )
    input_batch = np.expand_dims(input_batch, axis=-1)

    return input_batch