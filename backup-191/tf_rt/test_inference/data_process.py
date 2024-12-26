import numpy as np

def data_preprocess(img_path, img_shape, batch_size):
    from skimage import io
    from skimage.transform import resize
    url = img_path
    img = resize(io.imread(url), img_shape)
    input_batch = 255 * np.array(
        np.repeat(np.expand_dims(np.array(img), axis=0), batch_size, axis=0),
        dtype=np.float32,
    )
    input_batch = np.expand_dims(input_batch, axis=-1)

    return input_batch