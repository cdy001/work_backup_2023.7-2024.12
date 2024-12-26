import numpy as np
from PIL import Image
from skimage import img_as_float

def image_batch(img_paths, batch_size):
    batch_paths = []
    for i in range(0, len(img_paths), batch_size):
        if i + batch_size < len(img_paths):
            batch = img_paths[i:i+batch_size]
        else:
            batch = img_paths[i:]
        batch_paths.append(batch)
    return batch_paths

def read_images(batch_path, img_shape):
    image_batch = []
    for path in batch_path:
        # img = img_as_float(Image.open(path).resize(img_shape[::-1]))
        img = Image.open(path).resize(img_shape[::-1])
        image_batch.append(img)
    
    return np.expand_dims(np.array(image_batch), axis=-1)

if __name__ == "__main__":
    import os
    import glob
    import time
    root_path = "/var/cdy_data/aoyang/data/BGB1Q42E-D"
    img_paths = glob.glob(os.path.join(root_path, "*/*/*.bmp"))
    print(len(img_paths))
    batch_paths = image_batch(img_paths, 24)
    for batch_path in batch_paths:
        time_start = time.time()
        img_batch = read_images(batch_path, img_shape=(682, 682))
        time_end = time.time()
        print(f"{img_batch.shape}, time cost: {time_end - time_start:.4f}")