import os
import glob
import traceback
from PIL import Image
from matplotlib import colormaps



if __name__ == "__main__":
    # path = "/data/chendeyang/code/tensorflow_model_d/heatmap/IMAGE1_0174.bmp"
    # try:
    #     # img = tf.io.read_file(path)
    #     # _ = tf.io.decode_bmp(img, channels=0)
    #     with Image.open(path) as img:
    #         img.load()
    # except:
    #     print(traceback.format_exc())
    #     print(f"{path} is damaged")

    # print(list(colormaps))
    # print("jet" in list(colormaps), list(colormaps).index("jet"))

    data_path = "/var/cdy_data/aoyang/data/BGB1Q42E-D_tvt"
    save_model_path = "/data/chendeyang/code/tensorflow_model_d/tmp"

    model_path = os.path.join(save_model_path, "best_model.h5")
    show_img_path = glob.glob(os.path.join(data_path, 'train/0/*bmp'))[0]
    save_path_root = os.path.join(os.path.dirname(data_path), 'heatmap_result')
    class_names = os.listdir(os.path.join(data_path, 'train'))
    print(
        f"model_path: {model_path}, show_img_path: {show_img_path}, save_path_root: {save_path_root}")
    for item in class_names:
        if item in ["0", "1", "41"]:
            continue
        if os.path.isdir(os.path.join(data_path, 'train', item)):
            img_paths = glob.glob(os.path.join(data_path, f"*/{item}/*.bmp"))
            save_path = os.path.join(save_path_root, f"{item}")
            print(f"{item}: {len(img_paths)}, save_path: {save_path}")
