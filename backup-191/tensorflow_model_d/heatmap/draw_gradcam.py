import os
import sys
sys.path.append(os.getcwd())
import glob
import numpy as np

from utils import (
    time_cost,
    remove_invalid_strings_inplace,
    init_gpu_env,
    gen_dataset,
    create_save_path,
    gen_superimposed_img,
)
from cam_model import ModelWithConvOut, ModelWithConvOuts
from get_cam import get_gradcam, get_layercam


def heatmaps_sum_all(conv_maps, show_img_path, img_paths, alpha=0.4, save_path=None):
    conv_map_all = np.array([None])
    for i, conv_map in enumerate(conv_maps):
        conv_map_all = np.maximum(conv_map_all, conv_map) if conv_map_all.any() != None else conv_map
    if conv_map_all.all() == None:
        return
    conv_map_all /= np.max(conv_map_all)  # 归一化
    superimposed_img = gen_superimposed_img(conv_map_all, show_img_path, alpha)
    # Save the superimposed image
    cam_save_path = create_save_path(img_paths, save_path)
    cam_path = os.path.join(cam_save_path, "all_result.bmp")
    superimposed_img.save(cam_path)


def heatmaps_frequency_sum_all(conv_maps, show_img_path, img_paths, alpha=0.4, save_path=None):
    if len(conv_maps) <= 0:
        return
    conv_maps_stack = np.stack(conv_maps, axis=0)
    conv_maps_sum = np.sum(conv_maps_stack, axis=0)  # 全部相加，等价于根据样本频率加权
    conv_maps_sum = np.maximum(conv_maps_sum, 0)  # ReLU激活
    conv_maps_sum /= np.max(conv_maps_sum) # 归一化
    superimposed_img = gen_superimposed_img(conv_maps_sum, show_img_path, alpha)
    # Save the superimposed image
    cam_save_path = create_save_path(img_paths, save_path)
    cam_path = os.path.join(cam_save_path, "all_result_frequency.bmp")
    superimposed_img.save(cam_path)
    
    
@time_cost
def inference(model_path, img_paths, save_path=None, batch_size=1, alpha=0.4, save=True, class_idx=None, method="Grad-CAM"):
    if method not in ["Grad-CAM", "Layer-CAM"]:
        raise ValueError("Invalid method. Must be 'Grad-CAM' or 'Layer-CAM'.")
    if method == "Grad-CAM":  # Grad-CAM
        model = ModelWithConvOut(model_path)  # Grad-CAM model
        get_cam = get_gradcam
    else:  # Layer-CAM
        model = ModelWithConvOuts(model_path)  # Layer-CAM model
        get_cam = get_layercam
    conv_maps_all, class_idxes_all, class_outputs_all = [], [], []
    shape = model.input_spec[0].shape[1:3]  # (h, w)
    dataset = gen_dataset(img_paths, batch_size, shape)
    
    for k, img_arrays in enumerate(dataset):
        
        conv_maps, class_idxes, class_outputs = get_cam(model, img_arrays, class_idx)

        conv_maps_all.extend(conv_maps)
        class_idxes_all.extend(class_idxes)
        class_outputs_all.extend(class_outputs)

        if save:
            cam_save_path = create_save_path(img_paths, save_path)
            for i, conv_map in enumerate(conv_maps):
                if class_idxes[i] == 0:
                    continue
                img_path = img_paths[k*batch_size+i]
                superimposed_img = gen_superimposed_img(conv_map, img_path, alpha)
                # Save the superimposed image
                name = os.path.basename(img_path)
                cam_path = os.path.join(cam_save_path, name)
                superimposed_img.save(cam_path)
                print(f"img_path:{img_path}, class_idx:{class_idxes[i]}, class_output:{class_outputs[i]}")
        
    print(f"conv_maps before filter: {len(conv_maps_all)}")
    conv_maps_all = [conv_map for conv_map, class_idx in zip(conv_maps_all, class_idxes_all) if class_idx != 0]
    print(f"conv_maps after filter: {len(conv_maps_all)}")

    return conv_maps_all, class_idxes_all, class_outputs_all




if __name__ == "__main__":
    method = "Grad-CAM"
    device_id = 2
    alpha = 0.4
    
    model_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/350_epoch.h5"
    
    show_img_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/828/R19C08_22_L1_1720683295.bmp"
    save_path = None

    # 初始化GPU
    init_gpu_env(device_id)

    # 生成热力图并绘制
    img_root_path = "/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/828/Electrode_dirty"
    img_paths = glob.glob(os.path.join(img_root_path, "*.bmp"))
    remove_invalid_strings_inplace(img_paths)
    conv_maps, class_idxs, class_outputs = inference(
        model_path=model_path,
        img_paths=img_paths,
        batch_size=4,
        alpha=alpha,
        save_path=save_path,
        method=method,
        # class_idx=33
    )
    heatmaps_sum_all(
        conv_maps,
        show_img_path=show_img_path,
        img_paths=img_paths,
        alpha=alpha,
        save_path=save_path,
    )

    heatmaps_frequency_sum_all(
        conv_maps,
        show_img_path=show_img_path,
        img_paths=img_paths,
        alpha=alpha,
        save_path=save_path,
    )

    # root_path = '/var/cdy_data/aoyang/data/heatmap/BGB1Q42E/828'
    # files = os.listdir(root_path)
    # for dir_name in files:
    #     img_root_path = os.path.join(root_path, dir_name)
    #     if os.path.isdir(img_root_path):
    #         img_paths = glob.glob(os.path.join(img_root_path, "*.bmp"))
    #         remove_invalid_strings_inplace(img_paths)
    #         conv_maps, class_idxs, class_outputs = inference(
    #             model_path=model_path,
    #             img_paths=img_paths,
    #             batch_size=4,
    #             alpha=alpha,
    #             save_path=save_path,
    #             method=method,
    #             # class_idx=33
    #         )
    #         heatmaps_sum_all(
    #             conv_maps,
    #             show_img_path=show_img_path,
    #             img_paths=img_paths,
    #             alpha=alpha,
    #             save_path=save_path,
    #         )

            # heatmaps_frequency_sum_all(
            #     conv_maps,
            #     show_img_path=show_img_path,
            #     img_paths=img_paths,
            #     alpha=alpha,
            #     save_path=save_path,
            # )
        