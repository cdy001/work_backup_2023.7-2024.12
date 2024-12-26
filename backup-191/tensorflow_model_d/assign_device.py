import tensorflow as tf


def assign_gpus(specify_gpu_list):
    # Return a list of physical devices visible to the host runtime.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Num GPUs: {len(gpus)}")

    # 指定gpu
    # Visible devices must be set before GPUs have been initialized
    visible_gpus = []
    for i in specify_gpu_list:
        visible_gpus.append(gpus[i])
    tf.config.experimental.set_visible_devices(visible_gpus, 'GPU')
    print(f"Num GPUs of visible: {len(visible_gpus)}")

    # memory growth
    for visible_gpu in visible_gpus:
        # Memory growth must be set before GPUs have been initialized
        tf.config.experimental.set_memory_growth(visible_gpu, True)


assign_gpus([0])



