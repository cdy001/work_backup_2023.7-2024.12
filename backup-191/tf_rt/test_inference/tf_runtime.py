# 三方包
import numpy as np
import tensorflow as tf


class TfRuntime:
    def __init__(self, model_path, batch_size=1):
        self.model = tf.keras.models.load_model(model_path)
        self.batch_size = batch_size

    def __del__(self):
        del self.model

    @staticmethod
    def pre_process(data):
        dim = np.ndim(data)
        if dim == 3:
            data = np.expand_dims(data, axis=-1)
        data = data.astype(np.float32)
        return data

    @staticmethod
    def post_process(data):
        values, indices = tf.math.top_k(data, 2)
        post_data = {"values": values.numpy(), "indices": indices.numpy()}
        return post_data
    
    def predict(self, data):
        data = self.pre_process(data)
        pre = self.model(data, training=False)
        post_data = self.post_process(pre)
        return post_data


if __name__ == "__main__":
    tf_runtime = TfRuntime("model/1012_40_epoch.h5")
