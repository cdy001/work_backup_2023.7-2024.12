import os
import sys
sys.path.append(os.getcwd())
import traceback
import time
import glob
import tensorrt as trt
from multiprocessing import Pool
import tensorflow as tf

from tf2trt.model_transform.model_transform_v2 import TF_TRT, compute_max_batchsize


MODEL_TRANSFER_DIR = "models_label/error_models"

class TfTrt(TF_TRT):
    def __init__(self, tf_model_path, trt_model_path, max_batch_size=None, save_onnx=False, fp16_mode=False, gpu_id=0) -> None:
        super().__init__(tf_model_path, max_batch_size, save_onnx, fp16_mode, gpu_id)
        self.trt_model_path = trt_model_path

        # try:
        #     self.tf_model = tf.keras.models.load_model(tf_model_path)
        #     print(f"Model {tf_model_path} loaded successfully!")
        #     # 如果成功，可以返回模型或者做其他操作
        # except Exception as e:
        #     print(f"Failed to load model: {e}")
        # finally:
        #     # 清理资源，避免显存泄漏
        #     tf.keras.backend.clear_session()

    def _onnx2trt(self,):
        height, width, channels = self.tf_model.input_shape[1:]
        max_batch_size = self.max_batch_size if self.max_batch_size else compute_max_batchsize((height, width))
        # trt_path = f"{os.path.splitext(self.tf_model_path)[0]}_b{max_batch_size}.trt"
        # 设置 TensorRT 记录器
        logger = trt.Logger(trt.Logger.WARNING)
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # create a builder
        # create a network definition
        # create an ONNX parser
        with trt.Builder(logger) as builder, \
            builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, logger) as parser:

            # 解析onnx文件，填充计算图
            print("Begining onnx file parsing")
            parser.parse(self.onnx_model_bytes)
            print("Completed parsing of onnx file")

            # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
            print("Building an engine from onnx file this may take a while...")
            # 创建构建器配置
            config = builder.create_builder_config()
            
            # 是否为fp16
            if self.fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            # 动态推理
            min_shape = (1, height, width, channels)
            opt_shape = (int(max_batch_size/2), height, width, channels)
            max_shape = (max_batch_size, height, width, channels)
            profile = builder.create_optimization_profile()
            profile.set_shape("input_1", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            # 设置工作空间内存池的限制为1GB
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
            # Builds an engine for the given INetworkDefinition and IBuilderConfig
            engine =  builder.build_serialized_network(network, config)
            print("Completed creating Engine")

            # 序列化
            with open(self.trt_model_path, 'wb') as f:
                f.write(engine)

def transfer_trt_worker():
    while 1:
        for tf_model_file in glob.glob(os.path.join(MODEL_TRANSFER_DIR, "*.h5")):
            tf_model_file_name = os.path.split(tf_model_file)[-1]
            model_prefix, suffix = os.path.splitext(tf_model_file_name)
            trt_model_name = f"{model_prefix}.trt"
            trt_model_file = os.path.join(MODEL_TRANSFER_DIR, trt_model_name)
            if not os.path.exists(trt_model_file):
                try:
                    # 防止模型文件未上传完成
                    time.sleep(1)
                    TfTrt(tf_model_path=tf_model_file, trt_model_path=trt_model_file).transform()
                except:
                    print(traceback.format_exc())
                    time.sleep(0.1)
                    continue
        time.sleep(0.1)



if __name__ == "__main__":
    process_number = 2
    pool = Pool(process_number)
    # 开启模型转换worker
    pool.apply_async(transfer_trt_worker)

    pool.close()
    pool.join()
