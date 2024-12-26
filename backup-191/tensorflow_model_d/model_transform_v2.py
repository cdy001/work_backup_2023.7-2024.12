import os
import tf2onnx
import tensorflow as tf
import tensorrt as trt
import argparse
from cuda import cudart

def compute_max_batchsize(img_size):
    '''
    reference(precision=fp32):
        1. batch * height * width * channel = 24 * 500 * 500 * 1  (std_para)
            engine.device_memory_size = 528,004,608 bytes  (std_dev_mem)
            inference_mem = 822 MB = 861,929,472 bytes = dev_mem + 333,924,864 bytes
        2. batch * height * width * channel = 36 * 500 * 500 * 1  (para)
            engine.device_memory_size = 792007168 bytes = 1.5 * std_dev_mem
                                                        = (para / std_para) * std_dev_mem
            inference_mem = 1106 MB = 1,159,725,056 bytes = dev_mem + 367,717,888 bytes
        3. batch * height * width * channel = 24 * 714 * 714 * 1  (para)
            engine.device_memory_size = 1,081,635,840 bytes = 2.0485 * std_dev_mem
                                                            = (para / std_para) * std_dev_mem
            inference_mem = 1382 MB = 1,449,132,032 bytes = dev_mem + 367,496,192 bytes
        4. batch * height * width * channel = 24 * 682 * 682 * 1  (para)
            engine.device_memory_size = 987,067,392 bytes = 1.8694 * std_dev_mem
                                                          = (para / std_para) * std_dev_mem
            inference_mem = 1292 MB = 1,354,760,192 bytes = dev_mem + 367,692,800 bytes    
    '''
    std_dev_mem = 528004608
    device_id = cudart.cudaGetDevice()
    _, free_mem, total_mem = cudart.cudaMemGetInfo()
    # print(f"device: {device_id}, free_memory: {free_mem}, total_memory: {total_mem}")
    mem_per_process = total_mem // 6
    engine_dev_mem = mem_per_process - 5e+8
    # print(engine_dev_mem // (1024 ** 2))
    img_size_coefficient = (img_size[0] * img_size[1]) / (500 * 500)
    batchsize = 24 * engine_dev_mem / (std_dev_mem * img_size_coefficient)
    # print(batchsize)
    batchsize = int((batchsize // 4) * 4)
    # print(batchsize)

    return batchsize

class TF_TRT:
    def __init__(self, tf_model_path, max_batch_size=None, save_onnx=False, fp16_mode=False, gpu_id=0) -> None:
        # 设置GPU内存增长
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.tf_model_path = tf_model_path
        self.tf_model = tf.keras.models.load_model(tf_model_path)
        self.max_batch_size = max_batch_size
        self.save_onnx_path = f"{os.path.splitext(tf_model_path)[0]}.onnx" if save_onnx else None        
        self.fp16_mode = fp16_mode

    def _tf2onnx(self,):
        height, width, channels = self.tf_model.input_shape[1:]
        # 转换为 ONNX 模型并保持在内存中
        spec = (tf.TensorSpec((None, height, width, channels), tf.float32, name="input_1"),)
        model_proto, _ = tf2onnx.convert.from_keras(
            self.tf_model,
            input_signature=spec,
            output_path=self.save_onnx_path)
        if self.save_onnx_path:
            print(f"ONNX模型已保存至{self.save_onnx_path}")
        # 将模型序列化为字节流
        self.onnx_model_bytes = model_proto.SerializeToString()
        print("ONNX 模型已转换为字节流")

    def _onnx2trt(self,):
        height, width, channels = self.tf_model.input_shape[1:]
        max_batch_size = self.max_batch_size if self.max_batch_size else compute_max_batchsize((height, width))
        trt_path = f"{os.path.splitext(self.tf_model_path)[0]}_b{max_batch_size}.trt"
        # 设置 TensorRT 记录器
        logger = trt.Logger(trt.Logger.WARNING)
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

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
            with open(trt_path, 'wb') as f:
                f.write(engine)

    def transform(self,):
        self._tf2onnx()
        self._onnx2trt()

def args_config():
    parser = argparse.ArgumentParser(description="argparse for input.")
    parser.add_argument('--path', type=str, required=True, help="tf model path")
    parser.add_argument('--max_batch_size', type=int, required=False, default=None, help="max batch to inference every time")
    parser.add_argument('--save_onnx', type=bool, required=False, default=False, help="whether to save onnx model or not")
    parser.add_argument('--fp16', type=bool, required=False, default=False, help="whether to convert the model precision to fp16")
    parser.add_argument('--gpu', type=int, required=False, default=0, help="which gpu to use")

    args = parser.parse_args()

    return args

def main():
    args = args_config()
    tf_trt = TF_TRT(
        tf_model_path=args.path,
        max_batch_size=args.max_batch_size,
        save_onnx=args.save_onnx,
        fp16_mode=args.fp16,
        gpu_id=args.gpu
    )
    tf_trt.transform()

if __name__ == '__main__':
    main()