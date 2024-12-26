import os
import tf2onnx
import tensorflow as tf
import tensorrt as trt
import argparse


class TF_TRT:
    def __init__(self, tf_model_path, max_batch_size, save_onnx=False, fp16_mode=False, gpu_id=0) -> None:
        # 设置GPU内存增长
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        gpus = tf.config.experimental.list_physical_devices(device_type="GPU")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        self.tf_model = tf.keras.models.load_model(tf_model_path)
        self.max_batch_size = max_batch_size
        self.save_onnx_path = f"{os.path.splitext(tf_model_path)[0]}.onnx" \
        if save_onnx else None
        self.trt_path = f"{os.path.splitext(tf_model_path)[0]}_b{max_batch_size}.trt"
        self.fp16_mode = fp16_mode

    def __tf2onnx(self,):
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

    def __onnx2trt(self,):
        height, width, channels = self.tf_model.input_shape[1:]
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
            opt_shape = (int(self.max_batch_size/2), height, width, channels)
            max_shape = (self.max_batch_size, height, width, channels)
            profile = builder.create_optimization_profile()
            profile.set_shape("input_1", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

            # 设置工作空间内存池的限制为1GB
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
            # Builds an engine for the given INetworkDefinition and IBuilderConfig
            engine =  builder.build_serialized_network(network, config)
            print("Completed creating Engine")

            # 序列化
            with open(self.trt_path, 'wb') as f:
                # f.write(engine.serialize())
                f.write(engine)

    def transform(self,):
        self.__tf2onnx()
        self.__onnx2trt()

def args_config():
    parser = argparse.ArgumentParser(description="argparse for input.")
    parser.add_argument('--path', type=str, required=True, help="tf model path")
    parser.add_argument('--max_batch_size', type=int, required=True, help="max batch to inference every time")
    parser.add_argument('--save_onnx', type=bool, required=False, default=False, help="whether to save onnx model or not")
    parser.add_argument('--fp16_mode', type=bool, required=False, default=False, help="whether to convert the model precision to fp16")
    parser.add_argument('--gpu', type=int, required=False, default=0, help="which gpu to use")

    args = parser.parse_args()

    return args

def main():
    args = args_config()
    tf_trt = TF_TRT(
        tf_model_path=args.path,
        max_batch_size=args.max_batch_size,
        save_onnx=args.save_onnx,
        fp16_mode=args.fp16_mode,
        gpu_id=args.gpu
    )
    tf_trt.transform()

if __name__ == '__main__':
    main()