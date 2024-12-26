"""
1.What format should I save my model in?
2.What batch size(s) am I running inference at?
3.What precision am I running inference at?
4.What TensorRT path am I using to convert my model?
5.What runtime am I targeting?
"""
import os
import onnx
import tf2onnx
import subprocess
import tensorrt as trt
import tensorflow as tf
import time


class Tf2onnx2trt:
    def __init__(self, model_path, batch_size, d_dtype, gpu_device_number):
        self.model_path = model_path

        self.batch_size = batch_size
        self.d_dtype = d_dtype

        self.device = gpu_device_number

        self.onnx_path = os.path.splitext(self.model_path)[0] + ".onnx"
        # self.trt_path = os.path.splitext(self.model_path)[0] + ".trt"

    # tf模型转onnx模型
    def tf2onnx_function(self):
        # 禁用GPU
        tf.config.experimental.set_visible_devices([], 'GPU')

        model = tf.keras.models.load_model(self.model_path)

        model_proto, _ = tf2onnx.convert.from_keras(model, opset=13)

        onnx.save_model(model_proto, self.onnx_path)

        print("onnx done saving!")

    # onnx模型转tensorrt引擎文件
    def onnx2trt_function(self, img_size):
        H, W, C = img_size
        min_batch_size = 1
        normal_batch_size = self.batch_size
        max_batch_size = normal_batch_size + 16
        if self.d_dtype == "fp16":
            trt_path = os.path.splitext(self.model_path)[0] + "_fp16.trt"
            # commands = f"/root/TensorRT-8.4.0.6/bin/trtexec --onnx={self.onnx_path} --saveEngine={self.trt_path}  --explicitBatch --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
            commands = f"/root/TensorRT-8.4.0.6/bin/trtexec \
                        --onnx={self.onnx_path} \
                        --saveEngine={trt_path} \
                        --minShapes=input_1:{min_batch_size}x{H}x{W}x{C} \
                        --optShapes=input_1:{normal_batch_size}x{H}x{W}x{C} \
                        --maxShapes=input_1:{max_batch_size}x{H}x{W}x{C} \
                        --shapes=input_1:{normal_batch_size}x{H}x{W}x{C} \
                        --inputIOFormats=fp16:chw \
                        --outputIOFormats=fp16:chw \
                        --fp16 \
                        --device={self.device}"
        else:
            trt_path = os.path.splitext(self.model_path)[0] + "_fp32.trt"
            # commands = f"/root/TensorRT-8.4.0.6/bin/trtexec --onnx={self.onnx_path} --saveEngine={self.trt_path}  --explicitBatch"
            commands = f"/root/TensorRT-8.4.0.6/bin/trtexec \
                        --onnx={self.onnx_path} \
                        --saveEngine={trt_path} \
                        --minShapes=input_1:{min_batch_size}x{H}x{W}x{C} \
                        --optShapes=input_1:{normal_batch_size}x{H}x{W}x{C} \
                        --maxShapes=input_1:{max_batch_size}x{H}x{W}x{C} \
                        --shapes=input_1:{normal_batch_size}x{H}x{W}x{C} \
                        --device={self.device}"
        # print(commands)

        time_start = time.time()
        ret = subprocess.run(commands, shell=True)
        time_end = time.time()

        # print(ret)
        print("engine done saving! time consuming: %.2fs"%(time_end-time_start))


    def convert_onnx_to_engine(
        onnx_filename,
        engine_filename=None,
        max_batch_size=32,
        max_workspace_size=1 << 30,
        fp16_mode=True,
    ):
        logger = trt.Logger(trt.Logger.WARNING)
        with trt.Builder(
            logger
        ) as builder, builder.create_network() as network, trt.OnnxParser(
            network, logger
        ) as parser:
            builder.max_workspace_size = max_workspace_size
            builder.fp16_mode = fp16_mode
            builder.max_batch_size = max_batch_size

            print("Parsing ONNX file.")
            with open(onnx_filename, "rb") as model:
                if not parser.parse(model.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))

            print("Building TensorRT engine. This may take a few minutes.")
            engine = builder.build_cuda_engine(network)

            if engine_filename:
                with open(engine_filename, "wb") as f:
                    f.write(engine.serialize())

            return (engine,)


if __name__ == "__main__":
    model_path = "model/1O42A-0701.h5"
    batch_size = 96
    # img_size = [128, 608, 1]  # 0945A
    # img_size = [134, 588, 1]  # 0945Y
    # img_size = [606, 606, 1]  # BOB1J37H-AA
    # img_size = [682, 682, 1]  # BGB1Q42E
    d_dtype = "fp32"
    gpu_device_id = 1  # GPU ID

    tf2onnx2trt = Tf2onnx2trt(model_path, batch_size, d_dtype, gpu_device_id)

    tf2onnx2trt.tf2onnx_function()

    # tf2onnx2trt.onnx2trt_function(img_size=img_size)
    
