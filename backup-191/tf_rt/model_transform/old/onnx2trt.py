"""
1.What format should I save my model in?
2.What batch size(s) am I running inference at?
3.What precision am I running inference at?
4.What TensorRT path am I using to convert my model?
5.What runtime am I targeting?
"""
import os
import subprocess
import onnx
import tensorrt as trt


os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class OnnxTrt:
    def __init__(self, onnx_path, fp16_mode, max_batch_size):
        self.onnx_path = onnx_path
        self.fp16_mode = fp16_mode
        self.input_dynamic = True
        self.max_batch_size = max_batch_size
        if self.fp16_mode:
            self.trt_path = os.path.splitext(self.onnx_path)[0] + f"_b{max_batch_size}.trt"
        else:
            self.trt_path = os.path.splitext(self.onnx_path)[0] + f"_b{max_batch_size}.trt"

        self.inputeshape()

    def inputeshape(self):
        onnx_model  = onnx.load(self.onnx_path)
        input_shapes = [{input.name:[dim.dim_value for dim in input.type.tensor_type.shape.dim]}
                    for input in onnx_model.graph.input]
        self.height = input_shapes[0]["input_1"][1]
        self.weidth = input_shapes[0]["input_1"][2]
        self.channel = input_shapes[0]["input_1"][3]

    # onnx模型转tensorrt引擎文件
    def onnx2trt_trtexec(self):
        commands = f"/root/TensorRT-8.4.0.6/bin/trtexec --onnx={self.onnx_path} --saveEngine={self.trt_path}"

        if self.input_dynamic:
            middle_batch_size = int(self.max_batch_size/2)
            img_size = f"{self.height}x{self.weidth}x{self.channel}"
            commands = commands + f" --minShapes=input_1:1x{img_size} --optShapes=input_1:{middle_batch_size}x{img_size} --maxShapes=input_1:{self.max_batch_size}x{img_size} --shapes=input_1:{middle_batch_size}x{img_size}"
        else:
            commands = commands + " --explicitBatch"

        if self.fp16_mode:
            commands = commands + " --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --fp16"
        print(commands)

        ret = subprocess.run(commands, shell=True)
        print("engine done saving!")

    # onnx模型转tensorrt引擎文件
    def onnx2trt_function(self):
        logger = trt.Logger(trt.Logger.WARNING)
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        # create a builder
        # create a network definition
        # create an ONNX parser
        with trt.Builder(logger) as builder, builder.create_network(explicit_batch) as network, \
            trt.OnnxParser(network, logger) as parser:

            if not os.path.exists(self.onnx_path):
                quit("ONNX file {} not found!".format(self.onnx_path))
            print('loading onnx file from path {} ...'.format(self.onnx_path))
            # 解析onnx文件，填充计算图
            # parser.parse_from_file(onnx_file_path) # parser还有一个从文件解析onnx的方法
            with open(self.onnx_path, 'rb') as model:
                print("Begining onnx file parsing")
                parser.parse(model.read())
            print("Completed parsing of onnx file")

            # build configuration specifying how TensorRT should optimize the mode
            # 填充计算图完成后，则使用builder从计算图中创建CudaEngine
            print("Building an engine from file{}' this may take a while...".format(self.onnx_path))
            config = builder.create_builder_config()

            # 设置工作空间大小
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
            # 是否为fp16
            if self.fp16_mode:
                config.set_flag(trt.BuilderFlag.FP16)
            # 动态推理
            if self.input_dynamic:
                min_shape = (1, self.height, self.weidth, self.channel)
                opt_shape = (int(self.max_batch_size/2), self.height, self.weidth, self.channel)
                max_shape = (self.max_batch_size, self.height, self.weidth, self.channel)
                profile = builder.create_optimization_profile()
                profile.set_shape("input_1", min_shape, opt_shape, max_shape)
                config.add_optimization_profile(profile)

            # Builds an engine for the given INetworkDefinition and IBuilderConfig
            engine =  builder.build_engine(network, config)
            print("Completed creating Engine")

            # 序列化
            with open(self.trt_path, 'wb') as f:
                f.write(engine.serialize())


if __name__ == "__main__":
    onnx_path = "model/1O42A-0701.onnx"
    max_batch_size = 24
    fp16_mode = False

    onnxTtrt = OnnxTrt(onnx_path, fp16_mode, max_batch_size)

    # onnxTtrt.onnx2trt_trtexec()
    onnxTtrt.onnx2trt_function()