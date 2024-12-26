import onnx
import tensorrt as trt
import networkx as nx

def inputeshape(onnx_path):
    onnx_model  = onnx.load(onnx_path)
    input_shapes = [{input.name:[dim.dim_value for dim in input.type.tensor_type.shape.dim]}
                for input in onnx_model.graph.input]
    height = input_shapes[0]["input_1"][1]
    width = input_shapes[0]["input_1"][2]
    channel = input_shapes[0]["input_1"][3]
    return height, width, channel

def calculate_onnx_model_size(onnx_model_path):
    model = onnx.load(onnx_model_path)
    tensor_sizes = []
    
    for tensor in model.graph.initializer:
        size = 1
        for dim in tensor.dims:
            size *= dim
        dtype = onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[tensor.data_type]
        dtype_size = dtype.nbytes
        tensor_sizes.append(size * dtype_size)
    
    return sum(tensor_sizes)

def calculate_intermediate_activation_size(network, max_batch_size):
    total_size = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        for j in range(layer.num_outputs):
            shape = layer.get_output(j).shape
            size = abs(trt.volume(shape))
            dtype = layer.get_output(j).dtype
            dtype_size = dtype.itemsize
            total_size += size * dtype_size
    return total_size


def calculate_memory(engine):
    memory = 0
    for i in range(engine.num_bindings):
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = engine.get_binding_shape(i)
        memory += trt.volume(shape) * dtype.itemsize
    return memory

def build_engine(onnx_file_path, max_batch_size):
    height, width, channels = inputeshape(onnx_file_path)
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Parse the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # 创建builder config
        config = builder.create_builder_config()

        # 估算网络参数量
        # total_params = count_parameters(network)
        # print(f'Total parameters: {total_params}')
        
        # Assuming the first input is the dynamic input
        input_name = network.get_input(0).name
        min_shape = (1, height, width, channels)
        opt_shape = (int(max_batch_size/2), height, width, channels)
        max_shape = (max_batch_size, height, width, channels)

        # Set up optimization profiles for dynamic input shapes
        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
        
        config.add_optimization_profile(profile)

        print(builder.max_DLA_batch_size)
        
        # # 创建推理引擎
        # engine = builder.build_engine(network, config)
        # # 计算推理时的显存占用
        # memory_usage = calculate_memory(engine)
        # print(f'Memory usage during inference: {memory_usage / (1024 ** 2)} MB')

        # Calculate memory usage
        # activation_size = calculate_intermediate_activation_size(network, max_batch_size)
        # total_memory_usage = activation_size
        # print(f"memory usage: {total_memory_usage // (1024 ** 2)} MB")

        # # 设置工作空间内存池的限制为1GB
        # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GiB
        # # Builds an engine for the given INetworkDefinition and IBuilderConfig
        # engine =  builder.build_serialized_network(network, config)

        # return memory_usage

def main(onnx_file_path, max_batch_size):
    build_engine(onnx_file_path, max_batch_size)
    
if __name__ == "__main__":
    onnx_file_path = 'models_label/1C30D-0801.onnx'  # Replace with your ONNX model file path
    max_batch_size = 48
    main(onnx_file_path, max_batch_size)
