import tensorrt as trt
import numpy as np

global profile_max_batch

def get_binding_size(engine, tensor):
    global profile_max_batch
    size = abs(
            trt.volume(engine.get_tensor_shape(tensor))
            * profile_max_batch
            )
    dtype = engine.get_tensor_dtype(tensor)
    dtype_size = dtype.itemsize
    byte_size = size * dtype_size

    return byte_size

def main(engine_path):
    global profile_max_batch
    # Load the TensorRT engine
    with open(engine_path, 'rb') as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    device_memory_size = engine.device_memory_size
    print(f"engine size: {device_memory_size} bytes")

    # Calculate input and output sizes
    for i, tensor in enumerate(engine):
        if engine.get_tensor_mode(tensor) == trt.TensorIOMode.INPUT:
            profile_max_batch = engine.get_tensor_profile_shape(tensor, 0)[-1][0]
            size_in_bytes = get_binding_size(engine, tensor)
            print(f"Input '{tensor}' size: {size_in_bytes} bytes")
        else:
            size_in_bytes = get_binding_size(engine, tensor)
            print(f"Output '{tensor}' size: {size_in_bytes} bytes")
        # print(engine.get_binding_name(i), engine.binding_is_input(binding))  # binding名字，判断是否输入
        # print(engine.get_profile_shape(i, tensor))  # profile形状
        # print(engine.get_binding_shape(binding))  # binding形状
        # print(engine.get_tensor_shape(binding), engine.get_tensor_dtype(binding), engine.get_tensor_format_desc(binding))
        # print(engine.serialize())
        # print(engine.get_tensor_mode(tensor))

if __name__ == "__main__":
    engine_path = "models_label/1Q42E-D_0731_b24.trt"  # Replace with your TensorRT engine file path
    main(engine_path)
