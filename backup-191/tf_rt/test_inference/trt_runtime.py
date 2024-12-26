import numpy as np
import tensorrt as trt
from cuda import cudart


def topk_2d(arr, k):
    """
    返回二维数组中每行前 k 大的元素及其索引

    参数:
    arr (numpy.ndarray): 输入二维数组
    k (int): 要返回的最大元素的数量

    返回:
    tuple: 包含每行前 k 大元素的数组和对应的索引
    """
    if k <= 0 or k > arr.shape[1]:
        raise ValueError("k 必须在 1 和数组列数之间")
    
    # 获取排序后的索引
    sorted_indices = np.argsort(-arr, axis=1)[:, :k]
    # 获取每行的前 k 个元素
    topk_elements = np.take_along_axis(arr, sorted_indices, axis=1)
    
    return topk_elements, sorted_indices


# cpu和gpu上的地址
class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TtrRuntime:
    def __init__(self, trt_file, batch_size=1):
        self.trt_file = trt_file
        self.batch_size = batch_size

        # trt engine创建前首先初始化cuda上下文
        # self.cfx = cuda.Device(0).make_context()

        # load the engine from a file
        self.engine = self.load_engine()
        self.context = self.engine.create_execution_context()

        # 获取engine的input,output尺寸
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.profile_max_batch = None
        self.infer_shape()

        # Allocate device memory
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = None
        self.allocate_buffers()

    def __del__(self):
        cudart.cudaFree(self.inputs[0].device)
        cudart.cudaFree(self.outputs[0].device)
        cudart.cudaStreamDestroy(self.stream)
        # del self.inputs
        # del self.outputs
        # del self.stream
        # self.cfx.detach()

    def load_engine(self):
        # self.cfx.push()

        # Deserializing a Plan
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(self.trt_file, "rb") as f:
            serialized_engine = f.read()
        return runtime.deserialize_cuda_engine(serialized_engine)

    def infer_shape(self):
        # self.cfx.push()

        for binding in self.engine:
            if self.engine.binding_is_input(binding):
                self.input_shape = self.engine.get_binding_shape(binding)
                self.profile_max_batch = self.engine.get_profile_shape(0, binding)[-1][0]
                self.input_dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            else:
                self.output_shape = self.engine.get_binding_shape(binding)
        # self.cfx.pop()

    def allocate_buffers(self):
        # self.cfx.push()
        _, self.stream = cudart.cudaStreamCreate()
        for binding in self.engine:
            # 输入尺寸与类型
            size = abs(trt.volume(self.engine.get_binding_shape(binding)) * self.profile_max_batch)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = np.empty(size, dtype=dtype)
            _, device_mem = cudart.cudaMallocAsync(host_mem.nbytes, self.stream)
            # host_mem = cuda.pagelocked_empty(size, dtype)
            # device_mem = cuda.mem_alloc(host_mem.nbytes)

            # Append the device buffer to device bindings.
            self.bindings.append(int(device_mem))

            # Append to the appropriate list.
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                self.outputs.append(HostDeviceMem(host_mem, device_mem))

        # self.cfx.pop()

    def do_inference(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cudart.cudaMemcpyAsync(inp.device, inp.host.ctypes.data, inp.host.nbytes,
                                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream) for inp in inputs]

        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream)

        # Transfer predictions back from the GPU.
        [cudart.cudaMemcpyAsync(out.host.ctypes.data, out.device, out.host.nbytes,
                                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream) for out in outputs]

        # Synchronize the stream
        # stream.synchronize()
        cudart.cudaStreamSynchronize(self.stream)

        # Return only the host outputs.
        return [out.host for out in outputs]

    def pre_process(self, data):
        dim = np.ndim(data)
        if dim == 3:
            data = np.expand_dims(data, axis=-1)
        if self.input_dtype == np.float16:
            data = data.astype(np.float16)
        else:
            data = data.astype(np.float32)

        data = np.ascontiguousarray(data)
        return data

    def post_process(self, data, batch_size):
        data = data[0]
        class_num = self.output_shape[-1]
        data = data.reshape(-1, class_num)
        data = data[:batch_size, :]

        values, indices = topk_2d(data, 2)

        post_data = {"values": values, "indices": indices}

        return post_data

    def predict(self, data):
        # 推理前执行cfx.push()
        # self.cfx.push() 

        data = self.pre_process(data)
        self.inputs[0].host = data.ravel()
        batch_shape = data.shape
        B = batch_shape[0]
        H = batch_shape[1]
        W = batch_shape[2]
        C = batch_shape[3]
        self.context.set_binding_shape(self.engine.get_binding_index("input_1"), (B, H, W, C))
        trt_outputs = self.do_inference(self.context, bindings=self.bindings,
                                        inputs=self.inputs,
                                        outputs=self.outputs,
                                        stream=self.stream)

        post_data = self.post_process(trt_outputs, B)

        # 推理后执行cfx.pop()
        # self.cfx.pop()  

        return post_data


if __name__ == "__main__":
    trt_runtime = TtrRuntime("model/1Q42E-D_0606.trt", batch_size=12)
