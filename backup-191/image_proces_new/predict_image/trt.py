import numpy as np
import tensorrt as trt
import pycuda.driver as cuda

class TtrRuntime:
    def __init__(self, file, num_classes, target_dtype=np.float16):
        self.target_dtype = target_dtype
        self.num_classes = num_classes

        self.load(file)

        self.stream = None

    def load(self, file):
        f = open(file, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))

        self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

    def allocate_memory(self, batch):
        batch_size = batch.shape[0]
        self.output = np.empty(
            (batch_size, self.num_classes), dtype=self.target_dtype
        )  # Need to set both input and output precisions to FP16 to fully enable FP16
        
        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)

        self.bindings = [int(self.d_input), int(self.d_output)]

        self.stream = cuda.Stream()

    def predict(self, batch):  # result gets copied into output
        B, H, W, C = batch.shape
        self.context.set_binding_shape(self.engine.get_binding_index("input_1"), (B,H, W, C))
        if self.stream is None:
            self.allocate_memory(batch)
        # self.allocate_memory(batch)
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model
        self.context.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

        return self.output