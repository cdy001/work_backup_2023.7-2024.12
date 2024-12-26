import numpy as np
import tensorrt as trt
from skimage.transform import resize
from skimage import io
import time
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
        self.d_input = cuda.mem_alloc(batch.nbytes)
        self.d_output = cuda.mem_alloc(self.output.nbytes)

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
    
if __name__ == '__main__':
    trt_model = TtrRuntime("model/1Q42E-D_0606_fp32.trt", 42, target_dtype=np.float32)
    '''
    # Initialize CUDA
    cuda.init()
    device = cuda.Device(1)
    context = device.make_context()

    url = "R11C16_31_L1#804_227_1414_836_0_0_0_0.bmp"
    import cv2
    img = cv2.imread(url, flags=-1)
    # die_mat = []
    # die_coordinate = (
    #     [1267, 2977, 1882, 3587],
    #     [504, 2223, 1122, 2843],
    #     [2014, 2221, 2629, 2833],
    #     [4276, 3701, 4880, 4304]
    #     )
    # for coodinate in die_coordinate:
    #     x_min, y_min, x_max, y_max = coodinate
    #     img_die = img[y_min:y_max, x_min:x_max]
    #     die_mat.append(cv2.resize(img_die, (606, 606)))
    # die_mat = np.array(die_mat)
    # input_batch = np.expand_dims(die_mat, axis=-1).astype(np.float32)
    # print(input_batch.shape)
    img = cv2.resize(img, (606, 606))
    input_batch = np.array(
        np.repeat(np.expand_dims(np.array(img), axis=0), 64, axis=0),
        dtype=np.float32,
    )
    input_batch = np.expand_dims(input_batch, axis=-1)
    trt_model = TtrRuntime("model/BOB1-1218_fp32.trt", 42, target_dtype=np.float32)

    for i in range(100):
        start_time = time.time()
        pred_pro = trt_model.predict(input_batch)
        end_time = time.time()
        print("inference time: {:.4f}s".format((end_time - start_time)))
        label = np.argmax(pred_pro, axis=1)
        print(label)
        # print(label.shape, pred_pro.shape)

    context.pop()
    from pycuda.tools import clear_context_caches
    clear_context_caches()
    '''