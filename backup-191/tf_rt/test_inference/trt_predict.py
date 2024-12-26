import time

from test_inference.trt_runtime import TtrRuntime

def trtPredict(model_path, input_batch):
    trt_model = TtrRuntime(model_path)
    time_start = time.time()
    data = trt_model.predict(input_batch)
    print("time for inference: %.4fs"%(time.time() - time_start))
    label = data["indices"][:, 0]

    return label