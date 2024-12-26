import time

from predict_image.trt_runtime import TtrRuntime

def trtPredict(model_path, input_batch):
    trt_model = TtrRuntime(model_path)
    batch_size = trt_model.profile_max_batch
    time_start = time.time()
    data = trt_model.predict(input_batch)
    print("time for inference: %.4fs"%(time.time() - time_start))
    label = data["indices"][:, 0]
    pros = data["values"]

    return label, pros