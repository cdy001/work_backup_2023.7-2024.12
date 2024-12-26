import time

from predict_image.tf_runtime import TfRuntime


def tfPredict(model_path, input_batch):
    tf_model = TfRuntime(model_path)
    time_start = time.time()
    data = tf_model.predict(input_batch)
    print("time for inference: %.4fs"%(time.time() - time_start))
    label = data["indices"][:, 0]
    pros = data["values"]
    
    return label, pros