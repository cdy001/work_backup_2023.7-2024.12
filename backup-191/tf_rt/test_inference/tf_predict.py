import time

from test_inference.tf_runtime import TfRuntime


def tfPredict(model_path, input_batch):
    tf_model = TfRuntime(model_path)
    time_start = time.time()
    data = tf_model.predict(input_batch)
    print("time for inference: %.4fs"%(time.time() - time_start))
    label = data["indices"][:, 0]
    
    return label