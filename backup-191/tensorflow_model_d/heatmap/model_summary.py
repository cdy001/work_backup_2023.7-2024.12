import tensorflow as tf


if __name__ == "__main__":
    model_path = "heatmap/models/2X77G-0823.h5"
    model = tf.keras.models.load_model(model_path)
    h, w = model.input.type_spec.shape[-3:-1]
    print(model.input.type_spec.shape)
    print(f"h:{h}, w:{w}")
    fc_weights = model.layers[-1].weights[0]
    model.summary()