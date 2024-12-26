import tensorflow as tf
from tensorflow import keras

class ChannelAttention(keras.layers.Layer):
    def __init__(self, ratio=16):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio

    def build(self, input_shape):
        self.filters = input_shape[-1]

        self.shared_layer = keras.layers.Dense(self.filters // self.ratio, activation='relu')
        self.channel_attention = keras.layers.Dense(self.filters, activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_pool = self.shared_layer(avg_pool)
        max_pool = self.shared_layer(max_pool)

        channel_attention = self.channel_attention(avg_pool + max_pool)
        return inputs * channel_attention

class SpatialAttention(keras.layers.Layer):
    def __init__(self):
        super(SpatialAttention, self).__init__()

    def build(self, input_shape):
        self.filters = input_shape[-1]

        self.convolution = keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')

    def call(self, inputs):
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        spatial_attention = self.convolution(tf.concat([avg_pool, max_pool], axis=-1))
        return inputs * spatial_attention

class CBAMLayer(keras.layers.Layer):
    def __init__(self, ratio=16):
        super(CBAMLayer, self).__init__()
        self.channel_attention = ChannelAttention(ratio=ratio)
        self.spatial_attention = SpatialAttention()

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

class SelfAttentionLayer(keras.layers.Layer):
    def __init__(self, units=256):
        super(SelfAttentionLayer, self).__init__()
        self.units = units
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units
        })
        return config

    def build(self, input_shape):
        self.Wq = self.add_weight(name='Wq',
                                 shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wk = self.add_weight(name='Wk',
                                 shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wv = self.add_weight(name='Wv',
                                 shape=(input_shape[-1], self.units),
                                 initializer='glorot_uniform',
                                 trainable=True)

    def call(self, inputs):
        q = tf.matmul(inputs, self.Wq)
        k = tf.matmul(inputs, self.Wk)
        v = tf.matmul(inputs, self.Wv)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)
        
        output = tf.matmul(attention_scores, v)
        return output

def se_block(input_tensor, reduction_ratio=16):
    channels = input_tensor.shape[-1]
    
    # Global average pooling
    squeeze = keras.layers.GlobalAvgPool2D()(input_tensor)
    
    # Excitation
    excitation = keras.layers.Dense(channels // reduction_ratio, activation='relu')(squeeze)
    excitation = keras.layers.Dense(channels, activation='sigmoid')(excitation)
    excitation = keras.layers.Reshape((1, 1, channels))(excitation)
    
    # Scale input by excitation
    scaled_input = keras.layers.Multiply()([input_tensor, excitation])
    
    return scaled_input

# 构建模型
def build_model(img_shape, weight_path, num_class):
    img_shape = (img_shape[0], img_shape[1], img_shape[2])

    # base_model = tf.keras.applications.EfficientNetB1(input_shape=img_shape, include_top=False, weights=None)
    # base_model = tf.keras.applications.efficientnet.EfficientNetB1(input_shape=img_shape, include_top=False, weights=None)
    base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(input_shape=img_shape, include_top=False, weights=None)
    # base_model = tf.keras.applications.EfficientNetV2B1(input_shape=img_shape, include_top=False, weights=None)
    base_model.trainable = True
    base_model.load_weights(weight_path, by_name=True)

    x = base_model.output
    # x = se_block(base_model.output)  # se attention
    # x = SelfAttentionLayer()(base_model.output)  # sparse self-attention
    # x = CBAMLayer()(base_model.output)  # CBAM attention

    x = tf.keras.layers.GlobalAvgPool2D()(x)
    prediction_layer = tf.keras.layers.Dense(num_class, activation='softmax')(x)

    new_model = tf.keras.Model(inputs=base_model.input, outputs=prediction_layer)

    return new_model

if __name__ == '__main__':
    weight_path = 'tmp/0945U_EfficientNet-v2.h5'
    num_class = 40
    img_shape = [136, 592, 1]
    model = build_model(img_shape, weight_path, num_class)
    model.summary()
