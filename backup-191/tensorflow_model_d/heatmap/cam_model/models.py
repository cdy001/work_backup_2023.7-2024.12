from abc import abstractmethod

import tensorflow as tf


class BaseModel(tf.keras.Model):
    def __init__(self, model_path=None):
        super(BaseModel, self).__init__()
        self.original_model = tf.keras.models.load_model(model_path, compile=False)
        self.input_spec = self.original_model.input_spec

    @abstractmethod
    def _new_model(self, *args, **kwargs):        
        raise NotImplementedError

    def __call__(self, inputs):
        model = self._new_model()

        return model(inputs)


class ModelWithConvOut(BaseModel):
    '''
    This model is for Grad-CAM.
    '''
    def __init__(self, model_path=None, conv_layer_name="top_conv"):
        super().__init__(model_path)
        self.conv_layer_name = conv_layer_name
    def _new_model(self):
        model = tf.keras.Model(
            inputs=self.original_model.input,
            outputs=[
                self.original_model.get_layer(self.conv_layer_name).output,
                self.original_model.output
            ]
        )
        return model


class ModelWithConvOuts(BaseModel):
    '''
    This model is for Layer-CAM.
    '''
    def _new_model(self):
        model = tf.keras.Model(
            inputs=self.original_model.input,
            outputs=[
                [
                    self.original_model.get_layer("top_conv").output,
                    self.original_model.get_layer("block6i_add").output,
                    self.original_model.get_layer("block5f_add").output,
                    self.original_model.get_layer("block4d_add").output,
                    self.original_model.get_layer("block3c_add").output,
                    self.original_model.get_layer("block2c_add").output,
                    self.original_model.get_layer("block1b_add").output,
                ],
                self.original_model.output
            ]
        )
        return model