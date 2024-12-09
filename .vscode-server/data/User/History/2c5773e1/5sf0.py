import tf2onnx
import tensorflow as tf
from Models.transformer import TransformerEncoder
# Load your TensorFlow model
# tensorflow_model = "models/AcT_large.h5"


def build_act(self, transformer):
    inputs = tf.keras.layers.Input(shape=(30, 18*3))
    x = tf.keras.layers.Dense(self.d_model)(inputs)
    x = PatchClassEmbedding(self.d_model, 30, 
                            pos_emb=None)(x)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
    x = tf.keras.layers.Dense(self.mlp_head_size)(x)
    outputs = tf.keras.layers.Dense(self.config[self.config['DATASET']]['CLASSES'])(x)
    return tf.keras.models.Model(inputs, outputs)

transformer = TransformerEncoder(64 * 4, 4, 64*4*4, 0.4, tf.nn.gelu, 6)
self.model = self.build_act(transformer)

tensorflow_model = tf.keras.models.load_model("models/AcT_large.h5")

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(tensorflow_model, opset=13)

# Save the ONNX model
with open("models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
    
    
