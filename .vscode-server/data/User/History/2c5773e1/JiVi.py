import tf2onnx
import tensorflow as tf
from models.transformer import TransformerEncoder, PatchClassEmbedding
# Load your TensorFlow model
# tensorflow_model = "models/AcT_large.h5"


def build_act( transformer, d_model, mlp_head_size=512):
    inputs = tf.keras.layers.Input(shape=(31, 52))
    x = tf.keras.layers.Dense(d_model)(inputs)
    x = PatchClassEmbedding(d_model, 31, 
                            pos_emb=None)(x)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
    x = tf.keras.layers.Dense(mlp_head_size)(x)
    outputs = tf.keras.layers.Dense(400)(x)
    return tf.keras.models.Model(inputs, outputs)


transformer = TransformerEncoder(64 * 4, 4, 64*4*4, 0.4, tf.nn.gelu, 6)
print(transformer)
model = build_act(transformer, 64 * 4, 512)

# tensorflow_model = tf.keras.models.load_model("models/AcT_large.h5")
model.load_weights("models/AcT_large.h5")

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(model, opset=13)

# Save the ONNX model
with open("models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
    
    
