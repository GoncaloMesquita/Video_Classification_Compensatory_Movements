import tf2onnx
import tensorflow as tf
# Load your TensorFlow model
# tensorflow_model = "models/AcT_large.h5"
tensorflow_model = tf.keras.models.load_model("models/AcT_large.h5")

# Convert to ONNX
onnx_model, _ = tf2onnx.convert.from_keras(tensorflow_model, opset=13)

# Save the ONNX model
with open("models/model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
    
    
