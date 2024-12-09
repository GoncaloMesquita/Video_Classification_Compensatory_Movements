import tf2onnx
import onnx
import tensorflow as tf
from onnx2pytorch import ConvertModel
# Convert the model to ONNX format

loaded_model = tf.keras.models.load_model("models/AcT_large.h5")
onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)


# Convert ONNX model to PyTorch
pytorch_model = ConvertModel(onnx_model)
pytorch_model 