import tf2onnx
import onnx
from onnx2pytorch import ConvertModel
# Convert the model to ONNX format

loaded_model = tf.keras.models.load_model(&quot;iris_model.h5&quot;)
onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)


# Convert ONNX model to PyTorch
pytorch_model = ConvertModel(onnx_model)
pytorch_model 