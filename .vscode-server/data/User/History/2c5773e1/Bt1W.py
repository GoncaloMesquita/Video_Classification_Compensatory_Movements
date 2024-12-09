import tf2onnx
import onnx
import tensorflow as tf
from onnx2pytorch import ConvertModel
import torch

loaded_model = tf.keras.models.load_model("models/AcT_large.h5")

onnx_model, _ = tf2onnx.convert.from_keras(loaded_model)

# Convert ONNX model to PyTorch
pytorch_model = ConvertModel(onnx_model)
# torch.save(pytorch_model.state_dict(), "models/AcT_large.pth") 
for name, param in pytorch_model.named_parameters():
    print(f"Parameter name: {name}, shape: {param.shape}")