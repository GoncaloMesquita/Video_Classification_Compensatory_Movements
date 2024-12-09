from models.LSTM import LSTM
import torch
from models.AcT import AcT
from models.SkateFormer import SkateFormer
import torch.nn as nn

def create_model(model_name, input_size, hidden_size, num_layers, num_labels, dropout, checkpoint, mode, pretrained):
    # Import the necessary modules
    
    if model_name == 'LSTM':
        if mode == 'train':
            model = LSTM(input_size, hidden_size, num_layers, num_labels, dropout)
            
        if mode == 'test':
            model = LSTM(input_size, hidden_size, num_layers, num_labels, dropout)
            model.load_state_dict(torch.load(checkpoint), strict=True)
        
    elif model_name == 'AcT':
        if mode == 'train':
            model = AcT(dropout)
            checkpoints = torch.load("models/AcT_model_state_dict.pth", weights_only=True)
            
            for k in ['patch_embed.position_embedding.weight']:
                if k in checkpoints:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoints[k]
            
            model.load_state_dict(checkpoints, strict=False)
            model.dense1 = torch.nn.Linear(input_size, 256)
            model.final_dense = torch.nn.Linear(512, num_labels)
            
            # if pretrained:
                
            #     unfreeze_layers = ['dense1.weight', 'dense1.bias', 'final_dense.bias', 'final_dense.weight', 'patch_embed.position_embedding.weight']

            #     for name, param in model.named_parameters():
            #         if name not in unfreeze_layers:
            #             param.requires_grad = False
            #             print('Freez layer:', name)
        
            
        if mode == 'test':  
            model = AcT(dropout)
            model.dense1 = torch.nn.Linear(input_size,256)
            model.final_dense = torch.nn.Linear(512, num_labels)
            checkpoints = torch.load(checkpoint, weights_only=True)
            model.load_state_dict(checkpoints, strict=True)
            
    elif model_name == 'SkateFormer' :
        if mode == "train":
            model = SkateFormer(in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=6,
                 embed_dim=64, num_people=1, num_frames=64, num_points=50, kernel_size=7, num_heads=32,
                 type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1), type_4_size=(1, 1),
                 attn_drop=0., head_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=False, global_pool='avg')
            
            checkpoints = torch.load("models/SkateFormer_j.pt", weights_only=True)
            
            for k in ['head.weight', 'head.bias']:
                if k in checkpoints:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoints[k]
                    
            model.load_state_dict(checkpoints, strict=False)
                
    return model








# # Mapping TensorFlow weights to PyTorch

# # Helper function to map MHA, FFN, and LayerNorm weights
# def map_transformer_layer_weights(layer_idx, model_tf, state_dict):
#     # Get the layer weights from TensorFlow
#     weights = model_tf.get_layer('transformer_encoder').get_weights()
#     print(0+(16*layer_idx))
#     # Multi-Head Attention (MHA)
#     state_dict[f'transformer.layers.{layer_idx}.mha.wq.weight'] = torch.from_numpy(weights[0+(16*layer_idx)].T)
#     state_dict[f'transformer.layers.{layer_idx}.mha.wq.bias'] = torch.from_numpy(weights[1+(16*layer_idx)])

#     state_dict[f'transformer.layers.{layer_idx}.mha.wk.weight'] = torch.from_numpy(weights[2+(16*layer_idx)].T)
#     state_dict[f'transformer.layers.{layer_idx}.mha.wk.bias'] = torch.from_numpy(weights[3+(16*layer_idx)])

#     state_dict[f'transformer.layers.{layer_idx}.mha.wv.weight'] = torch.from_numpy(weights[4+(16*layer_idx)].T)
#     state_dict[f'transformer.layers.{layer_idx}.mha.wv.bias'] = torch.from_numpy(weights[5+(16*layer_idx)])

#     state_dict[f'transformer.layers.{layer_idx}.mha.dense.weight'] = torch.from_numpy(weights[6+(16*layer_idx)].T)
#     state_dict[f'transformer.layers.{layer_idx}.mha.dense.bias'] = torch.from_numpy(weights[7+(16*layer_idx)])

#     # Feedforward (FFN)
#     state_dict[f'transformer.layers.{layer_idx}.ffn.0.weight'] = torch.from_numpy(weights[8+(16*layer_idx)].T)
#     state_dict[f'transformer.layers.{layer_idx}.ffn.0.bias'] = torch.from_numpy(weights[9+(16*layer_idx)])

#     state_dict[f'transformer.layers.{layer_idx}.ffn.2.weight'] = torch.from_numpy(weights[10+(16*layer_idx)].T)
#     state_dict[f'transformer.layers.{layer_idx}.ffn.2.bias'] = torch.from_numpy(weights[11+(16*layer_idx)])

#     # Layer Normalization 1
#     state_dict[f'transformer.layers.{layer_idx}.layernorm1.weight'] = torch.from_numpy(weights[12+(16*layer_idx)])
#     state_dict[f'transformer.layers.{layer_idx}.layernorm1.bias'] = torch.from_numpy(weights[13+(16*layer_idx)])

#     # Layer Normalization 2
#     state_dict[f'transformer.layers.{layer_idx}.layernorm2.weight'] = torch.from_numpy(weights[14+(16*layer_idx)])
#     state_dict[f'transformer.layers.{layer_idx}.layernorm2.bias'] = torch.from_numpy(weights[15+(16*layer_idx)])

# # Map weights for each transformer layer (0-5)

# state_dict = model_new.state_dict()  # Get PyTorch model's state dict

# for i in range(6):  # 6 layers: 0-5
#     map_transformer_layer_weights(i, model_tf, state_dict)

# # MLP head
# state_dict['mlp_head.weight'] = torch.from_numpy(model_tf.get_layer('dense_37').get_weights()[0].T)
# state_dict['mlp_head.bias'] = torch.from_numpy(model_tf.get_layer('dense_37').get_weights()[1])

# # Final Dense layer
# state_dict['final_dense.weight'] = torch.from_numpy(model_tf.get_layer('dense_38').get_weights()[0].T)
# state_dict['final_dense.bias'] = torch.from_numpy(model_tf.get_layer('dense_38').get_weights()[1])

# #First Layer

# state_dict['dense1.weight'] = torch.from_numpy(model_tf.get_layer('dense_36').get_weights()[0].T)
# state_dict['dense1.bias'] = torch.from_numpy(model_tf.get_layer('dense_36').get_weights()[1])


# # Patch Embedding layer
# state_dict['patch_embed.class_embed'] = torch.from_numpy(model_tf.get_layer('patch_class_embedding').get_weights()[0])
# state_dict['patch_embed.position_embedding.weight'] = torch.from_numpy(model_tf.get_layer('patch_class_embedding').get_weights()[1])

# # Load the updated state_dict into the PyTorch model

# model_new.load_state_dict(state_dict)
# torch.save(model_new.state_dict(), "models/AcT_model_state_dict.pth")
# # Now the PyTorch model has the same weights as the TensorFlow model
