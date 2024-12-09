from models.LSTM import LSTM
import torch
from models.AcT import AcT
from models.moment import Moment    
from models.SkateFormer import SkateFormer
from models.momentdino import MomentDino
from models.MLP import MLP
import torch.nn as nn

def create_model(model_name, input_size, hidden_size, num_layers, num_labels, dropout, checkpoint, mode, pretrained, device):
    # Import the necessary modules
    
    if model_name == 'LSTM':
        if mode == 'train':
            model = LSTM(input_size, hidden_size[0], num_layers, num_labels, dropout)
            
        if mode == 'test':
            model = LSTM(input_size, hidden_size[0], num_layers, num_labels, dropout)
            print("checkpoint", checkpoint)
            model.load_state_dict(torch.load(checkpoint,weights_only=True, map_location=device), strict=True)
        
        
    elif model_name == 'AcT':
        if mode == 'train':
            model = AcT(dropout)
            checkpoints = torch.load("models/AcT_model_state_dict.pth",weights_only=True, map_location=device)
            
            for k in ['patch_embed.position_embedding.weight']:
                if k in checkpoints:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoints[k]
            
            model.load_state_dict(checkpoints, strict=False)
            model.dense1 = torch.nn.Linear(input_size, 256)
            model.final_dense = torch.nn.Linear(512, num_labels)
            
            if pretrained:
                
                unfreeze_layers = ['dense1.weight', 'dense1.bias', 'final_dense.bias', 'final_dense.weight', 'patch_embed.position_embedding.weight']

                for name, param in model.named_parameters():
                    if name not in unfreeze_layers:
                        param.requires_grad = False
                        print('Freez layer:', name)
        
        if mode == 'test':  
            model = AcT(dropout)
            model.dense1 = torch.nn.Linear(input_size,256)
            model.final_dense = torch.nn.Linear(512, num_labels)
            checkpoints = torch.load(checkpoint, weights_only=True, map_location=device)
            model.load_state_dict(checkpoints, strict=True)
            
            
    elif model_name == 'SkateFormer' :
        if mode == "train":

            model = SkateFormer( in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=6,
                 embed_dim=96, num_people=1, num_frames=769, num_points=33, kernel_size=7, num_heads=32,
                 type_1_size=(1,11), type_2_size=(1,11), type_3_size=(1,11), type_4_size=(1,11),
                 attn_drop=0., head_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=False, global_pool='avg')
            
            checkpoints = torch.load("models/SkateFormer_NTU120_j.pt",weights_only=True, map_location=device)
            
            for i in range(0,4):    
                for j in range(0,2):
                    for m in range(0,4):
                        for k in [
                            'head.weight', 'head.bias', 'joint_person_embedding',
                            f'stages.{i}.blocks.{j}.transformer.gconv',
                            f'stages.{i}.blocks.{j}.transformer.attention.{m}.relative_position_index' ,
                            f'stages.{i}.blocks.{j}.transformer.attention.{m}.relative_position_bias_table'
                        ]:
                            if k in checkpoints:
                                print(f"Removing key {k} from pretrained checkpoint")
                                del checkpoints[k]
                        
            missing_keys, unexpected_keys = model.load_state_dict(checkpoints, strict=False)
            
            if pretrained:
                unfreeze_layers = ['head.weight', 'head.bias', 'joint_person_embedding']
                for i in range(0, 4):
                    for j in range(0, 2):
                        for m in range(0, 4):
                            unfreeze_layers.extend([
                                f'stages.{i}.blocks.{j}.transformer.gconv.weight',
                                f'stages.{i}.blocks.{j}.transformer.gconv.bias',
                                f'stages.{i}.blocks.{j}.transformer.attention.{m}.relative_position_index',
                                f'stages.{i}.blocks.{j}.transformer.attention.{m}.relative_position_bias_table'
                            ])

                for name, param in model.named_parameters():
                    if name not in unfreeze_layers:
                        param.requires_grad = False
                        print('Freez layer:', name)

            if not missing_keys and not unexpected_keys:
                print("All weights loaded correctly.")
            else:
                print(f"Missing keys: {missing_keys}")
                print(f"Unexpected keys: {unexpected_keys}")
            
        if mode == 'test':  
            
            model = SkateFormer( in_channels=3, depths=(2, 2, 2, 2), channels=(96, 192, 192, 192), num_classes=6,
                 embed_dim=96, num_people=1, num_frames=769, num_points=33, kernel_size=7, num_heads=32,
                 type_1_size=(1,11), type_2_size=(1,11), type_3_size=(1,11), type_4_size=(1,11),
                 attn_drop=0., head_drop=0., drop=0., rel=True, drop_path=0., mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer_transformer=nn.LayerNorm, index_t=False, global_pool='avg')
            checkpoints = torch.load(checkpoint, weights_only=True, map_location=device)
            model.load_state_dict(checkpoints, strict=True)
                      
                      
    elif model_name == 'moment':
        
        if mode == 'train':
            
            model = Moment(num_labels, dropout)
            
            for name, param in model.named_parameters():
                if not name.startswith('moment_model.head.'):
                    param.requires_grad = False
                    print('Freezing layer:', name)
            
        if mode == 'test':
            
            model = Moment(num_labels, dropout)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device), strict=True)
            
            for name, param in model.named_parameters():
                if not name.startswith('moment_model.head.'):
                    param.requires_grad = False
                    print('Freezing layer:', name)
                
                
    elif model_name == 'moment+dino':
        
        if mode == 'train':
            
            model = MomentDino(num_labels, dropout, input_size, seq_length=512)

            for name, param in list(model.dinov2.named_parameters()) + list(model.moment_model.head.linear.named_parameters()):
                param.requires_grad = False
                print('Freezing layer:', name)
                                
        if mode == 'test':
            
            model = MomentDino(num_labels, dropout, input_size, seq_length=512)
            model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device), strict=True)
            
            
    if model_name == 'MLP':
        if mode == 'train':
            model = MLP(input_size, hidden_size[0], hidden_size[1], num_labels, dropout)
            
        if mode == 'test':
            model = MLP(input_size, hidden_size[0],hidden_size[1], num_labels, dropout)
            model.load_state_dict(torch.load(checkpoint,weights_only=True, map_location=device), strict=True)
        
    return model

