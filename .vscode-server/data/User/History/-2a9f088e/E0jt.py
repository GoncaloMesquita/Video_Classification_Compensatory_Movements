import torch
import numpy as np  
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.pseudo_labels import gradient_normal, grad_cam, grad_cam_II, attention_weights, gradient_integrated, gradient_normal_II
from sklearn.metrics import roc_auc_score


def training(model, dataloader, optimizer, criterion, device, save_dir, model_name, thresholds, clip_value, optuna, epoch, scaler):
    
    model.train()
    running_loss = 0.0
    all_targets, all_sigmoid_outputs, all_predictions = [], [], []
    thresholds = torch.tensor(thresholds, device=device)
    
    if epoch == 0 and model_name in ['moment', 'AcT', 'SkateFormer', 'moment+dino']:
        
        if model_name == 'moment+dino':
            for name, param in model.named_parameters():
                if any(substring in name for substring in ["dinov2.encoder.layer.3.attention.attention"]):
                    param.requires_grad = True
                    print('Unfreez Layers:' ,name)
        else:
            for param in model.parameters():
                param.requires_grad = True
                
        print("All layers unfrozen!")
    
    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(dataloader):
        
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device,non_blocking=True ).float()
        lengths = lengths.to('cpu')
        
        optimizer.zero_grad()

        # with torch.amp.autocast(device_type='cuda'):

        if model_name == 'moment':  
            outputs = model(inputs)
            
        elif model_name == 'moment+dino':
            inputs2 = inputs2.to(device, non_blocking=True).float()
            outputs = model(inputs, inputs2)
            
        else:
            outputs = model(inputs, lengths)
        
        loss = criterion(outputs, targets)
        # scaler.scale(loss).backward()
        loss.backward()
        
        if clip_value is not None:
            utils.clip_grad_value_(model.parameters(), clip_value)
                            
        # scaler.step(optimizer)
        # scaler.update()   
        optimizer.step()
        running_loss += loss.item()
        
        sigmoid_outputs = torch.sigmoid(outputs) 
        predicted = sigmoid_outputs > thresholds
        all_sigmoid_outputs.append(sigmoid_outputs.detach().cpu().numpy())
        all_targets.append(targets.cpu().detach().numpy())
        all_predictions.append(predicted.cpu().detach().numpy())
    
    if optuna:
        return running_loss / len(dataloader)
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0) 
    
    # metrics(all_targets, all_predictions, "train", save_dir, model_name)
    
    return running_loss / len(dataloader), all_targets, all_sigmoid_outputs


def validate(model, val_loader, criterion, device, model_name, thresholds, optuna):
    
    
    val_loss = 0.0
    all_targets, all_predictions, all_sigmoid_outputs = [], [], []
    thresholds = torch.tensor(thresholds)
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(val_loader):
            inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
            lengths = lengths.to('cpu')
            
            if model_name == 'moment':
                outputs = model(inputs)
            elif model_name == 'moment+dino':
                inputs2 = inputs2.to(device, non_blocking=True).float()
                outputs = model(inputs, inputs2)
            else: 
                outputs = model(inputs, lengths)
                
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            sigmoid_outputs = torch.sigmoid(outputs) 
            predicted = sigmoid_outputs > thresholds.to(inputs.device)
            
            all_sigmoid_outputs.append(sigmoid_outputs.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

        
    val_loss /= len(val_loader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0)
    
    if optuna:
        
        return val_loss, f1_score(all_targets, all_predictions, average='samples', zero_division=0)
    
    # _ = metrics(all_targets, all_predictions, mode, save_dir, model_name)
    
    return val_loss, all_targets, all_predictions, all_sigmoid_outputs




# import os
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt

# # Create a folder to save all the plots for the entire video
# output_dir = 'attention_maps'
# os.makedirs(output_dir, exist_ok=True)

# num_patches = 14  # Adjust based on your model's configuration

# # Assuming 'images' is your batch of input images
# # Iterate through the batch
# for idx, input_image in enumerate(images[0,0:lengths[0]]):
#     # Resize the input image from [256, 256, 3] to [224, 224, 3]
#     input_image = input_image.permute(2, 0, 1)

#     input_image = F.interpolate(
#         input_image.unsqueeze(0),  # Add batch dimension
#         size=(224, 224),          # Desired size
#         mode='bilinear',
#         align_corners=False
#     ).squeeze(0)  # Remove batch dimension

#     # Get the attention map from the model
#     attention_map = model.attention_weights[idx][-1]  # Last layer's attention map

#     # Average over heads
#     atten_mean = attention_map[:, 4]
#     # Get attention from class token to all patches
#     cls_token_attn = atten_mean[0, 0, 1:]  # Exclude the class token

#     # Normalize the attention map
#     cls_token_attn = cls_token_attn / cls_token_attn.max()

#     # Reshape to 2D attention map
    # num_patches = int(cls_token_attn.size(0) ** 0.5)
    # attention_map_2d = cls_token_attn.view(num_patches, num_patches)

#     # Resize the attention map to match the input image size (224 x 224)
#     attn_resized = F.interpolate(
#         attention_map_2d.unsqueeze(0).unsqueeze(0),
#         size=(224, 224),
#         mode='bilinear',
#         align_corners=False
#     ).squeeze(0).squeeze(0)

#     # Prepare the image for plotting
#     img = input_image.cpu().numpy()
#     img = img.transpose(1, 2, 0)  # Convert from [C, H, W] to [H, W, C]

#     # Plot and save the attention map over the image
#     plt.imshow(img)
#     plt.imshow(attn_resized.cpu().detach().numpy(), alpha=0.5, cmap='jet')
#     plt.axis('off')
#     plt.savefig(os.path.join(output_dir, f'attention_image_{idx}.png'))
#     plt.close()











# import torch
# import torch.nn.functional as F
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Example attention map extracted from the model
# attention_map = model.moment_model.attention_maps[0]
# maps = 0
# head = 6
# attention = []
# for atten in attention_map:
#     attention.append(atten.detach().cpu().numpy())

# attention_map = np.array(attention)
# attention_map_mean_head = attention_map.sum(axis=0)

# # Average across heads and take the CLS token
# #atten = attention_map_mean_head.mean(axis=1)  # Shape: [99, 64]
# atten = attention_map_mean_head.sum(axis=1)
# cls_token = atten.sum(axis=1)  # Shape: [99]

# # Convert cls_token back to tensor for interpolation
# cls_token_tensor = torch.tensor(cls_token).unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, 99]

# # Interpolation to new size [33, lengths]
# interpolated_map = F.interpolate(
#     cls_token_tensor,
#     size=(33, 512),  # Target size
#     mode="bilinear",
#     align_corners=False
# ).squeeze(0).squeeze(0)  # Shape: [33, lengths]

# # Normalize the interpolated saliency map for visualization
# normalized_map = (interpolated_map - interpolated_map.min()) / (interpolated_map.max() - interpolated_map.min())

# # Convert to NumPy for plotting
# normalized_map_np = normalized_map.cpu().numpy()

# # Visualization
# plt.figure(figsize=(10, 6))
# sns.heatmap(normalized_map_np[:,:lengths[0]], cmap="viridis", cbar=True)
# plt.title("Interpolated Saliency Map: Relation Between Joints and Sequence")
# plt.xlabel("Sequence Lengths")
# plt.ylabel("Number of Joints")
# plt.savefig(f'interpolated_attention_map_sum_sum')
# plt.show()
