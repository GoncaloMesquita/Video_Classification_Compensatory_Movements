import torch
import numpy as np  
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from visualization import gradient_normal, grad_cam, grad_cam_II, attention_weights, gradient_integrated, gradient_normal_II
from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam
from captum.attr import Saliency, DeepLift, ShapleyValueSampling,GradientShap, LayerIntegratedGradients
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


def validate(model, val_loader, criterion, device, mode, save_dir, model_name, thresholds, optuna, saliency_map, scaler, patient, treshold_labels, method):
    
    model.eval()
    if saliency_map and mode == 'testing' and model_name == 'LSTM':
        model.train()
    pseudo_labels = []
    val_loss = 0.0
    all_targets, all_predictions, all_sigmoid_outputs = [], [], []
    thresholds = torch.tensor(thresholds)
    
    if not saliency_map:
        context_manager = torch.no_grad()
    else:
        context_manager = torch.enable_grad()

    with context_manager:
    
        for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(val_loader):
            # if patient in [3, 11, 17] and batch_idx in [0, 15, 65]:
            inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
            
            if method == 'ig':
                if baseline is None:
                    baseline = torch.zeros_like(inputs).to(inputs.device)
    
                integrated_grads = torch.zeros_like(inputs).to(inputs.device)
            # lengths = lengths.to('cpu')
            
            # if saliency_map:
            #     inputs.requires_grad = True
            
            # if model_name == 'moment':
            #     outputs = model(inputs)
            # elif model_name == 'moment+dino':
            #     inputs2 = inputs2.to(device, non_blocking=True).float()
            #     outputs = model(inputs, inputs2)
            # else: 
            #     outputs = model(inputs, lengths)
                
            # loss = criterion(outputs, targets)
            # val_loss += loss.item()
            
            # sigmoid_outputs = torch.sigmoid(outputs) 
            # predicted = sigmoid_outputs > thresholds.to(inputs.device)
            
            # all_sigmoid_outputs.append(sigmoid_outputs.detach().cpu().numpy())
            # all_targets.append(targets.cpu().numpy())
            # all_predictions.append(predicted.cpu().numpy())

            if saliency_map and mode == 'testing' :
                if method == 'ig':
                    num_labels = targets.shape[1]
                    attributions = []
                    ig = IntegratedGradients(model)

                    # for label_idx in range(0):
                        
                        # attributions.append(ig.attribute(inputs, baselines=inputs * 0, target=label_idx,additional_forward_args=lengths  ))
                    attributions.append(integrated_gradients(inputs, model, 0, baseline=None, steps=50))
                    pseudo_labels.append(gradient_integrated( targets, lengths, criterion, model_name, model, batch_idx, patient, treshold_labels, attributions))
                    
                # if method == 'vg':
                #     pseudo_labels.append(gradient_normal(inputs, targets, outputs, save_dir, lengths, criterion, model_name, model, batch_idx, patient, treshold_labels))
                    
                # if method == 'grad_cam':
                #     grad_cam(inputs, targets, outputs, save_dir, lengths, criterion, model_name, model, batch_idx, patient)
                
                # if method == 'grad_cam_II':   
                #     grad_cam_II(inputs, targets, outputs, save_dir, lengths, criterion, model_name, model, batch_idx, patient)       

                inputs.grad = None
                # del  loss, inputs, targets, inputs2, lengths
                
                torch.cuda.empty_cache()
                
        
        if saliency_map and mode == 'testing':
            # all_targets = np.concatenate(all_targets, axis=0)
            # all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0)
            # num_labels = all_targets.shape[1]
            # aucs = []
            # for label_idx in range(num_labels):
            #     y_true = all_targets[:, label_idx]
            #     y_score = all_sigmoid_outputs[:, label_idx]
            #     if np.unique(y_true).size > 1:
            #         auc = roc_auc_score(y_true, y_score)
            #         print(f"AUC for label {label_idx}: {auc}")
            #     else:
            #         auc = None  # or set to 0.5, np.nan, etc., based on preference
            #         print(f"AUC for label {label_idx}: Undefined (only one class present in y_true)")
            return pseudo_labels
        
    val_loss /= len(val_loader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0)
    
    if optuna:
        
        return val_loss, f1_score(all_targets, all_predictions, average='samples', zero_division=0)
    
    # _ = metrics(all_targets, all_predictions, mode, save_dir, model_name)
    
    return val_loss, all_targets, all_predictions, all_sigmoid_outputs


def integrated_gradients(inputs, model, target_label_idx, baseline=None, steps=50):
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(inputs.device)
    
    integrated_grads = torch.zeros_like(inputs).to(inputs.device)
    

    for i in range(steps + 1):
        alpha = float(i) / steps
        scaled_input = baseline + alpha * (inputs - baseline)
        scaled_input = scaled_input.clone().detach().requires_grad_(True)

        outputs = model(scaled_input)

        loss = outputs[0, target_label_idx]

        loss.backward()

        # Accumulate gradients scaled by the step size
        integrated_grads += scaled_input.grad.detach() * (inputs - baseline) / steps

        # Clear gradients and intermediate variables
        model.zero_grad()
        scaled_input.grad = None
        del scaled_input, outputs, loss
        torch.cuda.empty_cache()

    del inputs
    # Detach integrated gradients and return
    return integrated_grads.detach()


# def integrated_gradients(inputs, model, target_label_idx, baseline=None, steps=50):
#     if baseline is None:
#         baseline = torch.zeros_like(inputs).to(inputs.device)
    
#     # Scale inputs and compute gradients
#     scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
#     grads = []

#     for scaled_input in scaled_inputs:
#         scaled_input = scaled_input.clone().detach().requires_grad_(True).float()
#         outputs = model(scaled_input)
#         loss = outputs[0, target_label_idx]
#         loss.backward()
#         grads.append(scaled_input.grad.detach().clone())
#         model.zero_grad()
#         del outputs, loss, scaled_input
#         torch.cuda.empty_cache()
    
#     # Average gradients and compute integrated gradients
#     avg_grads = torch.mean(torch.stack(grads), dim=0)
#     integrated_grads = (inputs - baseline) * avg_grads
    
#     if inputs.grad is not None:
#         inputs.grad.detach_()
#         inputs.grad.zero_()
#     model.zero_grad()
#     torch.cuda.empty_cache()
    
#     return integrated_grads


def smooth_gradients(inputs, model, target_label_idx, num_samples=50, noise_level=0.1):
        # Ensure inputs require gradients
    inputs = inputs.clone().detach().requires_grad_(True).float()

    # Store gradients for each noisy sample
    grads = []

    for _ in range(num_samples):
        # Add Gaussian noise to the input
        noisy_input = inputs + noise_level * torch.randn_like(inputs)
        noisy_input = noisy_input.clone().detach().requires_grad_(True).float()
        
        # Forward pass
        outputs = model(noisy_input)
        loss = outputs[0, target_label_idx]
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Check if gradients are not None
        if noisy_input.grad is not None:
            grads.append(torch.relu(noisy_input.grad.clone().detach()))  # Save the gradient
        
        # Reset gradients in the model
        model.zero_grad()

    # Check if grads list is empty
    if not grads:
        raise RuntimeError("No gradients were computed. Check the model and inputs.")

    # Average gradients across all samples
    avg_grads = torch.mean(torch.stack(grads), dim=0)

    # Return the averaged gradients as the SmoothGrad saliency map
    return avg_grads


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
