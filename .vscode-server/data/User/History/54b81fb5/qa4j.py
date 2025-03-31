import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency
from captum.attr import LayerGradCam
from utils.pseudo_labels import smooth_gradients, ig_grad
import math
import os

def visualization(model, test_loader, device, model_name, patient, method, trials, n_batch, dataset_name, input_size):

    if method == 'ig':
        # Create directory for saving saliency maps if it doesn't exist
        os.makedirs("saliency_maps/integrated_gradients", exist_ok=True)
        gradient_integrated(test_loader, model_name, model, patient, device, trials, n_batch, dataset_name, input_size)
    
    elif method == 'vg':
        os.makedirs("saliency_maps/vanilla_gradients", exist_ok=True)
        vanilla_gradients(test_loader, model_name, model, patient, device, trials, n_batch, dataset_name, input_size)
    
    # elif method == 'grad_cam':
    #     pseudo_labels.append(grad_cam(test_loader, criterion, model_name, model, patient, device, treshold_labels))
    return


def gradient_integrated(test_loader, model_name, model, patient, device, trials, n_batch, dataset_name, input_size):
    
    
    if dataset_name == 'SERE':
        label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
    elif dataset_name == 'Toronto':
        label_names = ['General compensation']
    elif dataset_name == 'MMAct':
        label_names = ['standing', 'crouching', 'walking',
                       'running', 'checking_time', 'waving_hand', 
                       'using_phone', 'talking_on_phone', 'kicking', 
                       'pointing', 'throwing', 'jumping', 
                       'exiting', 'entering', 'setting_down', 
                       'talking', 'opening', 'closing',
                       'carrying', 'loitering', 'transferring_object',
                       'looking_around', 'pushing', 'pulling',
                       'picking_up', 'fall', 'sitting_down', 
                       'using_pc', 'drinking', 'pocket_out', 
                       'pocket_in', 'sitting', 'using_phone_desk', 
                       'talking_on_phone_desk', 'standing_up', 
                       'carrying_light', 'carrying_heavy', 'Carrying_light']
        
    atual_batch = [math.floor(trial / n_batch) for trial in trials]
    atual_trial = [torch.abs(torch.tensor(trial) % n_batch) if trial < n_batch * len(test_loader) else torch.tensor(trial) for trial in trials]    
    atual_trial = sorted(atual_trial, key=lambda x: x.item())
    


    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        if batch_idx in atual_batch:
            trial_idx = atual_batch.index(batch_idx)
            if atual_trial[trial_idx] in range(len(inputs)+1):
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
                axes, axes2 = axes.flatten(), axes2.flatten()
                inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
                lengths = lengths.to('cpu')
                inputs.requires_grad = True

                for i in range(len(label_names)):
                    if model_name == 'moment' or  model_name == 'SkateFormer':
                        ig_gradients = ig_grad(inputs, model, lengths, i, baseline=None, steps=5)
                    else:
                        ig = IntegratedGradients(model)
                        ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=i, additional_forward_args=(lengths,), n_steps=5)

                    slc = torch.relu(ig_gradients)
                    if model_name != 'LSTM':
                        slc = smooth_gradients(slc, kernel_size=8)
                    saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], input_size // 3), mode='nearest').squeeze(0)

                    map = torch.zeros_like(saliency)
                    for j in range(saliency.shape[0]):
                        map[j] = (saliency[j] - saliency[j].min()) / (saliency[j].max() - saliency[j].min())

                    active_labels = [label_names[j] for j in range(1, len(label_names)) if targets[0, j] > 0]
                    active_labels_str = ", ".join(active_labels) if active_labels else "No compensation"

                    sns.heatmap(map[0][0:lengths[0]].T.detach().cpu().numpy(), cmap='RdYlBu', ax=axes[i], cbar_kws={'label': 'Normalized Saliency'})
                    axes[i].set_xlabel("Frames", fontsize=12)
                    axes[i].set_ylabel("Joints", fontsize=12)
                    axes[i].set_title(f"Label: {label_names[i]}", fontsize=14)
                    axes[i].tick_params(axis='both', which='major', labelsize=10)

                    head_joints = map[0][0:lengths[0], :].sum(dim=1).detach().cpu().numpy()
                    axes2[i].plot(head_joints)
                    axes2[i].set_title(f"Label: {label_names[i]}", fontsize=14)
                    axes2[i].set_xlabel("Frames", fontsize=12)
                    axes2[i].set_ylabel("Sum of Saliency", fontsize=12)

                fig.suptitle(f"{model_name}: Saliency Maps of Patient {patient + 1}: Trial {trials[trial_idx]} \n Active Labels: {active_labels_str}", fontsize=16, weight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(f"saliency_maps/integrated_gradients/{model_name}_patient_{patient}_trial_{trials[trial_idx]}_IG.png", dpi=300)
                plt.close(fig)

                fig2.suptitle(f"{model_name}: Sum of Saliency for Patient {patient + 1}: Trial {trials[trial_idx]}", fontsize=16, weight='bold')
                fig2.tight_layout(rect=[0, 0, 1, 0.96])
                fig2.savefig(f"saliency_maps/integrated_gradients/{model_name}_patient_{patient}_trial_{trials[trial_idx]}_sum_IG.png", dpi=300)
                plt.close(fig2)

    return


def vanilla_gradients(test_loader, model_name, model, patient, device, trials, n_batch, dataset_name, input_size):

    if dataset_name == 'SERE':
        label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
    elif dataset_name == 'Toronto':
        label_names = ['General compensation']
    elif dataset_name == 'MMAct':
        label_names = ['standing', 'crouching', 'walking',
                       'running', 'checking_time', 'waving_hand', 
                       'using_phone', 'talking_on_phone', 'kicking', 
                       'pointing', 'throwing', 'jumping', 
                       'exiting', 'entering', 'setting_down', 
                       'talking', 'opening', 'closing',
                       'carrying', 'loitering', 'transferring_object',
                       'looking_around', 'pushing', 'pulling',
                       'picking_up', 'fall', 'sitting_down', 
                       'using_pc', 'drinking', 'pocket_out', 
                       'pocket_in', 'sitting', 'using_phone_desk', 
                       'talking_on_phone_desk', 'standing_up', 
                       'carrying_light', 'carrying_heavy', 'Carrying_light']
        
    atual_batch = [math.floor(trial / n_batch) for trial in trials]
    atual_trial = [torch.abs(torch.tensor(trial) - n_batch) for trial in trials]


    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        if batch_idx in atual_batch:
            trial_idx = atual_batch.index(batch_idx)
            if atual_trial[trial_idx] in range(len(inputs)):
                inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
                lengths = lengths.to('cpu')
                inputs.requires_grad = True
                
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
                axes, axes2 = axes.flatten(), axes2.flatten()

                for i in range(len(label_names)):
                    if model_name == 'moment':
                        vg = Saliency(model)
                        vg_gradients = vg.attribute(inputs, target=i)  
                    else:
                        vg = Saliency(model)
                        vg_gradients = vg.attribute(inputs, target=i, additional_forward_args=(lengths,))

                    slc = torch.relu(vg_gradients)
                    if model_name != 'LSTM':
                        slc = smooth_gradients(slc, kernel_size=8)
                    saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], input_size //3), mode='nearest').squeeze(0)

                    map = torch.zeros_like(saliency)
                    for j in range(saliency.shape[0]):
                        map[j] = (saliency[j] - saliency[j].min()) / (saliency[j].max() - saliency[j].min())

                    active_labels = [label_names[j] for j in range(1, len(label_names)) if targets[0, j] > 0]
                    active_labels_str = ", ".join(active_labels) if active_labels else "No compensation"

                    sns.heatmap(map[0][0:lengths[0]].T.detach().cpu().numpy(), cmap='RdYlBu', ax=axes[i], cbar_kws={'label': 'Normalized Saliency'})
                    axes[i].set_xlabel("Frames", fontsize=12)
                    axes[i].set_ylabel("Joints", fontsize=12)
                    axes[i].set_title(f"Label: {label_names[i]}", fontsize=14)
                    axes[i].tick_params(axis='both', which='major', labelsize=10)

                    head_joints = map[0][0:lengths[0], :].sum(dim=1).detach().cpu().numpy()
                    axes2[i].plot(head_joints)
                    axes2[i].set_title(f"Label: {label_names[i]}", fontsize=14)
                    axes2[i].set_xlabel("Frames", fontsize=12)
                    axes2[i].set_ylabel("Sum of Saliency", fontsize=12)

                fig.suptitle(f"{model_name}: Saliency Maps of Patient {patient + 1}: Trial {trials[trial_idx]} \n Active Labels: {active_labels_str}", fontsize=16, weight='bold')
                fig.tight_layout(rect=[0, 0, 1, 0.96])
                fig.savefig(f"saliency_maps/vanilla_gradients/{model_name}_patient_{patient}_trial_{trials[trial_idx]}_IG.png", dpi=300)
                plt.close(fig)

                fig2.suptitle(f"{model_name}: Sum of Saliency for Patient {patient + 1}: Trial {trials[trial_idx]}", fontsize=16, weight='bold')
                fig2.tight_layout(rect=[0, 0, 1, 0.96])
                fig2.savefig(f"saliency_maps/vanilla_gradients/{model_name}_patient_{patient}_trial_{trials[trial_idx]}_sum_IG.png", dpi=300)
                plt.close(fig2)
                
    return


# def rollout(model, model_name):
    
#     # Compute the rollout of attention weights
#     if model_name == 'moment':
#         layers = model.moment_model.attention_maps[0]
#         num_layers = len(layers)        
#         rollout = torch.eye(layers[0].size(-1)).to(layers[0].device)
#         for i in range(num_layers):
#             attention = layers[i]
#             attention = attention + torch.eye(attention.size(-1)).to(attention.device)  # Add identity matrix to attention
#             attention = attention / attention.sum(dim=-1, keepdim=True)  # Normalize attention weights
#             rollout = torch.matmul(rollout, attention)
#     else:
#         layers = model.transformer.layers
#         num_layers = len(layers)    
#         rollout = torch.eye(layers[0].mha.attention_weights.size(-1)).to(layers[0].mha.attention_weights.device)
#         for i in range(num_layers):
#             attention = layers[i].mha.attention_weights
#             attention = attention + torch.eye(attention.size(-1)).to(attention.device)  # Add identity matrix to attention
#             attention = attention / attention.sum(dim=-1, keepdim=True)  # Normalize attention weights
#             rollout = torch.matmul(rollout, attention)
    
#     return rollout


# def grad_cam(input, targets, outputs, output_dir, lenghts, criterion, model_name, model, trial, patient):
    
#     # label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation','Exaggerated Shoulder Abd.', 'Trunk C.','Head C.' ]
#     label_names = ['General C']
#     rollout_attention = rollout(model, model_name)
    
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()

#     for label in range(len(label_names)):
        
#         model.zero_grad()
#         loss = criterion(outputs[:, label], targets[:, label])
        
#         loss.backward()
        
#         grads_input = torch.relu(input.grad)
#         # grads_input = grads_input[:,0:lenghts[0],:]
#         attention_weights_mean = rollout_attention.mean(dim=1)
        
#         if model_name == 'moment':
#             cls_token = attention_weights_mean[:,0,1:]
#             attention_weights_mean = torch.nn.functional.interpolate(attention_weights_mean.unsqueeze(0), size=(512), mode='nearest').squeeze(0) 
#         normalized_attention = attention_weights_mean / attention_weights_mean.sum(dim=-1, keepdim=True)
#         slc = torch.einsum('bsi,bsi->bsi', attention_weights_mean[:,0,1:].unsqueeze(-1), grads_input)
#         saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='bilinear', align_corners=False).squeeze(0) 
#         saliency = smooth_gradients(saliency, kernel_size=10)
        
#         map = torch.zeros_like(saliency)
        
#         for i in range(saliency.shape[0]):
#             map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
            
#         if patient == 0 and trial in [1, 15, 25, 35, 45, 55, 65]:

#             active_labels = [label_names[i] for i in range(len(label_names)) if targets[:, i] > 0]
#             active_labels_str = ", ".join(active_labels)
            
#             sns.heatmap(map[0][0:lenghts[0]].T.detach().cpu().numpy(), cmap='RdYlBu', ax=axes[label])
#             axes[label].set_xlabel("Frames")
#             axes[label].set_ylabel("Joints")
#             axes[label].set_title(f"Label: {label_names[label]}", fontsize=5)
            
#             if label == 0:
                
#                 fig2, axes2 = plt.subplots(3, 1, figsize=(10, 15))
                
#                 head_joints = map[0][0:lenghts[0], 7:9].sum(dim=1).detach().cpu().numpy()
#                 axes2[0].plot(head_joints)
#                 axes2[0].set_title("Head Joints (0-10)")
#                 axes2[0].set_xlabel("Frames")
#                 axes2[0].set_ylabel("Sum of Saliency")
                
#                 shoulder_joints = map[0][0:lenghts[0], 11:13].sum(dim=1).detach().cpu().numpy()
#                 axes2[1].plot(shoulder_joints)
#                 axes2[1].set_title("Shoulder Joints (11-12)")
#                 axes2[1].set_xlabel("Frames")
#                 axes2[1].set_ylabel("Sum of Saliency")
                
#                 specific_joints = map[0][0:lenghts[0], [11, 12, 23, 24]].sum(dim=1).detach().cpu().numpy()
#                 axes2[2].plot(specific_joints)
#                 axes2[2].set_title("Trunk Joints (11, 12, 23, 24)")
#                 axes2[2].set_xlabel("Frames")
#                 axes2[2].set_ylabel("Sum of Saliency")
                
#                 plt.tight_layout()
#                 plt.savefig(f"saliency_maps/grad_cam/{model_name}_patient_{patient}_trial_{trial}_label_{label}_plots.png", dpi=300)
#                 plt.close(fig2)
                
#     if patient == 0 and trial in [1, 15, 25, 35, 45, 55, 65]:

#         plt.suptitle(f"{model_name} model: Saliency Maps of Patient {patient + 1}:  Trial {trial} \n Active Labels: {active_labels_str}", fontsize=16, weight='bold')
#         plt.tight_layout(rect=[0, 0, 1, 0.96])
#         plt.savefig(f"saliency_maps/grad_cam/{model_name}_patient_{patient}_trial_{trial}_grad_cam.png", dpi=300)
#         plt.close()
        
#     return


# def attention_weights(input, targets, outputs, output_dir, lenghts, criterion, model_name, model, trial, patient):
    
#     label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation','Exaggerated Shoulder Abd.', 'Trunk C.','Head C.' ]

#     rollout_attention = rollout(model.transformer.layers)
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()

#     for label in range(len(label_names)):
        
#         # Zero the gradients
#         model.zero_grad()
        
#         # Calculate loss for specific label
#         loss = criterion(outputs[:, label], targets[:, label])
        
#         # Backward pass to compute gradients
#         loss.backward(retain_graph=True)
        
#         # Compute average attention weights
#         attention_weights_mean = rollout_attention.mean(dim=1)
#         attention_weights = attention_weights_mean[:, 0, 1:]  # Shape: [batch_size, seq_len]

#         # Normalize attention weights
#         attention_weights_norm = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
#         grads_input = input.grad 

#         # Compute weighted gradients for saliency map
#         weighted_grads = grads_input * attention_weights_norm.unsqueeze(-1)  # Broadcasting attention weights
#         saliency_map = weighted_grads.sum(dim=-1)  # Reduce across feature dimension
        
#         # Normalize saliency map
#         map = torch.zeros_like(saliency_map)
#         for i in range(saliency_map.shape[0]):
#             map[i] = (saliency_map[i] - saliency_map[i].min()) / (saliency_map[i].max() - saliency_map[i].min())
        
#         # Get active labels for title
#         active_labels = [label_names[i] for i in range(len(label_names)) if targets[:, i] > 0]
#         active_labels_str = ", ".join(active_labels)
        
#         # Plot the saliency map as a line plot for each label
#         axes[label].plot(map[0, :lenghts[0]].detach().cpu().numpy(), label="Saliency Map")
#         axes[label].set_xlabel("Frames")
#         axes[label].set_ylabel("Saliency")
#         axes[label].set_title(f"Label: {label_names[label]}", fontsize=10)
#         axes[label].legend()

#     # Save plot with detailed title
#     plt.suptitle(f"{model_name} Model: Saliency Maps of Patient {patient + 1}, Trial {trial} \nActive Labels: {active_labels_str}", fontsize=12, weight='bold')
#     plt.tight_layout(rect=[0, 0, 1, 0.96])
#     plt.savefig(f"saliency_maps/grad_cam/{model_name}_patient_{patient}_trial_{trial}_GC.png", dpi=300)
#     plt.close()    
    
#     return



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
