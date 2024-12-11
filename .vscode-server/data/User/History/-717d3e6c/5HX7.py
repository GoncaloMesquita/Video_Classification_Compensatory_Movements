import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency
from captum.attr import LayerGradCam
from captum.attr import Saliency, DeepLift, ShapleyValueSampling,GradientShap, LayerIntegratedGradients
from utils.pseudo_labels import smooth_gradients, ig_grad
import math

def visaulization(model, test_loader, criterion, device, model_name, patient, method, trials, n_batch):

    if method == 'ig':
        gradient_integrated(test_loader, model_name, model, patient, device, trials, n_batch)
    
    elif method == 'vg':
        vanilla_gradients(test_loader, criterion, model_name, model, patient, device, trials, n_batch)
    
    # elif method == 'grad_cam':
    #     pseudo_labels.append(grad_cam(test_loader, criterion, model_name, model, patient, device, treshold_labels))
            
    return
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients
import math

def gradient_integrated(test_loader, model_name, model, patient, device, trials, n_batch):
    
    label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation', 'Exaggerated Shoulder Abd.', 'Trunk C.', 'Head C.']
    atual_batch = [math.floor(trial / n_batch) for trial in trials]
    atual_trial = [torch.abs(trial - n_batch) for trial in trials]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes, axes2 = axes.flatten(), axes2.flatten()

    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        if batch_idx in atual_batch:
            trial_idx = atual_batch.index(batch_idx)
            if atual_trial[trial_idx] in range(len(inputs)):
                inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
                lengths = lengths.to('cpu')
                inputs.requires_grad = True

                for i in range(len(label_names)):
                    if model_name == 'moment':
                        ig_gradients = ig_grad(inputs, model, lengths, i, baseline=None, steps=5)
                    else:
                        ig = IntegratedGradients(model)
                        ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=i, additional_forward_args=(lengths,))

                    slc = torch.relu(ig_gradients)
                    if model_name != 'LSTM':
                        slc = smooth_gradients(slc, kernel_size=8)
                    saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='nearest').squeeze(0)

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


def vanilla_gradients(input, targets, outputs, output_dir, lenghts, criterion, model_name, model, trial, patient, treshold_labels):\

    label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation', 'Exaggerated Shoulder Abd.', 'Trunk C.', 'Head C.']
    atual_batch = [math.floor(trial / n_batch) for trial in trials]
    atual_trial = [torch.abs(trial - n_batch) for trial in trials]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 10))
    axes, axes2 = axes.flatten(), axes2.flatten()

    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        if batch_idx in atual_batch:
            trial_idx = atual_batch.index(batch_idx)
            if atual_trial[trial_idx] in range(len(inputs)):
                inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
                lengths = lengths.to('cpu')
                inputs.requires_grad = True

                for i in range(len(label_names)):
                    if model_name == 'moment':
                        ig_gradients = ig_grad(inputs, model, lengths, i, baseline=None, steps=5)
                    else:
                        ig = IntegratedGradients(model)
                        ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=i, additional_forward_args=(lengths,))

                    slc = torch.relu(ig_gradients)
                    if model_name != 'LSTM':
                        slc = smooth_gradients(slc, kernel_size=8)
                    saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='nearest').squeeze(0)

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


def rollout(model, model_name):
    
    # Compute the rollout of attention weights
    if model_name == 'moment':
        layers = model.moment_model.attention_maps[0]
        num_layers = len(layers)        
        rollout = torch.eye(layers[0].size(-1)).to(layers[0].device)
        for i in range(num_layers):
            attention = layers[i]
            attention = attention + torch.eye(attention.size(-1)).to(attention.device)  # Add identity matrix to attention
            attention = attention / attention.sum(dim=-1, keepdim=True)  # Normalize attention weights
            rollout = torch.matmul(rollout, attention)
    else:
        layers = model.transformer.layers
        num_layers = len(layers)    
        rollout = torch.eye(layers[0].mha.attention_weights.size(-1)).to(layers[0].mha.attention_weights.device)
        for i in range(num_layers):
            attention = layers[i].mha.attention_weights
            attention = attention + torch.eye(attention.size(-1)).to(attention.device)  # Add identity matrix to attention
            attention = attention / attention.sum(dim=-1, keepdim=True)  # Normalize attention weights
            rollout = torch.matmul(rollout, attention)
    
    return rollout


def grad_cam(input, targets, outputs, output_dir, lenghts, criterion, model_name, model, trial, patient):
    
    # label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation','Exaggerated Shoulder Abd.', 'Trunk C.','Head C.' ]
    label_names = ['General C']
    rollout_attention = rollout(model, model_name)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for label in range(len(label_names)):
        
        model.zero_grad()
        loss = criterion(outputs[:, label], targets[:, label])
        
        loss.backward()
        
        grads_input = torch.relu(input.grad)
        # grads_input = grads_input[:,0:lenghts[0],:]
        attention_weights_mean = rollout_attention.mean(dim=1)
        
        if model_name == 'moment':
            cls_token = attention_weights_mean[:,0,1:]
            attention_weights_mean = torch.nn.functional.interpolate(attention_weights_mean.unsqueeze(0), size=(512), mode='nearest').squeeze(0) 
        normalized_attention = attention_weights_mean / attention_weights_mean.sum(dim=-1, keepdim=True)
        slc = torch.einsum('bsi,bsi->bsi', attention_weights_mean[:,0,1:].unsqueeze(-1), grads_input)
        saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='bilinear', align_corners=False).squeeze(0) 
        saliency = smooth_gradients(saliency, kernel_size=10)
        
        map = torch.zeros_like(saliency)
        
        for i in range(saliency.shape[0]):
            map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
            
        if patient == 0 and trial in [1, 15, 25, 35, 45, 55, 65]:

            active_labels = [label_names[i] for i in range(len(label_names)) if targets[:, i] > 0]
            active_labels_str = ", ".join(active_labels)
            
            sns.heatmap(map[0][0:lenghts[0]].T.detach().cpu().numpy(), cmap='RdYlBu', ax=axes[label])
            axes[label].set_xlabel("Frames")
            axes[label].set_ylabel("Joints")
            axes[label].set_title(f"Label: {label_names[label]}", fontsize=5)
            
            if label == 0:
                
                fig2, axes2 = plt.subplots(3, 1, figsize=(10, 15))
                
                head_joints = map[0][0:lenghts[0], 7:9].sum(dim=1).detach().cpu().numpy()
                axes2[0].plot(head_joints)
                axes2[0].set_title("Head Joints (0-10)")
                axes2[0].set_xlabel("Frames")
                axes2[0].set_ylabel("Sum of Saliency")
                
                shoulder_joints = map[0][0:lenghts[0], 11:13].sum(dim=1).detach().cpu().numpy()
                axes2[1].plot(shoulder_joints)
                axes2[1].set_title("Shoulder Joints (11-12)")
                axes2[1].set_xlabel("Frames")
                axes2[1].set_ylabel("Sum of Saliency")
                
                specific_joints = map[0][0:lenghts[0], [11, 12, 23, 24]].sum(dim=1).detach().cpu().numpy()
                axes2[2].plot(specific_joints)
                axes2[2].set_title("Trunk Joints (11, 12, 23, 24)")
                axes2[2].set_xlabel("Frames")
                axes2[2].set_ylabel("Sum of Saliency")
                
                plt.tight_layout()
                plt.savefig(f"saliency_maps/grad_cam/{model_name}_patient_{patient}_trial_{trial}_label_{label}_plots.png", dpi=300)
                plt.close(fig2)
                
    if patient == 0 and trial in [1, 15, 25, 35, 45, 55, 65]:

        plt.suptitle(f"{model_name} model: Saliency Maps of Patient {patient + 1}:  Trial {trial} \n Active Labels: {active_labels_str}", fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"saliency_maps/grad_cam/{model_name}_patient_{patient}_trial_{trial}_grad_cam.png", dpi=300)
        plt.close()
        
    return


def attention_weights(input, targets, outputs, output_dir, lenghts, criterion, model_name, model, trial, patient):
    
    label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation','Exaggerated Shoulder Abd.', 'Trunk C.','Head C.' ]

    rollout_attention = rollout(model.transformer.layers)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for label in range(len(label_names)):
        
        # Zero the gradients
        model.zero_grad()
        
        # Calculate loss for specific label
        loss = criterion(outputs[:, label], targets[:, label])
        
        # Backward pass to compute gradients
        loss.backward(retain_graph=True)
        
        # Compute average attention weights
        attention_weights_mean = rollout_attention.mean(dim=1)
        attention_weights = attention_weights_mean[:, 0, 1:]  # Shape: [batch_size, seq_len]

        # Normalize attention weights
        attention_weights_norm = attention_weights / attention_weights.sum(dim=-1, keepdim=True)
        grads_input = input.grad 

        # Compute weighted gradients for saliency map
        weighted_grads = grads_input * attention_weights_norm.unsqueeze(-1)  # Broadcasting attention weights
        saliency_map = weighted_grads.sum(dim=-1)  # Reduce across feature dimension
        
        # Normalize saliency map
        map = torch.zeros_like(saliency_map)
        for i in range(saliency_map.shape[0]):
            map[i] = (saliency_map[i] - saliency_map[i].min()) / (saliency_map[i].max() - saliency_map[i].min())
        
        # Get active labels for title
        active_labels = [label_names[i] for i in range(len(label_names)) if targets[:, i] > 0]
        active_labels_str = ", ".join(active_labels)
        
        # Plot the saliency map as a line plot for each label
        axes[label].plot(map[0, :lenghts[0]].detach().cpu().numpy(), label="Saliency Map")
        axes[label].set_xlabel("Frames")
        axes[label].set_ylabel("Saliency")
        axes[label].set_title(f"Label: {label_names[label]}", fontsize=10)
        axes[label].legend()

    # Save plot with detailed title
    plt.suptitle(f"{model_name} Model: Saliency Maps of Patient {patient + 1}, Trial {trial} \nActive Labels: {active_labels_str}", fontsize=12, weight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"saliency_maps/grad_cam/{model_name}_patient_{patient}_trial_{trial}_GC.png", dpi=300)
    plt.close()    
    
    return




def gradient_normal_II(input, targets, outputs, output_dir, lenghts, criterion, model_name, model, trial, patient, treshold_labels, attributions):

    label_names = ['General C.', 'Shoulder C.', 'Shoulder Elevation','Exaggerated Shoulder Abd.', 'Trunk C.','Head C.' ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for label in range(len(label_names)):
        
        slc = torch.relu(attributions[label])
    
        slc = slc[:,0:lenghts[0],:]
        if model_name != 'LSTM':
            slc = smooth_gradients(slc, kernel_size=10)
        saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='nearest').squeeze(0)  
        
        map = torch.zeros_like(saliency)
        
        for i in range(saliency.shape[0]):
            map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
        
        if patient == 17 and trial in [1, 15, 25, 35, 45, 55, 65]:
            
            active_labels = [label_names[i] for i in range(1,len(label_names)) if targets[:, i] > 0]
            if active_labels:
                active_labels_str = ", ".join(active_labels)
            else:
                active_labels_str = "No compensation"
            
            sns.heatmap(map[0][0:lenghts[0]].T.detach().cpu().numpy(), cmap='RdYlBu', ax=axes[label], cbar_kws={'label': 'Normalized Saliency'})
            axes[label].set_xlabel("Frames", fontsize=12)
            axes[label].set_ylabel("Joints", fontsize=12)
            axes[label].set_title(f"Label: {label_names[label]}", fontsize=14)
            axes[label].tick_params(axis='both', which='major', labelsize=10)
            
            if label == 0:
                fig2, axes2 = plt.subplots(1, 1, figsize=(10, 15))
                
                head_joints = map[0][0:lenghts[0],:].sum(dim=1).detach().cpu().numpy()
                axes2.plot(head_joints)
                axes2.set_title("Joints")
                axes2.set_xlabel("Frames")
                axes2.set_ylabel("Sum of Saliency")
                
                plt.tight_layout()
                plt.savefig(f"saliency_maps/vanilla_gradients_II/{model_name}_patient_{patient}_trial_{trial}_label_{label}_plots_VG_II.png", dpi=300)
                plt.close(fig2)
                
        if label == 0:       
            binary_map = (map[0][0:lenghts[0]].sum(dim=1) < treshold_labels).int().detach().cpu().numpy()
    
    if patient == 17 and trial in [1, 15, 25, 35, 45, 55, 65]:

        plt.suptitle(f"{model_name}: Saliency Maps of Patient {patient + 1}:  Trial {trial} \n Active Labels: {active_labels_str}", fontsize=16, weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(f"saliency_maps/vanilla_gradients_II/{model_name}_patient_{patient}_trial_{trial}_VG_II.png", dpi=300)
        plt.close()
        
    plt.close(fig)
    return binary_map
