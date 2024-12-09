import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap


def gradient_normal(input, targets, outputs, output_dir, lenghts, criterion, model_name, model):

    label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation','Exaggerated Shoulder Abduction', 'Trunk Compensation','Head Compensation' ]
    trial = 60
    
    loss = criterion(outputs, targets)
    model.zero_grad()
    
    loss.backward(retain_graph=True)
    
    slc = torch.abs(input.grad)
    slc = torch.relu(slc)
    
    saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='nearest').squeeze(0)  
    
    map = torch.zeros_like(saliency)
    
    for i in range(saliency.shape[0]):
        map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())

    active_labels = [label_names[i] for i in range(0 , len(label_names)) if targets[trial, i] > 0]
    active_labels_str = ", ".join(active_labels)
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(map[trial][0:lenghts[trial]].T.detach().cpu().numpy(), cmap='RdYlBu')  # Use the first input for visualization
    plt.xlabel("Frames")
    plt.ylabel("Joints")

    plt.title(f"{model_name}: Saliency Map of trial: {trial} ; Patient 18 \n Active Labels: {active_labels_str}", fontsize=10)

    plt.savefig(f"saliency_maps/{model_name}_trial_{trial}_normal_gradient.png")
    plt.close()
    
    return


def grad_cam(input, targets, outputs, output_dir, lenghts, criterion, model_name,model, attention_weights):
    
    trial = 0
    label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation','Exaggerated Shoulder Abduction', 'Trunk Compensation','Head Compensation' ]
    
    loss = criterion(outputs, targets)
    model.zero_grad()
    
    loss.backward(retain_graph=True)
    grads_input = input.grad
    
    attention_weights_mean = attention_weights.mean(dim=1)
    
    slc = torch.einsum('bsi,bsi->bsi', attention_weights_mean[:,1:,1:].sum(dim=2).unsqueeze(-1), grads_input)
    # slc = torch.einsum('bsi,bsi->bsi', attention_weights_mean[:,0,1:].unsqueeze(-1), grads_input)
    
    saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='bilinear', align_corners=False).squeeze(0) 

    saliency = torch.relu(saliency)
    
    cmap = torch.zeros_like(saliency)
    
    for i in range(saliency.shape[0]):
        cmap[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
    
    active_labels = [label_names[i] for i in range(len(label_names)) if targets[trial, i] > 0]
    active_labels_str = ", ".join(active_labels)
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(cmap[trial][0:lenghts[trial]].T.detach().cpu().numpy(), cmap='RdYlBu')  # Use the first input for visualization
    plt.xlabel("Frames")
    plt.ylabel("Joints")
    plt.xlabel("Frames")
    plt.ylabel("Joints")

    plt.title(f"{model_name}: Saliency Map of trial: {trial} ; Patient 18 \n Active Labels: {active_labels_str}", fontsize=10)

    plt.savefig(f"saliency_maps/{model_name}_trial_{trial}_grad_cam.png")
    plt.close()
    
    return


def interpolate_inputs(baseline, input, steps):

    alphas = np.linspace(0, 1, steps).reshape(-1, 1, 1)  # Interpolation coefficients
    interpolated = baseline + alphas * (input - baseline)
    return interpolated

def compute_gradients(model, inputs, target_label_idx):

    inputs.requires_grad = True  # Enable gradients computation
    output = model(inputs)  # Forward pass
    target_output = output[:, target_label_idx]  # Select the target label's output

    # Compute gradients with respect to the inputs
    target_output.backward(torch.ones_like(target_output))
    gradients = inputs.grad
    return gradients

def integrated_gradients(model, input, targets, lenghts, model_name, baseline=None, target_label_idx=0, steps=50):
    
    label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation','Exaggerated Shoulder Abduction', 'Trunk Compensation','Head Compensation' ]
    
    trial = 60
    if baseline is None:
        baseline = torch.zeros_like(input)  # Set baseline as all-zero tensor by default

    # Generate interpolated inputs
    interpolated_inputs = interpolate_inputs(baseline, input, steps)

    # Initialize total gradients
    total_gradients = torch.zeros_like(input)

    # Compute the gradients for each interpolated input and accumulate
    for i, interpolated_input in enumerate(interpolated_inputs):
        interpolated_input_tensor = torch.tensor(interpolated_input, dtype=torch.float32)
        gradients = compute_gradients(model, interpolated_input_tensor, target_label_idx)
        total_gradients += gradients
    
    # Average gradients and multiply by the input - baseline difference
    avg_gradients = total_gradients / steps
    integrated_gradients = (input - baseline) * avg_gradients
    
    active_labels = [label_names[i] for i in range(len(label_names)) if targets[trial, i] > 0]
    active_labels_str = ", ".join(active_labels)
    
    plt.figure(figsize=(10, 5))
    sns.heatmap(integrated_gradients[trial][0:lenghts[trial]].T.detach().cpu().numpy(), cmap='RdYlBu')  # Use the first input for visualization
    plt.xlabel("Frames")
    plt.ylabel("Joints")
    plt.xlabel("Frames")
    plt.ylabel("Joints")

    plt.title(f"{model_name}: Saliency Map of trial: {trial} ; Patient 18 \n Active Labels: {active_labels_str}", fontsize=10)

    plt.savefig(f"saliency_maps/{model_name}_trial_{trial}_grad_cam.png")
    plt.close()
    
    return integrated_gradients








