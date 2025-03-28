import torch
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, Saliency
from captum.attr import LayerGradCam
from captum.attr import Saliency
import os
import pickle
import numpy as np


def pseudo_label(model, test_loader, device, save_dir, model_name, patient, treshold_labels, method, input_size, label_names):

    pseudo_labels = []
    
    if method == 'ig':
        pseudo_labels.append(gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels, input_size, label_names))
    
    elif method == 'vg':
        pseudo_labels.append(vanilla_gradients(test_loader, model_name, model, patient, device, treshold_labels, input_size, label_names))
    
    # elif method == 'grad_cam':
    #     pseudo_labels.append(grad_cam(test_loader, criterion, model_name, model, patient, device, treshold_labels))
            
    save_pseudo_labels(pseudo_labels, model_name, patient, method, treshold_labels)
    return pseudo_labels


def gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels, input_size, label_names):

    model.eval()
    if model_name == 'LSTM':
        model.train()
        
    binary_map = []
    alpha_percentil = 90
    
    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
        lengths = lengths.to('cpu')
        
        inputs.requires_grad = True
        batch_binary_map = []

        
        for j in range (0, len(label_names)):
            if model_name == 'moment':
                ig_gradients = ig_grad(inputs, model, lengths, j, baseline=None, steps=5)
            else: 
                ig = IntegratedGradients(model)
                ig_gradients = ig.attribute(inputs, baselines=inputs *0, target=j, additional_forward_args=lengths)
            
            slc = torch.relu(ig_gradients)
            
            if model_name != 'LSTM':
                slc = smooth_gradients(slc, kernel_size=8)
            saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], input_size), mode='nearest').squeeze(0)
            
            map = torch.zeros_like(saliency)
            
            for i in range(saliency.shape[0]):
                map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
                    
            for i in range(map.shape[0]):
                
                values = map[i][:lengths[i]].sum(axis=1).detach().cpu().numpy()
                alpha_threshold = np.percentile(values, alpha_percentil)  
                
                bi_map = np.full(values.shape, np.nan)
                bi_map[values > alpha_threshold] = 0  
                bi_map[values <= alpha_threshold] = 1  
                
                if targets[i, j] == 1:
                    bi_map[i,j] = 1
                
                batch_binary_map.append(bi_map)
                
            torch.cuda.empty_cache()
        
        batch_binary_map = np.array(batch_binary_map).T  # Transpose to make rows as batch and columns as labels
        binary_map.append(batch_binary_map)
    
    # Combine all batches into a single dataset
    binary_map = np.vstack(binary_map)
        
    return binary_map


def vanilla_gradients(test_loader, model_name, model, patient, device, treshold_labels, input_size, label_names):

    model.eval()
    if model_name == 'LSTM':
        model.train()
        
    binary_map = []
    alpha_percentil = 75
    maps_labels  = []
    for j in range(0, len(label_names)):
        
        batch_binary_map = []  
        maps = []
                
        for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
            inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
            lengths = lengths.to('cpu')
            inputs.requires_grad = True
            

            saliency = Saliency(model)
            
            if model_name == 'moment':
                grads = saliency.attribute(inputs, target=j)
            else:
                grads = saliency.attribute(inputs, target=j, additional_forward_args=lengths)
                
            slc = torch.relu(grads)
            
            if model_name != 'LSTM':
                slc = smooth_gradients(slc, kernel_size=8)
            saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], input_size), mode='nearest').squeeze(0)
            
            map = torch.zeros_like(saliency)
            
            for i in range(saliency.shape[0]):
                map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())

                values = map[i][:lengths[i]].sum(axis=1).detach().cpu().numpy()
                maps.append(values)
                
        maps_labels.append(maps)
        
    for i in range(maps_labels.shape[0]):
        
        alpha_threshold = np.percentile(values, alpha_percentil)  
        
        bi_map = np.full(values.shape, np.nan)
        bi_map[values > alpha_threshold] = 0  
        bi_map[values <= alpha_threshold] = 1  
        
        # Ensure pseudo label is 1 if the target is 1
        if targets[i, j] == 1:
            bi_map[j] = 1
        
        batch_binary_map.append(bi_map.tolist())
    
        # Combine binary maps for all labels into a single dataset row
        batch_binary_map = np.array(batch_binary_map).T  # Transpose to make rows as batch and columns as labels
        binary_map.append(batch_binary_map)
    
    # Combine all batches into a single dataset
    binary_map = np.vstack(binary_map)
    return binary_map


def save_thresholds_and_maps(thresholds, maps, model_name, patient):
    path = f'dataset/thresholds_maps/{model_name}'
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, f'thresholds_{patient+1}.pkl'), 'wb') as f:
        pickle.dump(thresholds, f)
    
    with open(os.path.join(path, f'maps_{patient+1}.pkl'), 'wb') as f:
        pickle.dump(maps, f)


def ig_grad(inputs, model, lengths, target_label_idx, baseline=None, steps=5):
    
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(inputs.device)
    
    integrated_grads = torch.zeros_like(inputs).to(inputs.device)
    
    for i in range(steps + 1):
        alpha = float(i) / steps
        scaled_input = baseline + alpha * (inputs - baseline)
        scaled_input = scaled_input.requires_grad_(True)

        outputs = model(scaled_input)

        loss = outputs[:, target_label_idx].sum()
        
        grads = torch.autograd.grad(loss, scaled_input, create_graph=False)[0]
        integrated_grads += grads * (inputs - baseline) / steps

        model.zero_grad()
        scaled_input.grad = None
        del scaled_input, outputs, loss
        torch.cuda.empty_cache()

    del inputs
    # Detach integrated gradients and return
    return integrated_grads.detach()


def smooth_gradients(grad, kernel_size=5):
    num_joints = grad.shape[2]
    
    # Create the smoothing kernel
    kernel = torch.ones((num_joints, 1, kernel_size), dtype=grad.dtype, device=grad.device) / kernel_size
    
    # Apply convolution to smooth the gradients
    smoothed_grad = torch.nn.functional.conv1d(
        grad.transpose(1, 2),  # Shape: (batch_size, num_joints, seq_length)
        kernel,
        padding=kernel_size // 2,
        groups=num_joints
    ).transpose(1, 2)  # Shape back to (batch_size, seq_length, num_joints)
    
    return smoothed_grad


def save_pseudo_labels(pseudo_labels, model_name, patient, method, treshold_labels):
    
    labels = [label for sublist in pseudo_labels for label in sublist]
    if len(treshold_labels) == 1:
        method_type = 'method1'
        trh = treshold_labels[0]
    else:
        method_type = 'method2'
        trh = f'{treshold_labels[0]}' + '_' + f'{treshold_labels[1]}'
    
    path = f'dataset/pseudo_labels/{model_name}/{method_type}/{method}_{trh}'
    os.makedirs(path, exist_ok=True)
    
    with open(os.path.join(path, f'pseudo_labels_{patient+1}.pkl'), 'wb') as f:
        pickle.dump(labels, f)
