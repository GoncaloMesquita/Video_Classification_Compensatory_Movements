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


def pseudo_label(model, test_loader, device, save_dir, model_name, patient, treshold_labels, method):

    pseudo_labels = []
    
    if method == 'ig':
        pseudo_labels.append(gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels))
    
    elif method == 'vg':
        pseudo_labels.append(vanilla_gradients(test_loader, model_name, model, patient, device, treshold_labels))
    
    # elif method == 'grad_cam':
    #     pseudo_labels.append(grad_cam(test_loader, criterion, model_name, model, patient, device, treshold_labels))
            
    save_pseudo_labels(pseudo_labels, model_name, patient, method, treshold_labels)
    return pseudo_labels


def gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels):

    model.eval()
    if model_name == 'LSTM':
        model.train()
    binary_map = []
    
    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
        lengths = lengths.to('cpu')
        
        inputs.requires_grad = True
        
        if model_name == 'moment':
            ig_gradients = ig_grad(inputs, model, lengths, 0, baseline=None, steps=5)
        else: 
            ig = IntegratedGradients(model)
            ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=0, additional_forward_args=lengths)
        
        slc = torch.relu(ig_gradients)
        
        if model_name != 'LSTM':
            slc = smooth_gradients(slc, kernel_size=8)
        saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='nearest').squeeze(0)
        
        map = torch.zeros_like(saliency)
        
        for i in range(saliency.shape[0]):
            map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
        
        # still need to revise this for several batches
        if len(treshold_labels) == 1:
            for i in range(0,map.shape[0]):
                binary_map.append((map[i][0:lengths[i]].sum(dim=1) < treshold_labels[0]).int().detach().cpu().numpy())
        else:
            for i in range(map.shape[0]):
                
                values = map[i][:lengths[i]].sum(axis=1).detach().cpu().numpy()  
                bi_map = np.full(values.shape, np.nan) 
                
                bi_map[values > treshold_labels[1]] = 0  
                bi_map[values < treshold_labels[0]] = 1  
                
                binary_map.append(bi_map)
        
        torch.cuda.empty_cache()
        
    return binary_map


def vanilla_gradients(test_loader,  model_name, model, patient, device, treshold_labels):\

    model.eval()
    if model_name == 'LSTM':
        model.train()
        
    binary_map = []
    
    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
        lengths = lengths.to('cpu')
        
        inputs.requires_grad = True
        
        if model_name == 'moment':
            outputs = model(inputs)
            vg = inputs.grad

        else: 
            outputs = model(inputs, lengths)
            vg = inputs.grad

        slc = torch.relu(vg)
        
        if model_name != 'LSTM':
            slc = smooth_gradients(slc, kernel_size=8)
        saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], 33), mode='nearest').squeeze(0)
        
        map = torch.zeros_like(saliency)
        
        for i in range(saliency.shape[0]):
            map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
        
        # still need to revise this for several batches
        if len(treshold_labels) == 1:
            for i in range(0,map.shape[0]):
                binary_map.append((map[i][0:lengths[i]].sum(dim=1) < treshold_labels[0]).int().detach().cpu().numpy())
            
        else:
            for i in range(map.shape[0]):
                
                values = map[i][:lengths[i]].sum(axis=1).detach().cpu().numpy()  
                bi_map = np.full(values.shape, np.nan) 
                
                bi_map[values > treshold_labels[1]] = 0  
                bi_map[values < treshold_labels[0]] = 1  
                
                binary_map.append(bi_map)
        
    return binary_map


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
