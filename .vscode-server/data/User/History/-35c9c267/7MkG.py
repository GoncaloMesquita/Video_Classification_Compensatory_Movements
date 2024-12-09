import torch
import numpy as np  
import torch.nn as nn
from utils import EarlyStopping, metrics
import torch.nn.utils as utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import metrics, plot_auc_curves
import matplotlib.pyplot as plt
import seaborn as sns
import torch.backends.cudnn as cudnn
from visualization import gradient_normal, grad_cam, integrated_gradients




def training(model, dataloader, optimizer, criterion, device, save_dir, model_name, thresholds, clip_value, optuna):
    
    
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    thresholds = torch.tensor(thresholds)
    
    for batch_idx, (inputs, lengths, targets) in enumerate(dataloader):
        
        inputs, targets, lengths = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32), lengths.to(device)
        
        optimizer.zero_grad()
        if model_name == 'moment':
            outputs = model(inputs)
        else: 
            outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        loss.backward()
        
        if clip_value is not None:
            utils.clip_grad_value_(model.parameters(), clip_value)
                            
        optimizer.step()
        
        running_loss += loss.item()
        
        sigmoid_outputs = torch.sigmoid(outputs) 
        predicted = sigmoid_outputs > thresholds.to(inputs.device)
        all_targets.append(targets.cpu().detach().numpy())
        all_predictions.append(predicted.cpu().detach().numpy())
    
    if optuna:
        return running_loss / len(dataloader)
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    metrics(all_targets, all_predictions, "train", save_dir, model_name)

    return running_loss / len(dataloader)


def validate(model, val_loader, criterion, device, mode, save_dir, model_name, thresholds, optuna):
    
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []
    all_sigmoid_outputs = []
    thresholds = torch.tensor(thresholds)
    # with torch.no_grad():
    
    for batch_idx, (inputs, lengths, targets) in enumerate(val_loader):
        
        inputs, targets = inputs.to(device, dtype=torch.float32), targets.to(device, dtype=torch.float32)
        
        # inputs.requires_grad = True
        
        if model_name == 'moment':
            outputs = model(inputs)
        else: 
            outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        
        val_loss += loss.item()
        sigmoid_outputs = torch.sigmoid(outputs) 
        predicted = sigmoid_outputs > thresholds.to(inputs.device)
        
        all_sigmoid_outputs.append(sigmoid_outputs.detach().cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predicted.cpu().numpy())
        
        # if saliency_map and mode == 'testing':
            # gradient_normal(inputs, targets, outputs, save_dir, lengths, criterion, model_name, model)
            
            # grad_cam(inputs, targets, outputs, save_dir, lengths, criterion, model_name, model, model.transformer.layers[-1].mha.attention_weights)
            
            # integrated_gradients(inputs, targets, lengths, model_name, save_dir, model_name, model)
            #shapley_values(inputs, targets, model, save_dir, model_name, lengths)
            #etfa(input, targets, outputs, model, save_dir, model_name,model.transformer.layers[-1].mha.attention_weights )
            # layerwise_relevance_propagation(input, targets, sigmoid_outputs, model, save_dir, model_name)
            
    

    val_loss /= len(val_loader)

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0)
    
    if optuna:
        
        return val_loss, f1_score(all_targets, all_predictions, average='samples', zero_division=0)
    
    _ = metrics(all_targets, all_predictions, mode, save_dir, model_name)
    
    return val_loss, all_targets, all_predictions, all_sigmoid_outputs
        