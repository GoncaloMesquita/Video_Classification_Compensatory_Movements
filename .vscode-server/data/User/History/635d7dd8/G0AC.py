import torch
import numpy as np  
import torch.nn as nn
from utils import EarlyStopping, metrics
import torch.nn.utils as utils
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def training(model, dataloader, optimizer, criterion, device, save_dir, model_name, treshold, clip_value, optuna, pretrained):
    
    if pretrained:
                
        unfreeze_layers = ['dense1.weight', 'final_dense.weight', 'patch_embed.position_embedding.weight']

        for name, param in model.named_parameters():
            if name in unfreeze_layers:
                param.requires_grad = True
    
    
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    for batch_idx, (inputs, lengths, targets) in enumerate(dataloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        
        loss = criterion(outputs, targets)
        if loss.grad_fn is None:
            print("Error: loss does not have grad_fn. Check requires_grad=True for all parameters.")
            return
        loss.backward()
        
        if clip_value is not None:
            utils.clip_grad_norm_(model.parameters(), clip_value)
            
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = torch.sigmoid(outputs) > treshold
        all_targets.append(targets.cpu().detach().numpy())
        all_predictions.append(predicted.cpu().detach().numpy())
    
    if optuna:
        return running_loss / len(dataloader), model
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    metrics(all_targets, all_predictions, "train", save_dir, model_name)

    return running_loss / len(dataloader)


def validate(model, val_loader, criterion, device, mode, save_dir, model_name, treshold, optuna):
    
    model.eval()
    val_loss = 0.0
    all_targets = []
    all_predictions = []

    # with torch.no_grad():
    
    for batch_idx, (inputs, lengths, targets) in enumerate(val_loader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs, lengths)
        loss = criterion(outputs, targets)
        
        val_loss += loss.item()
        # predicted = torch.round(torch.sigmoid(outputs))
        predicted = torch.sigmoid(outputs) > treshold
        
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    

    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    if optuna:
        
        return val_loss, f1_score(all_targets, all_predictions, average='samples', zero_division=0)
    
    metrics(all_targets, all_predictions, mode, save_dir, model_name)
    
    return val_loss, all_targets, all_predictions
        