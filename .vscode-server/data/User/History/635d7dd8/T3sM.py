import torch
import numpy as np  
import torch.nn as nn
from utils import EarlyStopping, metrics
import torch.nn.utils as utils


def training(model, dataloader, optimizer, criterion, device, save_dir, model_name):
    
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    clip_value = 0.5
    
    for batch_idx, (inputs, lengths, targets) in enumerate(dataloader):
        
        inputs,  targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        
        loss = criterion(outputs, targets)
        loss.backward()
        
        if clip_value is not None:
            utils.clip_grad_norm_(model.parameters(), clip_value)
            
        optimizer.step()
        
        running_loss += loss.item()
        
        # predicted = torch.round(torch.sigmoid(outputs))
        predicted = torch.sigmoid(outputs) > 0.25
        all_targets.append(targets.cpu().detach().numpy())
        all_predictions.append(predicted.cpu().detach().numpy())
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    metrics(all_targets, all_predictions, "train", save_dir, model_name)

    return running_loss / len(dataloader)


def validate(model, val_loader, criterion, device, mode, save_dir, model_name):
    
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
        predicted = torch.sigmoid(outputs) > 0.25
        
        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predicted.cpu().numpy())

    val_loss /= len(val_loader)
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    metrics(all_targets, all_predictions, mode, save_dir, model_name)
    
    return val_loss, all_targets, all_predictions
        