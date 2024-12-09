import torch
import numpy as np  
import torch.nn as nn
from utils import EarlyStopping, metrics


def training(model, dataloader, optimizer, criterion, device, output_dir):
    
    model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        predicted = torch.round(torch.sigmoid(outputs))
            
        all_targets.append(targets.cpu().detach().numpy())
        all_predictions.append(predicted.cpu().detach().numpy())
    
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    metrics(all_targets, all_predictions, "train", output_dir)

    return running_loss / len(dataloader)


def validate(model, val_loader, criterion, device, mode):
    
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            val_loss += loss.item()
            predicted = torch.round(torch.sigmoid(outputs))
            
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())
    
        val_loss /= len(val_loader)
        
        all_targets = np.concatenate(all_targets, axis=0)
        all_predictions = np.concatenate(all_predictions, axis=0)
        metrics(all_targets, all_predictions, mode)
    
    return val_loss, all_targets, all_predictions
        