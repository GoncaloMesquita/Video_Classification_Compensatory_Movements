import torch
import numpy as np  
import torch.nn as nn
import torch.nn.utils as utils
from sklearn.metrics import f1_score


def training(model, dataloader, optimizer, criterion, device, save_dir, model_name, thresholds, clip_value, optuna, epoch, scaler):
    
    model.train()
    running_loss = 0.0
    all_targets, all_sigmoid_outputs, all_predictions = [], [], []
    thresholds = torch.tensor(thresholds, device=device)
    
    if epoch == 0 and model_name in ['moment', 'AcT', 'SkateFormer', 'moment+dino']:
        
        if model_name == 'moment+dino':
            for name, param in model.named_parameters():
                if any(substring in name for substring in ["dinov2.encoder.layer.5.attention.attention", 'moment_model.encoder.block.7.layer.0.SelfAttention']):
                    param.requires_grad = True
                    print('Unfreez Layers:' ,name)
                    
        elif model_name == 'moment':
            for name, param in model.named_parameters():
                if name.startswith('module.moment_model.head.') or name.startswith('module.moment_model.encoder.block.7.'):
                    param.requires_grad = True
                    print('Unfreezing layer:', name)
        else:
            for param in model.parameters():
                param.requires_grad = True
                
            print("All layers unfrozen!")
    
        for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(dataloader):
            
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            lengths = lengths.to('cpu')

            optimizer.zero_grad()  # Clear gradients before new batch

            with torch.no_grad():  # Prevents unnecessary memory usage
                if model_name == 'moment':
                    outputs = model(inputs)
                elif model_name == 'moment+dino':
                    inputs2 = inputs2.to(device, non_blocking=True).float()
                    outputs = model(inputs, inputs2)
                else:
                    outputs = model(inputs, lengths)

            loss = criterion(outputs, targets)
            loss.backward(retain_graph=False)  # Ensure `retain_graph=False`
            optimizer.step()

            # Convert and store results efficiently
            # with torch.no_grad():
            #     sigmoid_outputs = torch.sigmoid(outputs).detach().cpu().numpy()
            #     predicted = sigmoid_outputs > thresholds
            #     all_sigmoid_outputs.append(sigmoid_outputs)
            #     all_predictions.append(predicted)
            #     all_targets.append(targets.cpu().numpy())

            # Remove unnecessary tensors & free memory
            del inputs, targets, outputs, inputs2
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    if optuna:
        return running_loss / len(dataloader)
    
    # all_targets = np.concatenate(all_targets, axis=0)
    # all_predictions = np.concatenate(all_predictions, axis=0)
    # all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0) 
    all_targets = None
    all_predictions = None  
    # metrics(all_targets, all_predictions, "train", save_dir, model_name)

    return running_loss / len(dataloader), all_targets, all_sigmoid_outputs


def validate(model, val_loader, criterion, device, model_name, thresholds, optuna):
    
    
    val_loss = 0.0
    all_targets, all_predictions, all_sigmoid_outputs = [], [], []
    thresholds = torch.tensor(thresholds)
    
    model.eval()
    with torch.no_grad():
    
        for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(val_loader):
            inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
            lengths = lengths.to('cpu')
            
            if model_name == 'moment':
                outputs = model(inputs)
            elif model_name == 'moment+dino':
                inputs2 = inputs2.to(device, non_blocking=True).float()
                outputs = model(inputs, inputs2)
            else: 
                outputs = model(inputs, lengths)
                
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            
            sigmoid_outputs = torch.sigmoid(outputs) 
            predicted = sigmoid_outputs > thresholds.to(inputs.device)
            
            all_sigmoid_outputs.append(sigmoid_outputs.detach().cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

        
    val_loss /= len(val_loader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_sigmoid_outputs = np.concatenate(all_sigmoid_outputs, axis=0)
    
    if optuna:
        
        return val_loss, f1_score(all_targets, all_predictions, average='samples', zero_division=0)
    
    # _ = metrics(all_targets, all_predictions, mode, save_dir, model_name)
    torch.cuda.empty_cache()
    return val_loss, all_targets, all_predictions, all_sigmoid_outputs

