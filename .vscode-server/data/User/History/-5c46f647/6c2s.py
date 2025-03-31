import torch
from captum.attr import IntegratedGradients, Saliency
import os
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, hamming_loss
import numpy as np


def pseudo_label(model, test_loader, device, save_dir, model_name, patient, treshold_labels, method, dataset_name, true_data, input_size, num_dimension):

    pseudo_labels = []
    
    if len(treshold_labels) == 1:
        method_type = 'method1'
        trh = treshold_labels[0]
    else:
        method_type = 'method2'
        trh = f'{treshold_labels[0]}' + '_' + f'{treshold_labels[1]}'
        
    save_dir = f'dataset/pseudo_labels/{dataset_name}/{model_name}/{method_type}/{method}_{trh}'
    os.makedirs(save_dir, exist_ok=True)
    
    if method == 'ig':
        pseudo_labels.append(gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels, input_size, dataset_name, num_dimension))
    
    elif method == 'vg':
        pseudo_labels.append(vanilla_gradients(test_loader, model_name, model, patient, device, treshold_labels, input_size, dataset_name, num_dimension))
    
    # pseudo_data_set_info(true_data, pseudo_labels, num_patients, save_dir)
    save_pseudo_labels(pseudo_labels, model_name, patient, method, treshold_labels, save_dir)
    
    return pseudo_labels


# def gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels, input_size, dataset_name):

#     model.eval()
#     if model_name == 'LSTM':
#         model.train()
#     binary_map = []
    
#     if dataset_name == 'SERE':
#         label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
#     elif dataset_name == 'Toronto':
#         label_names = ['General compensation']
#     elif dataset_name == 'MMAct':
#         label_names = ['standing', 'crouching', 'walking',
#                        'running', 'checking_time', 'waving_hand', 
#                        'using_phone', 'talking_on_phone', 'kicking', 
#                        'pointing', 'throwing', 'jumping', 
#                        'exiting', 'entering', 'setting_down', 
#                        'talking', 'opening', 'closing',
#                        'carrying', 'loitering', 'transferring_object',
#                        'looking_around', 'pushing', 'pulling',
#                        'picking_up', 'fall', 'sitting_down', 
#                        'using_pc', 'drinking', 'pocket_out', 
#                        'pocket_in', 'sitting', 'using_phone_desk', 
#                        'talking_on_phone_desk', 'standing_up', 
#                        'carrying_light', 'carrying_heavy', 'Carrying_light']
    
#     for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
#         inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
#         lengths = lengths.to('cpu')
        
#         inputs.requires_grad = True
        
#         for i in range(len(label_names)):
#             if model_name == 'moment':
#                 ig_gradients = ig_grad(inputs, model, None, i, baseline=None, steps=5)
#             elif  model_name == 'SkateFormer':
#                 ig_gradients = ig_grad(inputs, model, lengths, i, baseline=None, steps=5)
#             else: 
#                 ig = IntegratedGradients(model)
#                 ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=i, additional_forward_args=lengths, n_steps=5)
        
#         slc = torch.relu(ig_gradients)
        
#         if model_name != 'LSTM':
#             slc = smooth_gradients(slc, kernel_size=8)
#         saliency = torch.nn.functional.interpolate(slc.unsqueeze(0), size=(slc.shape[1], input_size//3), mode='nearest').squeeze(0)
        
#         map = torch.zeros_like(saliency)
        
#         for i in range(saliency.shape[0]):
#             map[i] = (saliency[i] - saliency[i].min()) / (saliency[i].max() - saliency[i].min())
        
#         # still need to revise this for several batches
#         if len(treshold_labels) == 1:
#             for i in range(0,map.shape[0]):
#                 binary_map.append((map[i][0:lengths[i]].sum(dim=1) < treshold_labels[0]).int().detach().cpu().numpy())
#         else:
#             for i in range(map.shape[0]):
                
#                 values = map[i][:lengths[i]].sum(axis=1).detach().cpu().numpy()  
#                 bi_map = np.full(values.shape, np.nan) 
                
#                 bi_map[values > treshold_labels[1]] = 0  
#                 bi_map[values < treshold_labels[0]] = 1  
                
#                 binary_map.append(bi_map)
        
#         torch.cuda.empty_cache()
        
#     return binary_map

# def gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels, input_size, dataset_name):

#     model.eval()
#     if model_name == 'LSTM':
#         model.train()
        
#     if dataset_name == 'SERE':
#         label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
#     elif dataset_name == 'Toronto':
#         label_names = ['General compensation']
#     elif dataset_name == 'MMAct':
#         label_names = ['standing', 'crouching', 'walking',
#                        'running', 'checking_time', 'waving_hand', 
#                        'using_phone', 'talking_on_phone', 'kicking', 
#                        'pointing', 'throwing', 'jumping', 
#                        'exiting', 'entering', 'setting_down', 
#                        'talking', 'opening', 'closing',
#                        'carrying', 'loitering', 'transferring_object',
#                        'looking_around', 'pushing', 'pulling',
#                        'picking_up', 'fall', 'sitting_down', 
#                        'using_pc', 'drinking', 'pocket_out', 
#                        'pocket_in', 'sitting', 'using_phone_desk', 
#                        'talking_on_phone_desk', 'standing_up', 
#                        'carrying_light', 'carrying_heavy', 'Carrying_light']
        
#     binary_map = []
#     action_maps = {label: [] for label in label_names}
#     alpha = 0.15
    
#     for i in range(len(label_names)):
    
#         thresholds = []
#         maps = []
        
#         for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
#             inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
#             lengths = lengths.to('cpu')
        
#             inputs.requires_grad = True

#             if model_name == 'moment':
#                 ig_gradients = ig_grad(inputs, model, None, i, baseline=None, steps=5)
#             elif  model_name == 'SkateFormer':
#                 ig_gradients = ig_grad(inputs, model, lengths, i, baseline=None, steps=5)
#             else: 
#                 ig = IntegratedGradients(model)
#                 ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=i, additional_forward_args=lengths, n_steps=5)

#             slc = torch.relu(ig_gradients)

#             if model_name != 'LSTM':
#                 slc = smooth_gradients(slc, kernel_size=8)
            
#             saliency = torch.nn.functional.interpolate(
#                 slc.unsqueeze(0), 
#                 size=(slc.shape[1], input_size // 3), 
#                 mode='nearest'
#             ).squeeze(0)
            
#             map = torch.zeros_like(saliency)
#             for j in range(saliency.shape[0]):
#                 map_min = saliency[j].min()
#                 map_max = saliency[j].max()
#                 map[j] = (saliency[j] - map_min) / (map_max - map_min + 1e-8)
            

#             # 1) Threshold per label
#             for j in range(map.shape[0]):
#                 data_1d = map[j][:lengths[j]].sum(axis=1)
                
#                 # Sort ascending
#                 sorted_data, _ = torch.sort(data_1d)
                
#                 # Index for (1 - alpha) quantile
#                 n = data_1d.shape[0]
#                 k = int(torch.ceil((1 - torch.tensor(alpha)) * n))
                
#                 # The threshold for label i
#                 threshold_i = sorted_data[k - 1]  # zero-based index => k-1
#                 thresholds.append(threshold_i.cpu().detach())
#                 maps.append(data_1d.cpu().detach())
            
#         # 2) Average those thresholds
#         thresholds_t = torch.stack(thresholds)      # shape: [num_labels]
#         avg_threshold = thresholds_t.mean()         # single scalar
        
#         # 3) Apply the single average threshold
#         for trial in maps:
#                 # Values <= avg_threshold => 1, above => 0
#             binarized = (trial <= avg_threshold).detach().cpu().int().tolist()
#             action_maps[label_names[i]].append(binarized)
                    
#     binary_map.append([action_maps[label] for label in label_names])
        
#     return binary_map


def gradient_integrated(test_loader, model_name, model, patient, device, treshold_labels, input_size, dataset_name, n_dim):

    model.eval()
    if model_name == 'LSTM':
        model.train()

    if dataset_name == 'SERE':
        label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 
                       'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
    elif dataset_name == 'Toronto':
        label_names = ['General compensation']
    elif dataset_name == 'MMAct':
        label_names = ['standing', 'crouching', 'walking', 'running', 'checking_time', 'waving_hand',
                       'using_phone', 'talking_on_phone', 'kicking', 'pointing', 'throwing', 'jumping',
                       'exiting', 'entering', 'setting_down', 'talking', 'opening', 'closing',
                       'carrying', 'loitering', 'transferring_object', 'looking_around', 'pushing', 'pulling',
                       'picking_up', 'fall', 'sitting_down', 'using_pc', 'drinking', 'pocket_out',
                       'pocket_in', 'sitting', 'using_phone_desk', 'talking_on_phone_desk', 'standing_up',
                       'carrying_light', 'carrying_heavy', 'Carrying_light']
        
    binary_map = []
    alpha = 0.38
    maps = []
    thresholds = []
    targets_save = []

    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
        lengths = lengths.to('cpu')

        inputs.requires_grad = True
        map_labels = []
        
        for i in range(len(label_names)):

            if model_name == 'moment':
                ig_gradients = ig_grad(inputs, model, None, i, baseline=None, steps=5)
            elif  model_name == 'SkateFormer':
                ig_gradients = ig_grad(inputs, model, lengths, i, baseline=None, steps=5)
            else: 
                ig = IntegratedGradients(model)
                ig_gradients = ig.attribute(inputs, baselines=inputs * 0, target=i, additional_forward_args=lengths, n_steps=2)

            slc = torch.relu(ig_gradients)
            
            if model_name != 'LSTM':
                slc = smooth_gradients(slc, kernel_size=8)
            
            saliency = torch.nn.functional.interpolate(
                slc.unsqueeze(0), 
                size=(slc.shape[1], input_size // n_dim), 
                mode='nearest'
            ).squeeze(0)
            
            map = torch.zeros_like(saliency)
            for j in range(saliency.shape[0]):
                mean_val = torch.mean(saliency[j])
                std_val = torch.std(saliency[j]) + 1e-8  
                map[j] = (saliency[j] - mean_val) / std_val
                map[j] = torch.clamp(map[j], min=-3.0, max=3.0)
                map[j] = (map[j] - map[j].min()) / (map[j].max() - map[j].min() + 1e-8)
        
            map_labels.append(map)
         
        targets_save.append(targets.tolist()) 
        map_labels = torch.stack(map_labels)  # stack all maps for each label
        map_labels = map_labels.permute(1, 2, 3, 0)  # Rearrange from [38,4,544,17] to [4,544,17,38]
        
        # 1) Threshold per label (apply directly to summed saliency maps)
        for j in range(map_labels.shape[0]):
            
            data_1d = map_labels[j][:lengths[j]].sum(axis=1)  # Sum directly
            
            sorted_data, _ = torch.sort(data_1d, axis=0)
            
            # Index for (1 - alpha) quantile
            n = data_1d.shape[0]
            k = int(torch.ceil((1 - torch.tensor(alpha)) * n))
            
            # The threshold for label i
            threshold_i = sorted_data[k - 1]  # zero-based index => k-1
            thresholds.append(threshold_i.cpu().detach())  # Store directly for individual use
            maps.append(data_1d.cpu().detach())  # Store per trial data
            
    targets_save = [item for sublist in targets_save for item in sublist]
    
    # Apply the threshold per label directly

    binary_map = []
    for i, (trial, trial_video, threshold_i) in enumerate(zip(maps, targets_save, thresholds)):  
        binarized = (trial <= threshold_i).detach().cpu().int().tolist()  # Apply threshold directly per instance

        for j in range(len(trial_video)):  
            if trial_video[j] == 1:  
                for frame in binarized:  
                    frame[j] = 1  
                    
                # Ensure only the label with the highest value in 'trial' keeps the zero
        # for frame_idx, frame in enumerate(binarized):
        #     zero_indices = [idx for idx, value in enumerate(frame) if value == 0]
        #     if zero_indices:
        #         max_value_idx = max(zero_indices, key=lambda idx: trial[frame_idx][idx])
        #     for idx in zero_indices:
        #         if idx != max_value_idx:
        #            frame[idx] = 1

        binary_map.append(binarized)
                  
    return binary_map


# def vanilla_gradients(test_loader,  model_name, model, patient, device, treshold_labels, input_size, dataset_name):

#     model.eval()
#     if model_name == 'LSTM':
#         model.train()
        
#     if dataset_name == 'SERE':
#         label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
#     elif dataset_name == 'Toronto':
#         label_names = ['General compensation']
#     elif dataset_name == 'MMAct':
#         label_names = ['standing', 'crouching', 'walking',
#                        'running', 'checking_time', 'waving_hand', 
#                        'using_phone', 'talking_on_phone', 'kicking', 
#                        'pointing', 'throwing', 'jumping', 
#                        'exiting', 'entering', 'setting_down', 
#                        'talking', 'opening', 'closing',
#                        'carrying', 'loitering', 'transferring_object',
#                        'looking_around', 'pushing', 'pulling',
#                        'picking_up', 'fall', 'sitting_down', 
#                        'using_pc', 'drinking', 'pocket_out', 
#                        'pocket_in', 'sitting', 'using_phone_desk', 
#                        'talking_on_phone_desk', 'standing_up', 
#                        'carrying_light', 'carrying_heavy', 'Carrying_light']
        
#     binary_map = []
#     alpha = 0.05
    
#     for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
#         inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
#         lengths = lengths.to('cpu')
        
#         inputs.requires_grad = True

#         vg_gradients = []
        
#         # -- Compute gradients for each label --
#         for i in range(len(label_names)):
#             if model_name == 'moment':
#                 vg = Saliency(model)
#                 vg_gradients = vg.attribute(inputs, target=i)
#                 # outputs = model(inputs)
#             else:
#                 vg = Saliency(model)
#                 vg_gradients = vg.attribute(inputs, target=i, additional_forward_args=(lengths,))
#                 # outputs = model(inputs, lengths)

#         # Shape: (num_labels, batch_size, channels, seq_length, ...)
        
#         # ReLU to keep only positive attributions
#             slc = torch.relu(vg_gradients)

#             # Optional smoothing
#             if model_name != 'LSTM':
#                 slc = smooth_gradients(slc, kernel_size=8)
            
#             # Interpolate to desired shape
#             saliency = torch.nn.functional.interpolate(
#                 slc.unsqueeze(0), 
#                 size=(slc.shape[1], input_size // 3), 
#                 mode='nearest'
#             ).squeeze(0)
            
#             # Normalize each label's map to [0,1]
#             map = torch.zeros_like(saliency)
#             for i in range(saliency.shape[0]):
#                 map_min = saliency[i].min()
#                 map_max = saliency[i].max()
#                 map[i] = (saliency[i] - map_min) / (map_max - map_min + 1e-8)
            
#             # --------------------------------------------------------------
#             # 1) Compute threshold per label via (1 - alpha) quantile
#             # 2) Average those thresholds
#             # 3) Binarize using the single averaged threshold
#             # --------------------------------------------------------------

#             # 1) Threshold per label
#             thresholds = []
#             for i in range(map.shape[0]):
#                 data_1d = map[i].flatten()
                
#                 # Sort ascending
#                 sorted_data, _ = torch.sort(data_1d)
                
#                 # Index for (1 - alpha) quantile
#                 n = data_1d.shape[0]
#                 k = int(torch.ceil((1 - alpha) * n))
                
#                 # The threshold for label i
#                 threshold_i = sorted_data[k - 1]  # zero-based index => k-1
#                 thresholds.append(threshold_i)
            
#             # 2) Average those thresholds
#             thresholds_t = torch.stack(thresholds)      # shape: [num_labels]
#             avg_threshold = thresholds_t.mean()         # single scalar
            
#             # 3) Apply the single average threshold
#             for i in range(map.shape[0]):
#                 # Values <= avg_threshold => 1, above => 0
#                 binarized = (map[i] <= avg_threshold).float()
#                 map[i] = binarized.view(map[i].shape)
                
                
#         # still need to revise this for several batches
#         # if len(treshold_labels) == 1:
#         #     for i in range(0,map.shape[0]):
#         #         binary_map.append((map[i][0:lengths[i]].sum(dim=1) < treshold_labels[0]).int().detach().cpu().numpy())
            
#         # else:
#         #     for i in range(map.shape[0]):
                
#         #         values = map[i][:lengths[i]].sum(axis=1).detach().cpu().numpy()  
#         #         bi_map = np.full(values.shape, np.nan) 
                
#         #         bi_map[values > treshold_labels[1]] = 0  
#         #         bi_map[values < treshold_labels[0]] = 1  
                
#         #         binary_map.append(bi_map)
        
#     return binary_map

def vanilla_gradients(test_loader, model_name, model, patient, device, treshold_labels, input_size, dataset_name, n_dim):

    model.eval()
    if model_name == 'LSTM':
        model.train()

    if dataset_name == 'SERE':
        label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 
                       'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
    elif dataset_name == 'Toronto':
        label_names = ['General compensation']
    elif dataset_name == 'MMAct':
        label_names = ['standing', 'crouching', 'walking', 'running', 'checking_time', 'waving_hand',
                       'using_phone', 'talking_on_phone', 'kicking', 'pointing', 'throwing', 'jumping',
                       'exiting', 'entering', 'setting_down', 'talking', 'opening', 'closing',
                       'carrying', 'loitering', 'transferring_object', 'looking_around', 'pushing', 'pulling',
                       'picking_up', 'fall', 'sitting_down', 'using_pc', 'drinking', 'pocket_out',
                       'pocket_in', 'sitting', 'using_phone_desk', 'talking_on_phone_desk', 'standing_up',
                       'carrying_light', 'carrying_heavy', 'Carrying_light']
        
    binary_map = []
    alpha = 0.30
    maps = []
    thresholds = []
    targets_save = []

    for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
        inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
        lengths = lengths.to('cpu')

        inputs.requires_grad = True
        map_labels = []
        
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
            
            saliency = torch.nn.functional.interpolate(
                slc.unsqueeze(0), 
                size=(slc.shape[1], input_size // n_dim), 
                mode='nearest'
            ).squeeze(0)
            
            map = torch.zeros_like(saliency)
            for j in range(saliency.shape[0]):
                mean_val = torch.mean(saliency[j])
                std_val = torch.std(saliency[j]) + 1e-8  
                map[j] = (saliency[j] - mean_val) / std_val
                map[j] = torch.clamp(map[j], min=-3.0, max=3.0)
                map[j] = (map[j] - map[j].min()) / (map[j].max() - map[j].min() + 1e-8)
        
            map_labels.append(map)
         
        targets_save.append(targets.tolist()) 
        map_labels = torch.stack(map_labels)  # stack all maps for each label
        map_labels = map_labels.permute(1, 2, 3, 0)  # Rearrange from [38,4,544,17] to [4,544,17,38]
        
        # 1) Threshold per label (apply directly to summed saliency maps)
        for j in range(map_labels.shape[0]):
            
            data_1d = map_labels[j][:lengths[j]].sum(axis=1)  # Sum directly
            
            sorted_data, _ = torch.sort(data_1d, axis=0)
            
            # Index for (1 - alpha) quantile
            n = data_1d.shape[0]
            k = int(torch.ceil((1 - torch.tensor(alpha)) * n))
            
            # The threshold for label i
            threshold_i = sorted_data[k - 1]  # zero-based index => k-1
            thresholds.append(threshold_i.cpu().detach())  # Store directly for individual use
            maps.append(data_1d.cpu().detach())  # Store per trial data
            
    targets_save = [item for sublist in targets_save for item in sublist]
    
    # Apply the threshold per label directly
    binary_map = []
    for i, (trial, trial_video, threshold_i) in enumerate(zip(maps, targets_save, thresholds)):  
        binarized = (trial <= threshold_i).detach().cpu().int().tolist()  # Apply threshold directly per instance

        for j in range(len(trial_video)):  
            if trial_video[j] == 1:  
                for frame in binarized:  
                    frame[j] = 1  
                    
        # for frame_idx, frame in enumerate(binarized):
        #     zero_indices = [idx for idx, value in enumerate(frame) if value == 0]
        #     if zero_indices:
        #         max_value_idx = max(zero_indices, key=lambda idx: trial[frame_idx][idx])
        #     for idx in zero_indices:
        #         if idx != max_value_idx:
        #            frame[idx] = 1

        binary_map.append(binarized)
                  
    return binary_map

# def vanilla_gradients(test_loader,  model_name, model, patient, device, treshold_labels, input_size, dataset_name):

#     model.eval()
#     if model_name == 'LSTM':
#         model.train()
        
#     if dataset_name == 'SERE':
#         label_names = ['General compensation', 'Shoulder Compensation', 'Shoulder Elevation', 'Exaggerated Shoulder Abduction', 'Trunk Compensation', 'Head Compensation']
#     elif dataset_name == 'Toronto':
#         label_names = ['General compensation']
#     elif dataset_name == 'MMAct':
#         label_names = ['standing', 'crouching', 'walking',
#                        'running', 'checking_time', 'waving_hand', 
#                        'using_phone', 'talking_on_phone', 'kicking', 
#                        'pointing', 'throwing', 'jumping', 
#                        'exiting', 'entering', 'setting_down', 
#                        'talking', 'opening', 'closing',
#                        'carrying', 'loitering', 'transferring_object',
#                        'looking_around', 'pushing', 'pulling',
#                        'picking_up', 'fall', 'sitting_down', 
#                        'using_pc', 'drinking', 'pocket_out', 
#                        'pocket_in', 'sitting', 'using_phone_desk', 
#                        'talking_on_phone_desk', 'standing_up', 
#                        'carrying_light', 'carrying_heavy', 'Carrying_light']
        
#     binary_map = []
#     alpha = 0.25
#     maps = []
#     thresholds = []
#     targets_save = []

#     for batch_idx, (inputs, targets, lengths, inputs2) in enumerate(test_loader):
        
#         inputs, targets = inputs.to(device, non_blocking=True).float(), targets.to(device, non_blocking=True).float()
#         lengths = lengths.to('cpu')
    
#         inputs.requires_grad = True
#         map_labels = []
        
#         for i in range(len(label_names)):

#             if model_name == 'moment':
#                 vg = Saliency(model)
#                 vg_gradients = vg.attribute(inputs, target=i)
#             else:
#                 vg = Saliency(model)
#                 vg_gradients = vg.attribute(inputs, target=i, additional_forward_args=(lengths,))

#             slc = torch.relu(vg_gradients)
            
#             if model_name != 'LSTM':
#                 slc = smooth_gradients(slc, kernel_size=8)
            
#             saliency = torch.nn.functional.interpolate(
#                 slc.unsqueeze(0), 
#                 size=(slc.shape[1], input_size // 3), 
#                 mode='nearest'
#             ).squeeze(0)
            
#             map = torch.zeros_like(saliency)
#             for j in range(saliency.shape[0]):
#                 map_min = saliency[j].min()
#                 map_max = saliency[j].max()
#                 map[j] = (saliency[j] - map_min) / (map_max - map_min + 1e-8)
        
#             map_labels.append(map)
         
#         targets_save.append(targets.tolist()) 
#         map_labels = torch.stack(map_labels)  # stack all maps for each label
#         map_labels = map_labels.permute(1, 2, 3, 0)  # Rearrange from [38,4,544,17] to [4,544,17,38]
#         # 1) Threshold per label
#         for j in range(map_labels.shape[0]):
            
#             data_1d = map_labels[j][:lengths[j]].sum(axis=1)
            
#             sorted_data, _ = torch.sort(data_1d, axis=0)
            
#             # Index for (1 - alpha) quantile
#             n = data_1d.shape[0]
#             k = int(torch.ceil((1 - torch.tensor(alpha)) * n))
            
#             # The threshold for label i
#             threshold_i = sorted_data[k - 1]  # zero-based index => k-1
#             thresholds.append(threshold_i.cpu().detach())
#             maps.append(data_1d.cpu().detach())
            
#     thresholds_t = torch.stack(thresholds)      
#     avg_threshold = thresholds_t.mean(axis=0)         
#     targets_save = [item for sublist in targets_save for item in sublist]
    
#     for i, (trial, trial_video) in enumerate(zip(maps, targets_save)):  
#         binarized = (trial <= avg_threshold).detach().cpu().int().tolist()

#         for j in range(len(trial_video)):  
#             if trial_video[j] == 1:  
#                 for frame in binarized:  
#                     frame[j] = 1  

#         binary_map.append(binarized)
                  
#     return binary_map


def ig_grad(inputs, model, lengths, target_label_idx, baseline=None, steps=5):
    
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(inputs.device)
    
    integrated_grads = torch.zeros_like(inputs).to(inputs.device)
    
    for i in range(steps + 1):
        alpha = float(i) / steps
        scaled_input = baseline + alpha * (inputs - baseline)
        scaled_input = scaled_input.requires_grad_(True)
        if lengths == None:
            outputs = model(scaled_input)
        else:    
            outputs = model(scaled_input, lengths)

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


def save_pseudo_labels(pseudo_labels, model_name, patient, method, treshold_labels, save_dir):
    
    labels = [label for sublist in pseudo_labels for label in sublist]

    
    with open(os.path.join(save_dir, f'pseudo_labels_{patient+1}.pkl'), 'wb') as f:
        pickle.dump(labels, f)



