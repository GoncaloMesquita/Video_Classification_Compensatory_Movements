import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
from create_model import create_model
from utils import EarlyStopping, plotting_loss, create_dataloader, load_data, save_best_model_callback, metrics, plot_auc_curves, plot_auc_test, metrics_evaluate, standardize_skeleton
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from engine import training, validate
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    hidden_size = trial.suggest_int("hidden_size", 192, 192)
    num_layers = trial.suggest_int("num_layers", 1,1)
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [64])
    epochs = trial.suggest_int("epochs", 250, 300, step=50)
    patience = trial.suggest_int("patience", 20, 25, step=5)
    delta = trial.suggest_float("delta", 1e-4, 1e-2,)
    clip_value = trial.suggest_float("clip_value", 0.2, 1.0, step=0.1)
    eta = trial.suggest_float("eta", 1e-7, 1e-5,)
    threshold = [trial.suggest_float(f"threshold_label_{i+1}", 0.1, 0.5, step=0.05) for i in range(6)]
    
    val_min = 1000
    # Initialize model
    device = torch.device(f'cuda:{args.n_device}' if torch.cuda.is_available() else 'cpu')
    
    
    data = load_data(args.data_label, args.data_skeletons, args.model_name, args.save_dir)
    
    for i , (X_t, y_t, X_val, y_val, X_test, y_test) in enumerate (data):

        print(f"Fold {i}")
        
        if i > 0:
            break
        
        X_train_standardized = standardize_skeleton(X_t)
        X_val_standardized = standardize_skeleton(X_val)
            
        train_loader = create_dataloader(X_train_standardized, y_t, batch_size, True, args.model_name)
        val_loader = create_dataloader(X_val_standardized, y_val, batch_size, False, args.model_name)
        
        model = create_model(args.model_name, args.input_size, hidden_size, num_layers, args.num_labels, dropout, args.checkpoint, args.mode, args.pretrained)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss().to(device)
        early_stopping = EarlyStopping(patience=patience, model_name=args.model_name, learning_rate=learning_rate, batch_size=batch_size, output_dir=args.save_dir, delta=delta, optuna=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta)

        for epoch in range(epochs):
            
            train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, threshold, clip_value, args.optuna)
            val_loss, f1_score = validate(model, val_loader, criterion, device, "validation", args.save_dir, args.model_name, threshold, args.optuna)

            early_stopping(val_loss, model, 0, epoch)
            if early_stopping.early_stop:
                break

            scheduler.step()
            
            if val_min > val_loss:
                val_min = val_loss
            
    return val_min


def train(args):
    
    args.optuna = False
    device = torch.device(f'cuda:{args.n_device}' if torch.cuda.is_available() else 'cpu')
    dist.init_process_group(backend='nccl') 
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    cross_val_data = load_data(args.data_label, args.data_skeletons, args.model_name, args.save_dir)    
    
    cross_train_losses = []
    cross_val_losses = []
    metrics_test = []
    auc_test = []
    mean_fpr = np.linspace(1e-6, 1, 100)
    fig, ax = plt.subplots(3,2, figsize=(20, 16))
    n_splits = 18    
    
    for i , (X_t, y_t, X_val, y_val, X_test, y_test) in enumerate (cross_val_data):
        
        print(f"Fold {i}")

        X_train_standardized = standardize_skeleton(X_t)
        X_val_standardized = standardize_skeleton(X_val)
        
        train_loader = create_dataloader(X_train_standardized, y_t, args.batch_size, True, args.model_name)
        val_loader = create_dataloader(X_val_standardized, y_val, args.batch_size, False, args.model_name)

        early_stopping = EarlyStopping(patience=args.patience, model_name=args.model_name, learning_rate=args.learning_rate,batch_size = args.batch_size, output_dir=args.save_dir ,verbose=True, delta=args.delta)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, args.mode, args.pretrained)
        # model.to(device)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        
        criterion = nn.BCEWithLogitsLoss().to(device)
        
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta)

        epoch_train_losses = []
        epoch_val_losses = []
        
        if args.mode == "train":
            
            for epoch in range(args.epochs):
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print("Training...")

                train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, args.threshold, args.clip_value, args.optuna, rank)
                epoch_train_losses.append(train_loss)
                
                print(f"Train Loss: {train_loss:.4f}")
                
                torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name}_last.pth")
                
                scheduler.step()
                print("Validation...")

                val_loss, targets , predictions, _ = validate(model, val_loader, criterion, device, "validation", args.save_dir, args.model_name, args.threshold, args.optuna, rank)
                epoch_val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
        cross_train_losses.append(epoch_train_losses)
        cross_val_losses.append(epoch_val_losses)
                       
        
        print("Testing... \n")
        
        X_test_standardized = standardize_skeleton(X_test)
        test_loader = create_dataloader(X_test_standardized, y_test, args.batch_size, False, args.model_name)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, "test", args.pretrained, rank)
        # model.to(device)
        model.to(rank)
        model = DDP(model, device_ids=[rank])
        
        _, targets , predictions, sigmoid_output = validate(model, test_loader, criterion, device, "testing", args.save_dir, args.model_name, args.threshold, optuna=False)
        
        auc_cross = plot_auc_curves(targets, sigmoid_output, ax, i, n_splits, mean_fpr)
        auc_test.append(auc_cross)
        
        metrics_cross = metrics(targets, predictions, "test", args.save_dir, args.model_name)
        metrics_test.append(metrics_cross)
        
    # plotting_loss(np.mean(cross_train_losses, axis=0), np.mean(cross_val_losses, axis=0), i, epoch, args.model_name, args.save_dir, args.batch_size, args.learning_rate)
    metrics_evaluate( metrics_test, args.save_dir, args.model_name)
    plot_auc_test( auc_test, args.save_dir, args.model_name, mean_fpr, ax)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train an LSTM/AcT/Moment model for video classification")
    # Model parameters
    parser.add_argument("--model_name", type=str, default="LSTM", help="Name of the model")
    parser.add_argument("--input_size", type=int, default=2048, help="Input feature size")
    parser.add_argument("--hidden_size", type=int, default=512, help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_labels", type=int, default=5, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--patience", type=int, default=13, help="Early stopping patience")
    parser.add_argument("--delta", type=float, default=0.0, help="Minimum change in validation loss")
    parser.add_argument("--mode", type=str, default="train", help="Train or test mode")
    parser.add_argument("--clip_value", type=float, default=0.7, help="Gradient clipping value")
    parser.add_argument('--threshold', type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], help='List of thresholds for multi-label classification')
    parser.add_argument("--eta", type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument("--optuna", action='store_true', help="Use optuna for hyperparameter optimization")

    # Other arguments
    parser.add_argument("--data_label", type=str, default="dataset1", help="Path to the dataset")
    parser.add_argument("--data_skeletons", type=str, default="dataset2", help="Path to the dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint for testing")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the model")
    parser.add_argument("--n_device", type=int, default=0, help="Number of devices to use")

    args = parser.parse_args()
    print(args)
        
    args.save_dir = f"{args.save_dir}/{args.model_name}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.optuna : 
        # study = optuna.create_study(directions=["minimize", "maximize"])
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=40, n_jobs=1)  
            
        all_trials = os.path.join(args.save_dir, f"{args.model_name}_all_trials.csv")
        df = study.trials_dataframe()
        df.to_csv(all_trials)
        
        best_params = study.best_trial.params
        print("Best parameters: ", best_params)
        best_params_file = os.path.join(args.save_dir, f"{args.model_name}_best_params.txt")
        with open(best_params_file, "w") as f:
            for key, value in best_params.items():
                f.write(f"{key}: {value}\n")
                setattr(args, key, value)    
                
        for key, value in best_params.items():
            if key.startswith('threshold_label_'):
                index = int(key.split('_')[-1]) - 1  # Extract the index and convert to zero-based
                args.threshold[index] = value
            else:
                setattr(args, key, value)
                
    print("Arguments: ", args)
    args.save_dir = f"{args.save_dir}/{args.model_name}_bs{args.batch_size}_lr{args.learning_rate:.3e}_hs{args.hidden_size}_th{'_'.join(f'{th:.3f}' for th in args.threshold)}_eta{args.eta:.3e}_pt{args.pretrained}/"
    
    if args.mode == 'train':
        os.makedirs(args.save_dir, exist_ok=True)
        args.checkpoint = os.path.join(args.save_dir, f"{args.model_name}_best.pth")
        
    train(args)



