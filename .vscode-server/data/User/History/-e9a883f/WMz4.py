import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
from create_model import create_model
from utils import EarlyStopping, plotting_loss, create_dataloader, load_data, save_best_model_callback
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from engine import training, validate


def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e-3, log=True)
    hidden_size = trial.suggest_int("hidden_size", 1,1)
    num_layers = trial.suggest_int("num_layers", 1,1)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128])
    epochs = trial.suggest_int("epochs", 100, 200, step =50)
    patience = trial.suggest_int("patience", 15, 20, step=5)
    delta = trial.suggest_float("delta", 0.0, 0.0)
    clip_value = trial.suggest_float("clip_value", 0.2, 1.0, step=0.1)
    eta = trial.suggest_float("eta", 0.0001, 0.001)
    threshold = [trial.suggest_float(f"threshold_label_{i+1}", 0.1, 0.5, step=0.05) for i in range(6)]

    # Initialize model
    device = torch.device(f'cuda:{args.n_device}' if torch.cuda.is_available() else 'cpu')
    
    X, y = load_data(args.data_label, args.data_skeletons, args.model_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)
    
    X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=12)
        
    
    train_loader = create_dataloader(X_t, y_t, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)
    
    model = create_model(args.model_name, args.input_size, hidden_size, num_layers, args.num_labels, dropout, args.checkpoint, args.mode, args.pretrained)
    model.to(device)
    trial.set_user_attr("model", model)
    
    y_t = np.array(y_t)
    pos_weight = (len(y_t) - y_t.sum(axis=0)) / y_t.sum(axis=0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    # criterion = nn.BCEWithLogitsLoss().to(device)
    early_stopping = EarlyStopping(patience=patience, model_name=args.model_name, learning_rate=learning_rate, batch_size=batch_size, output_dir=args.save_dir, delta=delta, optuna=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta)


    # Training loop
    for epoch in range(epochs):
        train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, threshold, clip_value, args.optuna)
        val_loss, f1_score = validate(model, val_loader, criterion, device, "validation", args.save_dir, args.model_name, threshold, args.optuna)

        early_stopping(val_loss, model, 0, epoch)
        if early_stopping.early_stop:
            break

        scheduler.step()

    return val_loss


def train(args):
    
    args.optuna = False
    device = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')

    
    X, y = load_data(args.data_label, args.data_skeletons, args.model_name)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=10)
    
    for i in range(1):
        
        print(f"Fold {i}")
        
        checkpoint = os.path.join(args.save_dir, f"{args.model_name}_0_best.pth")
        X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=12)
        
        train_loader = create_dataloader(X_t, y_t, args.batch_size)
        val_loader = create_dataloader(X_val, y_val, args.batch_size)

        early_stopping = EarlyStopping(patience=args.patience, model_name=args.model_name, learning_rate=args.learning_rate,batch_size = args.batch_size, output_dir=args.save_dir ,verbose=True, delta=args.delta)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, checkpoint, args.mode, args.pretrained)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]) 
        model.to(device)
        
        y_t = np.array(y_t)
        pos_weight = (len(y_t) - y_t.sum(axis=0)) / y_t.sum(axis=0)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device)).to(device)
        # criterion = nn.BCEWithLogitsLoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.eta)

        epoch_train_losses = []
        epoch_val_losses = []
        
        if args.mode == "train":
            
            # Training loop
            for epoch in range(args.epochs):
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print("Training...")

                train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, args.threshold, args.clip_value, args.optuna)
                epoch_train_losses.append(train_loss)
                
                print(f"Train Loss: {train_loss:.4f}")
                
                torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name}_{i}_last.pth")
                
                scheduler.step()
                print("Validation...")

                val_loss, targets , predictions = validate(model, val_loader, criterion, device, "validation", args.save_dir, args.model_name, args.threshold, args.optuna)
                epoch_val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
            
                early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            plotting_loss(epoch_train_losses, epoch_val_losses, i, epoch, args.model_name, args.save_dir, args.batch_size, args.learning_rate)
        
        print("Testing... \n")
        
        test_loader = create_dataloader(X_test, y_test, args.batch_size)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, checkpoint, "test", args.pretrained)
        model.to(device)
        _, targets , predictions = validate(model, test_loader, criterion, device, "testing", args.save_dir, args.model_name, args.threshold, optuna=False)  
 

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
    parser.add_argument("--gamma", type=float, default=0.2, help="regularization LR parameter")
    parser.add_argument("--clip_value", type=float, default=0.7, help="Gradient clipping value")
    parser.add_argument('--threshold', type=float, nargs='+', default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], help='List of thresholds for multi-label classification')
    parser.add_argument("--eta", type=float, default=0.001, help="Minimum learning rate")
    parser.add_argument("--step_size", type=int, default=10, help="Step size for learning rate scheduler")
    parser.add_argument("--optuna", action='store_true', help="Use optuna for hyperparameter optimization")

    # Other arguments
    parser.add_argument("--data_label", type=str, default="dataset1", help="Path to the dataset")
    parser.add_argument("--data_skeletons", type=str, default="dataset2", help="Path to the dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None, nargs='+', help="Path to the model checkpoint for testing")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the model")
    parser.add_argument("--n_device", type=int, default=0, help="Number of devices to use")

    args = parser.parse_args()
    print(args)
    args.checkpoint = None
        
    args.save_dir = f"{args.save_dir}/{args.model_name}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.optuna : 
        # study = optuna.create_study(directions=["minimize", "maximize"])
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50, n_jobs=1, callbacks=[save_best_model_callback])  
            
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
    args.save_dir = f"{args.save_dir}/{args.model_name}_bs{args.batch_size}_lr{args.learning_rate:.3e}_hs{args.hidden_size}_gm{args.gamma:.3f}_th{'_'.join(f'{th:.3f}' for th in args.threshold)}_ss{args.step_size}_eta{args.eta:.3e}/"
    
    if args.mode == 'train':
        os.makedirs(args.save_dir, exist_ok=True)
        
    train(args)



