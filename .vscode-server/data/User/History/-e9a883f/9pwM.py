import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
from create_model import create_model
from utils import EarlyStopping, plotting_loss, create_dataloader, load_data
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from engine import training, validate


def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    hidden_size = trial.suggest_int("hidden_size", 128, 512, step=128)
    num_layers = trial.suggest_int("num_layers", 1, 5)
    dropout = trial.suggest_float("dropout", 0.2, 0.5)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 264, 512])
    epochs = trial.suggest_int("epochs", 100, 200, step =50)
    patience = trial.suggest_int("patience", 10, 15, step=5)
    delta = trial.suggest_float("delta", 0.0, 0.0)
    gamma = trial.suggest_float("gamma", 0.1, 0.9)
    clip_value = trial.suggest_float("clip_value", 0.1, 1.0, step=0.2)
    threshold = trial.suggest_float("threshold", 0.1, 0.5, step=0.1)
    step_size = trial.suggest_int("step_size", 5, 20, step=5)

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X, y = load_data(args.data_label, args.data_skeletons)
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
    
    train_loader = create_dataloader(X_train, y_train, batch_size)
    val_loader = create_dataloader(X_val, y_val, batch_size)
    
    model = create_model(args.model_name, args.input_size, hidden_size, num_layers, args.num_labels, dropout, args.checkpoint, "train")
    model.to(device)

    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    early_stopping = EarlyStopping(patience=patience, model_name=args.model_name, learning_rate=learning_rate, batch_size=batch_size, output_dir=args.save_dir, delta=delta, optuna=True)

    # Training loop
    for epoch in range(epochs):
        train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, threshold, clip_value, True)
        val_loss, f1_score = validate(model, val_loader, criterion, device, "validation", args.save_dir, args.model_name, threshold, True)

        early_stopping(val_loss, model, 0, epoch)
        if early_stopping.early_stop:
            break

        scheduler.step()
        
    

    return val_loss, f1_score



def train(args):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    X, y = load_data(args.data_label, args.data_skeletons)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    checkpoint = np.zeros((1,))
    for i in range(1):
        
        print(f"Fold {i}")
        
        X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
        
        train_loader = create_dataloader(X_t, y_t, args.batch_size)
        val_loader = create_dataloader(X_val, y_val, args.batch_size)

        early_stopping = EarlyStopping(patience=args.patience, model_name=args.model_name, learning_rate=args.learning_rate,batch_size = args.batch_size, output_dir=args.save_dir ,verbose=True, delta=args.delta)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, args.mode)
        model.to(device)
        
        # Define optimizer and loss function
        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

        epoch_train_losses = []
        epoch_val_losses = []
        
        if args.mode == "train":
            # Training loop
            for epoch in range(args.epochs):
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print("Training... \n")

                train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, args.threshold, args.clip_value)
                epoch_train_losses.append(train_loss)
                print(f"Train Loss: {train_loss:.4f}")
                
                torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name}_{i}_last.pth")
                
                scheduler.step()
                print("Validation... \n")

                val_loss, targets , predictions = validate(model, val_loader, criterion, device, "validation", args.save_dir, args.model_name, args.threshold, optuna=False)
                epoch_val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
            
                early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            plotting_loss(epoch_train_losses, epoch_val_losses, i, epoch, args.model_name, args.save_dir, args.batch_size, args.learning_rate)
        
        print("Testing... \n")
        
        test_loader = create_dataloader(X_test, y_test, args.batch_size)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, f"{args.save_dir}/{args.model_name}_{i}_best.pth", "test")
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
    parser.add_argument("--gamma", type=int, default=0.2, help="regularization LR parameter")
    parser.add_argument("--clip_value", type=float, default=0.7, help="Gradient clipping value")
    parser.add_argument('--threshold', type=float, default=0.25, help='Threshold for binary classification')
    parser.add_argument("--step_size", type=int, default=10, help="Step size for learning rate scheduler")
    
    # Other arguments
    parser.add_argument("--data_label", type=str, default="dataset1", help="Path to the dataset")
    parser.add_argument("--data_skeletons", type=str, default="dataset2", help="Path to the dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None, nargs='+', help="Path to the model checkpoint for testing")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the model")
    
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = np.zeros((1,))
        
    study = optuna.create_study(direction=["minimize", "maximize"])
    study.optimize(objective, n_trials=50, n_jobs=1)  
    
    args.save_dir = f"{args.save_dir}/{args.model_name}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Save all trials
    all_trials = os.path.join(args.save_dir, f"{args.model_name}_all_trials.csv")
    df = study.trials_dataframe()
    df.to_csv(all_trials)
    
    # Save the best parameters 
    
    best_params = study.best_trial.params
    print(f"Best parameters saved to {best_params}")
    best_params_file = os.path.join(args.save_dir, f"{args.model_name}_best_params.txt")
    with open(best_params_file, "w") as f:
        for key, value in best_params.items():
            f.write(f"{key}: {value}\n")
    
    for key, value in best_params.items():
        setattr(args, key, value)
    
    print("Argumetns: ", args)
        
    args.save_dir = f"{args.save_dir}/{args.model_name}_bs{args.batch_size}_lr{args.learning_rate}_hs{args.hidden_size}_gm{args.gamma}_th{args.threshold}_ss{args.step_size}/"
    os.makedirs(args.save_dir, exist_ok=True)

    
    train(args)



