import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
from create_model import create_model
from utils import EarlyStopping, plotting_loss, create_dataloader
import numpy as np
from sklearn.model_selection import train_test_split
from engine import training, validate

def load_data(data_dir_labels, data_dir_skeletons):
    
    ske = np.load(data_dir_skeletons)
    labels = np.load(data_dir_labels)

    return ske, labels

def train(args):
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

    args.save_dir = f"{args.save_dir}/{args.model_name}/"
    os.makedirs(args.save_dir, exist_ok=True)
    X, y = load_data(args.data_label, args.data_skeletons)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    
    checkpoint = np.zeros((1,))
    for i in range(1):
        
        print(f"Fold {i}")
        
        X_t, X_val, y_t, y_val = train_test_split(X_train, y_train, test_size=0.10, random_state=42)
        
        train_loader = create_dataloader(X_t, y_t, args.batch_size)
        val_loader = create_dataloader(X_val, y_val, args.batch_size)

        early_stopping = EarlyStopping(patience=args.patience, model_name=args.model_name, learning_rate=args.learning_rate,batch_size = args.batch_size, output_dir=args.save_dir ,verbose=True, delta=0)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, args.mode)
        model.to(device)
        
        # Define optimizer and loss function
        criterion = nn.BCEWithLogitsLoss()  
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        epoch_train_losses = []
        epoch_val_losses = []
        
        if args.mode == "train":
            # Training loop
            for epoch in range(args.epochs):
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print("Training... \n")

                train_loss = training(model, train_loader, optimizer, criterion, device)
                epoch_train_losses.append(train_loss)
                print(f"Train Loss: {train_loss:.4f}")
                
                torch.save(model.state_dict(), f"{args.save_dir}/{args.model_name}_{args.batch_size}_{i}_{args.learning_rate}_last.pth")
                
                print("Validation... \n")

                val_loss, targets , predictions = validate(model, val_loader, criterion, device, "Validation:")
                epoch_val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
            
                early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            plotting_loss(epoch_train_losses, epoch_val_losses, i, epoch, args.model_name, args.save_dir, args.batch_size, args.learning_rate)
        
        print("Testing... \n")
        
        test_loader = create_dataloader(X_test, y_test, args.batch_size)
        model = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, f"{args.save_dir}/{args.model_name}_{args.batch_size}_{i}_{args.learning_rate}_best.pth", "test")
        model.to(device)
        _, targets , predictions = validate(model, test_loader, criterion, device, "Testing:")  
 

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
    parser.add_argument("--mode", type=str, default="train", help="Train or test mode")
    
    # Other arguments
    parser.add_argument("--data_label", type=str, default="dataset1", help="Path to the dataset")
    parser.add_argument("--data_skeletons", type=str, default="dataset2", help="Path to the dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None, nargs='+', help="Path to the model checkpoint for testing")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the model")
    
    args = parser.parse_args()
    if args.checkpoint is None:
        args.checkpoint = np.zeros((1,))
        
    print(args)  
    
    train(args)
