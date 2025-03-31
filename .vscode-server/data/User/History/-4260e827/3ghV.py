import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import os
from create_model import create_model
from utils.utils import EarlyStopping, plotting_loss, create_dataloader, load_data, metrics, plot_auc_curves, plot_auc_test, metrics_evaluate, load_data_video, load_pseudo_label
import numpy as np
from sklearn.model_selection import train_test_split
import optuna
from engine import training, validate
from utils.pseudo_labels import pseudo_label
from utils.visualization import visualization
import torch.cuda.amp
import torch.backends.cudnn as cudnn
import pickle
from tqdm import tqdm


def objective(trial):

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-3, step=0.0001)
    hidden_size = 192
    num_layers = 1
    dropout = trial.suggest_float("dropout", 0.1, 0.4)
    batch_size = trial.suggest_categorical("batch_size", [16])
    epochs = 250
    patience = 20
    delta = 0
    clip_value = trial.suggest_float("clip_value", 0.5, 1.0, step=0.1)
    eta = trial.suggest_float("eta", 1e-6, 1e-5,)
    threshold = [0.5, 0.5 , 0.5, 0.5, 0.5, 0.5]
    val_avg = []
    
    val_min = 1000
    # Initialize model
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    
    data = load_data(args.data_label, args.data_skeletons, args.model_name, args.save_dir)
    
    for i , (X_t, y_t, X_test, y_test) in enumerate (data):

        print(f"Fold {i}")
        
        if i > 2:
            break
        
        X_t, X_val, y_t, y_val = train_test_split(X_t, y_t, test_size=0.1, random_state=10)
        # X_train_standardized = standardize_skeleton(X_t)
        # X_val_standardized = standardize_skeleton(X_val)
        X_train_standardized = X_t
        X_val_standardized = X_val
            
        train_loader = create_dataloader(X_train_standardized, y_t, batch_size, True, args.model_name)
        val_loader = create_dataloader(X_val_standardized, y_val, batch_size, False, args.model_name)
        
        model = create_model(args.model_name, args.input_size, hidden_size, num_layers, args.num_labels, dropout, args.checkpoint, args.mode, args.pretrained, device)
        model.to(device)

        criterion = nn.BCEWithLogitsLoss().to(device)
        early_stopping = EarlyStopping(patience=patience, model_name=args.model_name, learning_rate=learning_rate, batch_size=batch_size, output_dir=args.save_dir, delta=delta, optuna=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=eta)
        scaler = torch.cuda.amp.GradScaler()
        

        for epoch in range(epochs):
            
            train_loss = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, threshold, clip_value, args.optuna, epoch)
            val_loss, f1_score = validate(model, val_loader, criterion, device, args.model_name, threshold, args.optuna)

            early_stopping(val_loss, model, 0, epoch)
            if early_stopping.early_stop:
                break

            scheduler.step()
            
            if val_min > val_loss:
                val_min = val_loss
                
        val_avg.append(val_min)
    
    val = np.mean(val_avg)
    return val


def train_I(args):
    
    args.optuna = False
    cross_train_losses, cross_val_losses, metrics_test, auc_test = [], [], [], []
    mean_fpr = np.linspace(1e-6, 1, 100)
    n_splits = args.num_patients
    mode = 'mode'
    X_t_2 , X_val_2, X_test_2 = None, None , None
    
    fold_save_dir = os.path.join(args.save_dir, f"saved_models")
    os.makedirs(fold_save_dir, exist_ok=True)
    
    # device = torch.device(f'cuda:{args.n_device}' if torch.cuda.is_available() else 'cpu')
    device = torch.device(f'cuda:2' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        cudnn.benchmark = True

    cross_val_data, index_keep = load_data(args.data_label, args.data_skeletons, args.num_patients)    
    
    if args.model_name == 'moment+dino':
        cross_val_data_2 = load_data_video(args.data_label, args.data_trials, index_keep, args.trainII, args.num_patients) 
    
    for i , (X_t, y_t, X_test, y_test) in enumerate (cross_val_data):
        
        if args.model_name == 'moment+dino':
            
            X_t_2, X_test_2 = cross_val_data_2[i]
            X_t_2, X_val_2, y_t_2, y_val_2 = train_test_split(X_t_2, y_t, test_size=args.test_size, random_state=args.random_seed)
        
        print(f"Fold {i}")
        if args.mode == "train" or mode == "train":
            
            # if i in [0, 1, 2, 3,  6, 16] and not args.saliency_map:
            #     continue
        
            X_t, X_val, y_t, y_val = train_test_split(X_t, y_t, test_size=args.test_size, random_state=args.random_seed)
            
            train_loader = create_dataloader(X_t, y_t, args.batch_size, True, args.model_name, args.trainII, X_t_2)
            val_loader = create_dataloader(X_val, y_val, args.batch_size, False, args.model_name,args.trainII, X_val_2)

            early_stopping = EarlyStopping(patience=args.patience, model_name=args.model_name, learning_rate=args.learning_rate,batch_size = args.batch_size, output_dir=fold_save_dir ,verbose=True, delta=args.delta)
            model, criterion, optimizer, scheduler, scaler = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, args.mode, args.pretrained, device, args.learning_rate, args.eta, args.epochs )
            model.to(device)
            
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            #     # if isinstance(model, torch.nn.DataParallel):
            #     #     model = model.module

            epoch_train_losses = []
            epoch_val_losses = []
                
            for epoch in tqdm(range(args.epochs), desc="Epochs"):
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print("Training...")
                
                train_loss, targets, sigmoid_output = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, args.threshold, args.clip_value, args.optuna, epoch, scaler)
                plot_auc_curves(targets, sigmoid_output, i, n_splits, mean_fpr, "train", args.first_label)
                epoch_train_losses.append(train_loss)
                
                print(f"Train Loss: {train_loss:.4f}")
                
                torch.save(model.state_dict(), f"{fold_save_dir}/{args.model_name}_last.pth")
                
                print("Validation...")

                val_loss, targets , predictions, sigmoid_output = validate(model, val_loader, criterion, device, args.model_name, args.threshold, args.optuna)
                
                plot_auc_curves(targets, sigmoid_output, i, n_splits, mean_fpr, "validation", args.first_label)
                
                scheduler.step()
                
                epoch_val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                args.checkpoint = early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
            torch.cuda.empty_cache()
            cross_train_losses.append(epoch_train_losses)
            cross_val_losses.append(epoch_val_losses)
            mode = 'test'

        elif args.mode == 'pseudo-label':
            
            if f"{args.checkpoint.rsplit('/', 1)[1][-10]}" != i:
                if i > 10 :
                    print(f"{args.checkpoint.rsplit('.', 1)[0][:-7]}")
                    args.checkpoint = f"{args.checkpoint.rsplit('.', 1)[0][:-7]}" + f"{i}_best.pth"
                else:
                    args.checkpoint = f"{args.checkpoint.rsplit('.', 1)[0][:-6]}" + f"{i}_best.pth"
                    
            print(args.checkpoint)
            test_loader = create_dataloader(X_test, y_test, args.batch_size, False, args.model_name, args.trainII, X_test_2)
            model , criterion, optimizer, scheduler, scaler = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, "test", args.pretrained, device, args.learning_rate, args.eta, args.epochs)
            model.to(device)
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
            
            pseudo_label(model, test_loader, device, args.save_dir, args.model_name, i, args.treshold_labels, args.method)

        elif args.mode == 'visualization':
            
            if f"{args.checkpoint.rsplit('/', 1)[1][-10]}" != i:
                if i > 10:
                    args.checkpoint = f"{args.checkpoint.rsplit('.', 1)[0][:-7]}" + f"{i}_best.pth"
                else:
                    args.checkpoint = f"{args.checkpoint.rsplit('.', 1)[0][:-6]}" + f"{i}_best.pth"
            
            if i in args.vis_patients:
                test_loader = create_dataloader(X_test, y_test, args.batch_size, False, args.model_name, args.trainII, X_test_2)
                model, criterion, optimizer, scheduler, scaler = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, "test", args.pretrained, device, args.learning_rate, args.eta, args.epochs )
                model.to(device)
                if torch.cuda.device_count() > 1:
                    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
                visualization(model, test_loader, device, args.model_name, i, args.method ,args.vis_trials, args.batch_size)
                
        if args.mode == 'test' or mode == 'test':    
            print("Testing... \n")
            test_loader = create_dataloader(X_test, y_test, args.batch_size, False, args.model_name, args.trainII, X_test_2)
            model, criterion, optimizer, scheduler, scaler = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, "test", args.pretrained, device,  args.learning_rate, args.eta, args.epochs )
            model.to(device)
            # if torch.cuda.device_count() > 1:
            #     model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        
            _ , targets , predictions, sigmoid_output = validate(model, test_loader, criterion, device, args.model_name, args.threshold, args.optuna)
            
        
            auc_cross = plot_auc_curves(targets, sigmoid_output, i, n_splits, mean_fpr, "testing", args.first_label)
            auc_test.append(auc_cross)
            
            metrics_cross = metrics(targets, predictions, "test", args.save_dir, args.model_name)
            metrics_test.append(metrics_cross)
            if mode == 'test':
                mode = 'train'       

    if args.mode == 'visualization' or args.mode == 'pseudo-label':
        return   
    metrics_evaluate(metrics_test, args.save_dir, args.model_name)
    plot_auc_test(auc_test, args.save_dir, args.model_name, mean_fpr, args.first_label)
        

def train_II(args):
        
    args.optuna = False
    cudnn.benchmark = True
    cross_train_losses, cross_val_losses, metrics_test, auc_test = [], [], [], []
    mean_fpr = np.linspace(1e-6, 1, 100)
    n_splits = 18    
    X_t_2 , X_val_2, X_test_2 = None, None , None
    
    if args.first_label:
        args.save_dir = os.path.join(args.save_dir, "first_label/testing")
    else:
        args.save_dir = os.path.join(args.save_dir, "all_labels")
    if args.true_labels:
        args.save_dir = os.path.join(args.save_dir, "true_labels")
    else:
        args.save_dir = os.path.join(args.save_dir, "pseudo_labels")
        
    pseudo_dir_parts = args.data_pseudo_dir.split('/')
    args.save_dir = os.path.join(args.save_dir, f"{pseudo_dir_parts[-3]}/{pseudo_dir_parts[-2]}")
    fold_save_dir = os.path.join(args.save_dir, f"saved_models")
    os.makedirs(fold_save_dir, exist_ok=True)
    
    device = torch.device(f'cuda:{args.n_device}' if torch.cuda.is_available() else 'cpu')
    
    cross_val_data = load_pseudo_label(args.data_skeletons, args.data_true_dir, args.data_pseudo_dir, args.first_label, args.true_labels, args.model_name, args.num_patients)    
    
    for i , (X_t, y_t, X_test, y_test) in enumerate (cross_val_data):
        
        if i in [ 1, 6, 16]:
            continue
        
        print('Fold:', i)   
        X_t, X_val, y_t, y_val = train_test_split(X_t, y_t, test_size=0.1, random_state=10)

        train_loader = create_dataloader(X_t, y_t, args.batch_size, True, args.model_name, args.trainII, X_t_2)
        val_loader = create_dataloader(X_val, y_val, args.batch_size, False, args.model_name,args.trainII, X_val_2)

        early_stopping = EarlyStopping(patience=args.patience, model_name=args.model_name, learning_rate=args.learning_rate,batch_size = args.batch_size, output_dir=fold_save_dir ,verbose=True, delta=args.delta)
        model, criterion, optimizer, scheduler, scaler = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, args.mode, args.pretrained, device, args.learning_rate, args.eta, args.epochs )
        model.to(device)
        # model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        
        epoch_train_losses = []
        epoch_val_losses = []
        
        if args.mode == "train":
            
            for epoch in range(args.epochs):
                
                print(f"Epoch {epoch+1}/{args.epochs}")
                print("Training...")
                
                # train_loader.sampler.set_epoch(epoch)

                train_loss, targets, sigmoid_output = training(model, train_loader, optimizer, criterion, device, args.save_dir, args.model_name, args.threshold, args.clip_value, args.optuna, epoch, scaler)
                plot_auc_curves(targets, sigmoid_output, i, n_splits, mean_fpr, "train", args.first_label)
                epoch_train_losses.append(train_loss)
                
                print(f"Train Loss: {train_loss:.4f}")
                
                # if rank == 0:
                torch.save(model.state_dict(), f"{fold_save_dir}/{args.model_name}_last.pth")
                
                print("Validation...")

                val_loss, targets , predictions, sigmoid_output = validate(model, val_loader, criterion, device, args.model_name, args.threshold, args.optuna)
                
                plot_auc_curves(targets, sigmoid_output, i, n_splits, mean_fpr, "validation", args.first_label)
                
                scheduler.step()
                
                epoch_val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.4f}")
                
                args.checkpoint = early_stopping(val_loss, model, i, epoch)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
        cross_train_losses.append(epoch_train_losses)
        cross_val_losses.append(epoch_val_losses)

        print("Testing... \n")

        test_loader = create_dataloader(X_test, y_test, args.batch_size, False, args.model_name,args.trainII, X_test_2)
        model, criterion, optimizer, scheduler, scaler = create_model(args.model_name, args.input_size, args.hidden_size, args.num_layers, args.num_labels, args.dropout, args.checkpoint, "test", args.pretrained, device, args.learning_rate, args.eta, args.epochs )
        model.to(device)
        # model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        _ , targets , predictions, sigmoid_output = validate(model, test_loader, criterion, device, args.model_name, args.threshold, args.optuna)
        
        auc_cross = plot_auc_curves(targets, sigmoid_output, i, n_splits, mean_fpr, "testing", args.first_label)
        auc_test.append(auc_cross)
    
    plot_auc_test(auc_test, args.save_dir, args.model_name, mean_fpr, args.trainII)
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train an LSTM/AcT/Moment model for video classification")
    parser.add_argument("--model_name", type=str, default="LSTM", help="Name of the model")
    parser.add_argument("--input_size", type=int, default=2048, help="Input feature size")
    parser.add_argument("--hidden_size", type=int, nargs='+', default=[128], help="Hidden layer sizes")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of LSTM layers")
    parser.add_argument("--num_labels", type=int, default=5, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--num_patients", type=int, default=18, help="Number of patients")
    
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
    parser.add_argument("--test_size", type=float, default=0.1, help="Validation set size")
    parser.add_argument("--random_seed", type=int, default=10, help="Random seed")
    parser.add_argument("--vis_patients", type=int, nargs='+', default=[0,0,0], help="Patients to visualize")

    # Other arguments
    parser.add_argument("--data_label", type=str, default="dataset1", help="Path to the dataset")
    parser.add_argument("--data_skeletons", type=str, default="dataset2", help="Path to the dataset")
    parser.add_argument("--data_trials", type=str, default="dataset3", help="Path to the dataset")
    parser.add_argument("--data_true_dir", type=str, default="dataset4", help="Path to the dataset")
    parser.add_argument("--data_pseudo_dir", type=str, default="dataset5", help="Path to the dataset")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained model")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to the model checkpoint for testing")
    parser.add_argument("--save_dir", type=str, default="checkpoints", help="Directory to save the model")
    parser.add_argument("--saliency_map", action='store_true', help="Generate saliency maps")
    parser.add_argument("--method", type=str, default="vanilla", help="Method for generating saliency maps")
    parser.add_argument("--treshold_labels", type=float,nargs='+', default='4.7', help="Threshold labels for pseudo-labels")
    parser.add_argument("--trainII", action='store_true', help="Train the second part of the model")
    parser.add_argument("--true_labels", action='store_true', help="Use true labels for training")
    parser.add_argument("--first_label", action='store_true', help="Use true labels for training")
    parser.add_argument("--n_device", type=int, default=0, help="Number of devices to use")

    args = parser.parse_args()
    print(args)
        
    args.save_dir = f"{args.save_dir}/{args.model_name}"
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.optuna : 
        # study = optuna.create_study(directions=["minimize", "maximize"])
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=20, n_jobs=1)  
            
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
    second_folder = args.data_skeletons.split('/')[1]
    first_word = second_folder.split('_')[0]
    args.save_dir = f"{args.save_dir}/{first_word}/{args.model_name}_bs{args.batch_size}_lr{args.learning_rate:.3e}_hs{args.hidden_size[0]}_th{'_'.join(f'{th:.3f}' for th in args.threshold)}_eta{args.eta:.3e}_pt{args.pretrained}/"
    
    if args.mode == 'train':
        os.makedirs(args.save_dir, exist_ok=True)

    if args.trainII:
        train_II(args)
    else:
        train_I(args)
        



