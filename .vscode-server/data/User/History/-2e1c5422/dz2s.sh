path_labels="dataset/labels.npy"
path_skeletons="dataset/dataset.npy"

model_name="LSTM"
patch_save="Results/"

python3 main.py \

    --model_name, LSTM,
    --input_size, 165,
    --hidden_size, 64,
    --num_layers, 1,
    --num_labels, 6,
    --dropout, 0.5,
    --batch_size, 32,
    --epochs, 30,
    --learning_rate, 0.001,
    --patience, 30,
    --mode, train,
    --data_label, dataset/labels.npy,
    --data_skeletons, dataset/dataset.npy, 
    --save_dir, Results/

