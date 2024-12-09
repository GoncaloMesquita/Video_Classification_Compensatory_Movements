path_labels="dataset/labels.npy"
path_labels="dataset/labels.npy"
path_skeletons="dataset/dataset.npy"

model_name="LSTM"
patch_save="Results/"

python3 main.py \
    --model_name $model_name \
    --input_size 165 \
    --hidden_size 64 \
    --num_layers 1 \
    --num_labels 6 \
    --dropout 0.5 \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.001 \
    --patience 30 \
    --mode train \
    --data_label $path_labels \
    --data_skeletons $path_skeletons \
    --save_dir $patch_save

