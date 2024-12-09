path_labels="dataset/data_labels.pt"
path_skeletons="dataset/data_skeletons.pt"


model_name="LSTM"
patch_save="Results/"

python3 main.py \
    --model_name $model_name \
    --input_size 99 \
    --hidden_size 64 \
    --num_layers 2 \
    --num_labels 6 \
    --dropout 0.5 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.001 \
    --patience 15 \
    --delta 0 \
    --mode train \
    --data_label $path_labels \
    --data_skeletons $path_skeletons \
    --save_dir $patch_save

