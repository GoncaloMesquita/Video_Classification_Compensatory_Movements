path_labels="dataset/data_labels.pt"
path_skeletons="dataset/data_skeletons.pt"

model_name="LSTM"
patch_save="Results/"

python3 main.py \
    --model_name $model_name \
    --input_size 99 \
    --hidden_size 384 \
    --num_layers 1 \
    --num_labels 6 \
    --dropout 0.328693634445413 \
    --batch_size 128 \
    --epochs 200 \
    --learning_rate 0.0015209165905619407 \
    --patience 10 \
    --delta 0.0 \
    --gamma 0.6603671689102271 \
    --clip_value 0.5 \
    --threshold 0.4 \
    --step_size 20 \
    --mode train \
    --data_label $path_labels \
    --data_skeletons $path_skeletons \
    --save_dir $patch_save

