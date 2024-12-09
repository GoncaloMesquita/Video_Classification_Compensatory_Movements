path_labels="dataset/labels_per_person.pt"
path_skeletons="dataset/data_skeletons_per_person.pt"

model_name="AcT"
path_save="Results/"
# export CUDA_LAUNCH_BLOCKING=1
# unset LD_LIBRARY_PATH
python3 main.py \
    --model_name $model_name \
    --input_size 99 \
    --hidden_size 192 \
    --num_layers 1 \
    --num_labels 6 \
    --dropout 0.328693634445413 \
    --batch_size 32 \
    --epochs 10 \
    --learning_rate 2.8e-07 \
    --patience 20 \
    --delta 0.0 \
    --clip_value 1 \
    --threshold 0.4 0.4 0.4 0.4 0.4 0.4 \
    --eta 0.00001 \
    --mode train \
    --data_label $path_labels \
    --data_skeletons $path_skeletons \
    --pretrained \
    --optuna\
    --save_dir $path_save \
    --n_device 0 \

