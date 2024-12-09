path_labels="dataset/data_labels.pt"
path_skeletons="dataset/data_skeletons.pt"

model_name="AcT"
path_save="Results/"
path_chekpoint="Results/$model_name/AcT_bs64_lr7.941315653357396e-06_hs1_gm0.1_th[0.416,0.441,0.49,0.447,0.477,0.399]_ss15/AcT_0_best.pth"
# export CUDA_LAUNCH_BLOCKING=1

# unset LD_LIBRARY_PATH

python3 main.py \
    --model_name $model_name \
    --input_size 99 \
    --hidden_size 1 \
    --num_layers 1 \
    --num_labels 6 \
    --dropout 0.3215 \
    --batch_size 64 \
    --epochs 200 \
    --learning_rate 7.941315653357396e-06 \
    --patience 20 \
    --delta 0.0 \
    --gamma 0.1 \
    --clip_value 0.5 \
    --threshold 0.416 0.441 0.490 0.447 0.477 0.399 \
    --step_size 15 \
    --eta 0.000715 \
    --mode test \
    --data_label $path_labels \
    --data_skeletons $path_skeletons \
    --pretrained \
    --checkpoint $path_chekpoint \
    --save_dir $path_save \
    --n_device 0 \

