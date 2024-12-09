

path_labels="dataset/labels_per_person.pt"
path_skeletons="dataset/data_skeletons_per_person.pt"

model_name="AcT"
path_save="Results/"
# export CUDA_LAUNCH_BLOCKING=1
# unset LD_LIBRARY_PATH

# Run the Python script with torchrun
python -m torch.distributed.run --nproc_per_node=4 main.py  \
    --model_name $model_name \
    --input_size 99 \
    --hidden_size 192 \
    --num_layers 1 \
    --num_labels 6 \
    --dropout 0.2320088098820616 \
    --batch_size 64 \
    --epochs 250 \
    --learning_rate 8.097688329162332e-05 \
    --patience 20 \
    --delta 0.008535437945169335 \
    --clip_value 0.8 \
    --threshold 0.4 0.25 0.25 0.25 0.15000000000000002 0.4 \
    --eta 4.412418664183598e-07 \
    --mode train \
    --data_label $path_labels \
    --data_skeletons $path_skeletons \
    --pretrained \
    --save_dir $path_save \
    --n_device 0 \