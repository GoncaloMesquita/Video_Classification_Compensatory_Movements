

path_labels="dataset/labels_per_person.pt"
path_skeletons="dataset/data_skeletons_per_person.pt"

model_name="moment+dino"
path_save="Results/"
path_trial="trials"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export CUDA_LAUNCH_BLOCKING=1
# unset LD_LIBRARY_PATH

# Run the Python script with torchrun
# python -m torch.distributed.run --nproc_per_node=4 main.py  \

# python3 main.py  \
#     --model_name LSTM \
#     --input_size 99 \
#     --hidden_size 192 \
#     --num_layers 1 \
#     --num_labels 6 \
#     --dropout 0.16 \
#     --batch_size 1 \
#     --epochs 100 \
#     --learning_rate 0.0001 \
#     --patience 5 \
#     --delta 0.0 \
#     --clip_value 0.9 \
#     --threshold 0.5 0.4 0.5 0.35 0.30000000000000004 0.1 \
#     --eta 0.0001 \
#     --mode pseudo-label \
#     --data_label dataset/labels_per_person.pt \
#     --data_skeletons dataset/data_skeletons_per_person.pt \
#     --data_true_dir dataset/frame_level_labels.pt \
#     --checkpoint Results/LSTM/LSTM_newdata_cross_bs16_lr6.161e-04_hs192_th0.500_0.400_0.500_0.350_0.300_0.100_eta9.548e-05_ptTrue/saved_models/LSTM_0_best.pth \
#     --save_dir Results/ \
#     --method ig \
#     --treshold_labels 2.5 3.5\
#     --n_device 0


python3 main.py  \
    --model_name MLP \
    --input_size 99 \
    --hidden_size 256 128 \
    --num_layers 1 \
    --num_labels 1 \
    --dropout 0.20 \
    --batch_size 64 \
    --epochs 100 \
    --learning_rate 0.001 \
    --patience 5 \
    --delta 0.0 \
    --clip_value 0.9 \
    --threshold 0.5 0.4 0.5 0.35 0.30000000000000004 0.1 \
    --eta 0.001 \
    --mode train \
    --data_label dataset/labels_per_person.pt \
    --data_skeletons dataset/data_skeletons_per_person.pt \
    --data_trial trials \
    --data_true_dir dataset/frame_level_labels.pt \
    --data_pseudo_dir dataset/pseudo_labels/LSTM/method2/vg_4.0_6.5/pseudo_labels_1.pkl \
    --save_dir Results/ \
    --trainII \
    --first_label \
    --n_device 1

    

# python3 main.py  \
#     --model_name moment+dino \
#     --input_size 99 \
#     --hidden_size 192 \
#     --num_layers 1 \
#     --num_labels 6 \
#     --dropout 0.16 \
#     --batch_size 16 \
#     --epochs 100 \
#     --learning_rate 0.0001 \
#     --patience 15 \
#     --delta 0.0 \
#     --clip_value 0.7 \
#     --threshold 0.5 0.4 0.5 0.35 0.30000000000000004 0.1 \
#     --eta 0.00001 \
#     --mode test \
#     --data_label dataset/labels_per_person.pt \
#     --data_skeletons dataset/data_skeletons_per_person.pt \
#     --data_trial trials \
#     --checkpoint Results/moment+dino/moment+dino_bs16_only_first/saved_models/moment+dino_0_best.pth \
#     --save_dir Results/ \
#     --n_device 0


# python3 main.py  \
#     --model_name moment+dino \
#     --input_size 99 \
#     --hidden_size 192 \
#     --num_layers 1 \
#     --num_labels 6 \
#     --dropout 0.20 \
#     --batch_size 4 \
#     --epochs 10 \
#     --learning_rate 0.001 \
#     --patience 3 \
#     --delta 0.0 \
#     --clip_value 0.7 \
#     --threshold 0.5 0.4 0.5 0.35 0.30000000000000004 0.1 \
#     --eta 0.0001 \
#     --mode train \
#     --data_label dataset/labels_per_person.pt \
#     --data_skeletons dataset/data_skeletons_per_person.pt \
#     --data_trial trials \
#     --save_dir Results/ \
#     --n_device 0