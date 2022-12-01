# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset untagged --augmentation
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --augmentation
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset untagged
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged



MODEL_FOLDER=/home/jchan/beeid/notebooks/cmc_experiments/track_models/

# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --model_folder ${MODEL_FOLDER}2110260232_untagged_Track_mean_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --augmentation --model_folder ${MODEL_FOLDER}2110252243_untagged_augmentataion_Track_mean_simplecnnv2_convb3_dim_128

# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset untagged --augmentation --agg attention
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --augmentation --agg attention
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset untagged --agg attention
# CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --agg attention


CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --agg attention --model_folder ${MODEL_FOLDER}2110261639_untagged_track_attention_simplecnnv2_convb3_dim_128
CUDA_VISIBLE_DEVICES=1 python track_train.py --dataset tagged --augmentation --agg attention --model_folder ${MODEL_FOLDER}2110261353_untagged_augmentataion_track_attention_simplecnnv2_convb3_dim_128
