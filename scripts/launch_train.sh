# CUDA_VISIBLE_DEVICES=1 python train.py --dataset untagged --augmentation
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset tagged --augmentation
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset untagged
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset tagged


CUDA_VISIBLE_DEVICES=1 python train.py --dataset tagged --augmentation --model_folder ../models4/2203292136_untagged_augmentataion_simplecnnv2_convb3_dim_128
CUDA_VISIBLE_DEVICES=1 python train.py --dataset tagged --model_folder ../models4/2203300150_untagged_simplecnnv2_convb3_dim_128



# MODEL_FOLDER=/home/jchan/beeid/notebooks/cmc_experiments/models2/


# CUDA_VISIBLE_DEVICES=1 python train.py --dataset untagged_augmented --augmentation
# CUDA_VISIBLE_DEVICES=1 python train.py --dataset untagged_augmented
