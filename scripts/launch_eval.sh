
#!/bin/zsh
MODEL_FOLDER=/home/jchan/beeid/notebooks/cmc_experiments/models3/

# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110210144_untagged_augmentataion_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110210506_tagged_augmentataion_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110210527_untagged_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110210848_tagged_simplecnnv2_convb3_dim_128


# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110211135_untagged_augmentataion_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110211658_tagged_augmentataion_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110211736_untagged_simplecnnv2_convb3_dim_128
# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110212348_tagged_simplecnnv2_convb3_dim_128


#CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110220215_tagged_untagged_simplecnnv2_convb3_dim_128
#CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110220144_tagged_augmentataion_untagged_augmentataion_simplecnnv2_convb3_dim_128


# CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}2110291337_untagged_augmentataion_mobilenetv2_1.00_224
for model in $(ls $MODEL_FOLDER); do
    if [ -f "$MODEL_FOLDER$model/topN.csv" ]; then
        echo "File Exist $MODEL_FOLDER$model/topN.csv."
    else
        CUDA_VISIBLE_DEVICES=0 python eval.py --model_folder ${MODEL_FOLDER}$model
    fi 
done
