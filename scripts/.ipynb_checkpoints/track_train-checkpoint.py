import os

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow_addons.losses import TripletSemiHardLoss, TripletHardLoss
from beeid2.models import MeanAggLayer, AttentionAggLayer
from beeid2.models import simple_cnnv2, TrackDistanceLoss, TrackModel
from beeid2.data_utils import get_track_dataset, get_dataset, DATASET_FILENAMES
from beeid2.viz import log_sensitivity_map
from functools import partial
import pandas as pd
from datetime import datetime
import argparse


LATENT_DIM=128
CONV_BLOCKS=3
SCALE_FACTOR=4
TRACK_LEN=4
MODELS_DIR = "/home/jchan/beeid/notebooks/cmc_experiments/track_models"


MARGIN=0.2
LEARNING_RATE=0.001

# Train the network
BATCH_SIZE=256
EPOCHS=1000
PATIENCE=10

def train_model(dataset, augmentation=False, model_folder=None, agg="mean"):
    
    now = datetime.now() # current date and time
    date_time = now.strftime("%y%m%d%H%M")
    
    # Load Dataset
    train_dataset, valid_dataset = get_track_dataset(dataset, augmentation=augmentation)
    _, valid_dataset_image_level = get_dataset(dataset, augmentation=augmentation)
    sample_batch = next(iter(valid_dataset_image_level.batch(32)))
    
    # Model
    if model_folder is None:
        backbone = simple_cnnv2(input_shape=(56, 56, 3), conv_blocks=CONV_BLOCKS, latent_dim=LATENT_DIM)
        
        if agg == "mean":
            agg_layer = MeanAggLayer
            model_name = "track_mean_" + backbone.name
        elif agg == "attention":
            agg_layer = AttentionAggLayer
            model_name = "track_attention_" + backbone.name
        else:
            print("Invalid aggregation method.")
            exit(1)
            
        track_model = TrackModel(backbone, model_name, agg_layer, track_len=TRACK_LEN, margin=MARGIN)
    else:
        if model_folder[-1] == "/":
            model_folder = model_folder[:-1]
        track_model = load_model(os.path.join(model_folder, "model.tf"))
        backbone = track_model.layers[0].layer
        _, model_name = os.path.split(model_folder)
        model_name = "_".join(model_name.split("_")[1:])

    # Folders
    if augmentation:
        folder_name = dataset + "_augmentataion_" + model_name
    else:
        folder_name = dataset + "_" + model_name
    
    model_folder = os.path.join(MODELS_DIR, date_time + "_" + folder_name)
    log_folder = os.path.join(model_folder, "logs")
    sm_folder = os.path.join(log_folder, "sensitivity_map")
    checkpoint_folder = os.path.join(model_folder, "checkpoints")
    metrics_filename = os.path.join(model_folder, "metrics.csv")
    checkpoint_format = os.path.join(checkpoint_folder, "{epoch:02d}-{val_loss:.2f}.hdf5")
    model_filename = os.path.join(model_folder, "model.tf")


    # Compile the model
    track_model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss=TripletSemiHardLoss(margin=MARGIN))

    # Callbacks
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    checkpoints = tf.keras.callbacks.ModelCheckpoint(checkpoint_format, mode='auto', save_freq='epoch')
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder)
    metrics_loger = tf.keras.callbacks.CSVLogger(metrics_filename, separator=',', append=False)
    file_writer = tf.summary.create_file_writer(sm_folder)
    log_sensitivity_map_func = partial(log_sensitivity_map, sample_batch=sample_batch, file_writer=file_writer, model=backbone)
    sensitivity_map_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=log_sensitivity_map_func)
    callbacks=[earlystop, checkpoints, tensorboard, metrics_loger, sensitivity_map_callback]

    # Training
    history = track_model.fit(train_dataset.batch(BATCH_SIZE), validation_data=valid_dataset.batch(BATCH_SIZE), epochs=EPOCHS, callbacks=callbacks)
    track_model.save(model_filename)
    
    
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Triplet Loss')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Choose dataset to train.')
    parser.add_argument('--augmentation', action='store_true',
                        help='Apply augmentation.')
    parser.add_argument('--model_folder', type=str, default=None,
                        help='Model Folder of pretrained model.')
    parser.add_argument('--agg', type=str, choices=("mean", "attention"),
                        help='Aggregation Method.')

    args = parser.parse_args()
    
    
    train_model(args.dataset, augmentation=args.augmentation, model_folder=args.model_folder, agg=args.agg)
    
    