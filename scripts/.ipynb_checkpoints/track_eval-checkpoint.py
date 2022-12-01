import argparse
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from beeid2.evaluation import track_full_evaluation, mAP_track_model_evaluation
from beeid2.models import simple_cnnv2
from tensorflow_addons.losses import TripletSemiHardLoss, TripletHardLoss
from functools import partial


def random_sampling(x, track_len=4):
    rand_idx = tf.random.uniform(shape=[], maxval=track_len, dtype=tf.int32)
    return x[:, rand_idx, :]

def image2track_model(model, track_len=4):
    random_sampling_func = partial(random_sampling, track_len=track_len)
    track_model = tf.keras.Sequential()
    track_model.add(tf.keras.layers.TimeDistributed(model, input_shape=(track_len, 56, 56, 3)))
    track_model.add(tf.keras.layers.Lambda(random_sampling_func))
    return track_model
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Evaluate Track Models')
    parser.add_argument('--model_folder', type=str, required=True,
                        help='Model to evaluate.')
    parser.add_argument('-n', '--ndistractors', type=int, default=10,
                        help='Amount of distractors to evaluate.')
    parser.add_argument('--image', action="store_true",
                        help='Load image level model.')
    parser.add_argument('--track_len', type=int, default=4,
                        help='Track Lenght.')

    args = parser.parse_args()

    model_file = os.path.join(args.model_folder, "model.tf")
    output_file = os.path.join(args.model_folder, "results.csv")
    mAP_file = os.path.join(args.model_folder, "mAP.csv")
    
    model = load_model(model_file)
    
    if args.image:
        model = image2track_model(model, args.track_len)
    
    
    track_full_evaluation(output_file, model, n_distractors=args.ndistractors, plot=False, track_len=args.track_len)
    
    mAP15 = mAP_track_model_evaluation(model, timegap=15, timegap_unit="m", track_len=args.track_len)
    mAP1D = mAP_track_model_evaluation(model, timegap=1, timegap_unit="D", track_len=args.track_len)
    
    mAP_dict = {"time_gap": ["15min", "1day"], "mAP": [mAP15, mAP1D]}
    mAP_df = pd.DataFrame(mAP_dict)
    mAP_df.to_csv(mAP_file, index=False)