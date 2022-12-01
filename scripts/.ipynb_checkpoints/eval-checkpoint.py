import argparse
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from beeid2.evaluation import full_evaluation, mAP_evaluation, topN_evaluation
from beeid2.models import simple_cnnv2
from tensorflow_addons.losses import TripletSemiHardLoss, TripletHardLoss





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Train Triplet Loss')
    parser.add_argument('--model_folder', type=str, required=True,
                        help='Model to evaluate.')
    parser.add_argument('-n', '--ndistractors', type=int, default=10,
                        help='Amount of distractors to evaluate.')

    args = parser.parse_args()

    model_file = os.path.join(args.model_folder, "model.tf")
    output_file = os.path.join(args.model_folder, "results.csv")
    mAP_file = os.path.join(args.model_folder, "mAP.csv")
    topN_file = os.path.join(args.model_folder, "topN.csv")
    
    model = load_model(model_file, custom_objects={'tf': tf})
    
    
#     full_evaluation(output_file, model, n_distractors=args.ndistractors, plot=False)
    
    mAP15 = mAP_evaluation(model, timegap=15, timegap_unit="m")
    mAP1D = mAP_evaluation(model, timegap=1, timegap_unit="D")
    
    mAP_dict = {"time_gap": ["15min", "1day"], "mAP": [mAP15, mAP1D]}
    mAP_df = pd.DataFrame(mAP_dict)
    mAP_df.to_csv(mAP_file, index=False)
    
    top1_15m = topN_evaluation(model, N=1, timegap=15, timegap_unit="m")
    top1_1D = topN_evaluation(model, N=1, timegap=1, timegap_unit="D")
    
    top3_15m = topN_evaluation(model, N=3, timegap=15, timegap_unit="m")
    top3_1D= topN_evaluation(model, N=3, timegap=1, timegap_unit="D")
    
    topN_dict = {"time_gap": ["15min", "1day", "15min", "1day"],
                 "topN": [1, 1, 3, 3], "acc": [top1_15m, top1_1D, top3_15m, top3_1D]}
    topN_df = pd.DataFrame(topN_dict)
    topN_df.to_csv(topN_file, index=False)