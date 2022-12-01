import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from ipywidgets import interact
import pandas as pd
from beeid2.utils import sensitivity_map
from beeid2.data_utils import filename2image
from sklearn.metrics import precision_recall_curve, auc
from skimage import io


EVALUATION_FILES = {
    "test": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_unnormalized.csv",
    "valid_with_shared_ids_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/valid_with_shared_ids_unnormalized.csv",
    "valid_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/valid_galleries_unnormalized.csv",
    "test_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_galleries_unnormalized.csv",
    "test_no_train_overlap_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_no_train_overlap_unnormalized.csv",
    "test_same_hour_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_same_hour2_unnormalized.csv",
    "test_same_hour_diff_day_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_different_day_same_hour2_unnormalized.csv",
    "test_diff_day_cmc": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_different_day2_unnormalized.csv",
    "track_test_same_hour": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_same_hour3_unnormalized.csv",
    "track_test_same_hour_diff_day": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_different_day_same_hour3_unnormalized.csv",
    "track_test_diff_day": "/home/jchan/beeid/notebooks/cmc_experiments/data/test_different_day3_unnormalized.csv",

}



def get_query_gallery(query, query_df, dfGroupedbyTagId, limit=None):
    same_tag = (query_df.track_tag_id == query.track_tag_id)
    different_global_track = (query_df.global_track_id != query.global_track_id)
    same_tag_pool = query_df[same_tag & different_global_track]
    key = same_tag_pool.sample().iloc[0]
    
#     negatives = dfGroupedbyTagId.sample()
    negatives = dfGroupedbyTagId.apply(lambda x: x.iloc[np.random.randint(len(x))])
    different_tag = (negatives.index != query.track_tag_id)
    negatives = negatives[different_tag]
    
    if limit is not None:
        negatives = negatives.sample(limit)
    query_gallery = np.concatenate(([query.filename, key.filename], negatives.filename.values))
    return query_gallery

def compute_distance(query, gallery):
    cos_dist = np.matmul(query, gallery.T)
    euclid_dist = -(cos_dist - 1)
    return euclid_dist

def compute_rank(euclid_dist):
    return np.argmin(np.argsort(euclid_dist))


def split_query_gallery(predictions):
    query = np.expand_dims(predictions[0], axis=0)
    gallery = predictions[1:]
    return query, gallery

def calculate_rank(query, query_df, vdf, gallery_size=10):
    
    query_gallery =  get_query_gallery(query, query_df, vdf, limit=gallery_size)
    
    images = filename2image(query_gallery)
    
    pred = model.predict(images.batch(32))
    
    query, gallery = split_query_gallery(pred)

    dist = compute_distance(query, gallery)
    rank = compute_rank(dist)
    
    return rank


def average_precision(vdf, gallery_size=10):

    query_df = vdf.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)

    queries = query_df.groupby("track_tag_id").sample()

    accs = list()
    
    for i, query in queries.iterrows():
        rank = calculate_rank(query, query_df, vdf, gallery_size=gallery_size)
        accs.append(np.array([rank < 1, rank < 5]))
    accs = np.array(accs)
    return np.mean(accs, axis=0)


def split_queries_galleries(predictions):
    queries_emb = list()
    galleries_emb = list()
    for q_gallery in predictions:
        query, gallery = split_query_gallery(q_gallery)
        queries_emb.append(query)
        galleries_emb.append(gallery)
    return np.array(queries_emb), np.array(galleries_emb)


def cmc_evaluation(model, df, iterations=100, gallery_size=10):
    """
    model: keras model
    df: a dataframe with the image to evaluate
    
    """
    cdf = df.copy()
    
    query_df = cdf.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    dfGroupedbyTagId = cdf.groupby("track_tag_id")
    
    ranks = np.zeros((iterations, gallery_size))

    for it in tqdm(range(iterations)):
        queries = query_df.groupby("track_tag_id").sample()
        queries_and_galleries = list()
        for i, query_data in queries.iterrows():
            query_gallery =  get_query_gallery(query_data, query_df, dfGroupedbyTagId, limit=gallery_size)
            queries_and_galleries.append(query_gallery)

        queries_and_galleries = np.array(queries_and_galleries).ravel()

        images = filename2image(queries_and_galleries)
        predictions = model.predict(images.batch(32))

        query_gallery_size = gallery_size + 2
        queries_emb = predictions[::query_gallery_size]

        pred_idx = np.arange(0, len(predictions))
        galleries_emb = predictions[np.mod(pred_idx, query_gallery_size) != 0]

        queries_emb = queries_emb.reshape(len(queries), 1, -1)
        galleries_emb = galleries_emb.reshape(len(queries), query_gallery_size - 1, -1 )

        # Calucluate distance
        cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
        euclid_dist = -(cos_dist - 1)

        # Calculate Rank
        r = np.argmin(np.argsort(euclid_dist), axis=2)
        r = np.squeeze(r)

        for i in range(gallery_size):
            ranks[it][i] = np.mean(r < (i + 1))
    return np.mean(ranks, axis=0)


def plot_cmc(ranks_means, filename=None):
    x = np.arange(1, len(ranks_means) + 1)
    plt.figure(figsize=(20, 10))
    plt.plot(x, ranks_means, 'o', markersize=8)
    plt.plot(x, ranks_means, 'b-', linewidth=2)
    plt.grid(True)
    plt.ylim(0.0, 1.0)
    plt.yticks(np.arange(0, 1.05, 0.05));
    plt.xticks(np.arange(1, len(ranks_means) + 1, 1));
    plt.xlabel("Rank")
    plt.ylabel("Matching Rate %")
    plt.tick_params(axis='y', which='minor', bottom=False)
    if filename is not None:
        plt.savefig(filename)
        
        
def plot_query_gallery(query_gallery, dist=None, limit=67):
    rows = int(np.ceil(limit/12.0))
    fig, ax = plt.subplots(rows, 12, figsize=(20, 15))
    
    ax = ax.ravel()
    
    for i, image in enumerate(query_gallery):
        ax[i].imshow(image)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        if i == 0:
            ax[i].set_title("Query")
            ax[i].spines['bottom'].set_color('red')
            ax[i].spines['top'].set_color('red') 
            ax[i].spines['right'].set_color('red')
            ax[i].spines['left'].set_color('red')
            if dist is not None:
                ax[i].set_xlabel("Rank : {}".format(compute_rank(dist)))
        elif i == 1:
            ax[i].set_title("Key")
            ax[i].spines['bottom'].set_color('green')
            ax[i].spines['top'].set_color('green') 
            ax[i].spines['right'].set_color('green')
            ax[i].spines['left'].set_color('green')
            if dist is not None:
                ax[i].set_xlabel("{:.5f}".format(dist[0][i-1]))
        else:
            ax[i].set_title("Distractor")
            if dist is not None:
                ax[i].set_xlabel("{:.9f}".format(dist[0][i-1]))
                
def plot_query_sensitivity_gallery(query_gallery, s, dist=None, limit=67):
    rows = int(np.ceil(limit/12.0))
    fig, ax = plt.subplots(rows, 12, figsize=(20, 15))
    
    ax = ax.ravel()
    
    for i, (image, smap) in enumerate(zip(query_gallery, s)):
        ax[i].imshow(image)
        ax[i].imshow(smap, alpha=0.4)
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        if i == 0:
            ax[i].set_title("Query")
            ax[i].spines['bottom'].set_color('red')
            ax[i].spines['top'].set_color('red') 
            ax[i].spines['right'].set_color('red')
            ax[i].spines['left'].set_color('red')
            if dist is not None:
                ax[i].set_xlabel("Rank : {}".format(compute_rank(dist)))
        elif i == 1:
            ax[i].set_title("Key")
            ax[i].spines['bottom'].set_color('green')
            ax[i].spines['top'].set_color('green') 
            ax[i].spines['right'].set_color('green')
            ax[i].spines['left'].set_color('green')
            if dist is not None:
                ax[i].set_xlabel("{:.5f}".format(dist[0][i-1]))
        else:
            ax[i].set_title("Distractor")
            if dist is not None:
                ax[i].set_xlabel("{:.9f}".format(dist[0][i-1]))
        

        

def get_interactive_plot_query_gallery(model, df):
    cdf = df.copy()
    query_df = cdf.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    queries_ids = query_df.track_tag_id.unique()
    queries_len = len(queries_ids)

    @interact
    def _interactive_plot_query_gallery(query_id=(0, queries_len + 1), save=False, filename=""):
        queries = query_df.groupby("track_tag_id").sample()
        query_data = queries.iloc[query_id]
        grouped = cdf.groupby("track_tag_id")

        query_gallery =  get_query_gallery(query_data, query_df, grouped, limit=10)

        images = filename2image(query_gallery)

        pred = model.predict(images.batch(32))

        query, gallery = split_query_gallery(pred)

        dist = compute_distance(query, gallery)

        plot_query_gallery(images, dist, limit=10)
        
        if save and filename != "":
            plt.tight_layout()
            plt.savefig(filename)

        s = list()
        for i in images:
            s.append(sensitivity_map(model, i.numpy(), occlude_size=8))
        plot_query_sensitivity_gallery(images, s, dist, limit=10)

        if save and filename != "":
            folder, file = os.path.split(filename)
            filename = os.path.join(folder, "sm_" + file)
            plt.tight_layout()
            plt.savefig(filename)
            
            
def cmc_evaluation_df(model, df):
    """
    model: keras model
    df: a dataframe with the image to evaluate
    
    """
    
    queries_and_galleries = df.filename.values
    images = filename2image(queries_and_galleries)
    predictions = model.predict(images.batch(32), verbose=True)

    query_gallery_size = df.image_id.max() + 1
    n_distractors = query_gallery_size - 2
    
    galleries_per_iteraration = len(df.gallery_id.unique())
    iterations = df.iteration_id.max() + 1
    total_galleries =  galleries_per_iteraration * iterations
    
    queries_emb = predictions[::query_gallery_size]
    
    
    pred_idx = np.arange(0, len(predictions))
    galleries_emb = predictions[np.mod(pred_idx, query_gallery_size) != 0]

    queries_emb = queries_emb.reshape(total_galleries, 1, -1)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, -1 )

    # Calucluate distance
    cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
    euclid_dist = -(cos_dist - 1)

    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)
    
    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))
        
    return ranks

def full_evaluation_df(model, n_distractors=10, plot=False):
    
    # valid_with_shared_ids_df = pd.read_csv(EVALUATION_FILES["valid_with_shared_ids_cmc"])
    # valid_with_shared_ids_df = valid_with_shared_ids_df[valid_with_shared_ids_df.image_id < n_distractors + 2]
    # valid_with_shared_ids_ranks = cmc_evaluation_df(model, valid_with_shared_ids_df)
    # if plot:
    #     plot_cmc(valid_with_shared_ids_ranks)
        
        
    # valid_df = pd.read_csv(EVALUATION_FILES["valid_cmc"])
    # valid_df = valid_df[valid_df.image_id < n_distractors + 2]
    # valid_ranks = cmc_evaluation_df(model, valid_df)
    # if plot:
    #     plot_cmc(valid_ranks)
        
    # test_df = pd.read_csv(EVALUATION_FILES["test_cmc"])
    # test_df = test_df[test_df.image_id < n_distractors + 2]
    # test_ranks = cmc_evaluation_df(model, test_df)
    # if plot:
    #     plot_cmc(test_ranks)
        
    # test_without_overlap_df = pd.read_csv(EVALUATION_FILES["test_no_train_overlap_cmc"])
    # test_without_overlap_df = test_without_overlap_df[test_without_overlap_df.image_id < n_distractors + 2]
    # test_without_overlap_ranks = cmc_evaluation_df(model, test_without_overlap_df)
    # if plot:
    #     plot_cmc(test_without_overlap_ranks)
    
    
    tsh_df = pd.read_csv(EVALUATION_FILES["test_same_hour_cmc"])
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_evaluation_df(model, tsh_df)
    if plot:
        plot_cmc(tsh_ranks)
        
    tddsh_df = pd.read_csv(EVALUATION_FILES["test_same_hour_diff_day_cmc"])
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_evaluation_df(model, tddsh_df)
    if plot:
        plot_cmc(tddsh_ranks)
        
    tdd_df = pd.read_csv(EVALUATION_FILES["test_diff_day_cmc"])
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_evaluation_df(model, tdd_df)
    if plot:
        plot_cmc(tdd_ranks)
        
    result_dict = {
        # "valid_with_shared_ids": valid_with_shared_ids_ranks,
        # "valid": valid_ranks,
        # "test": test_ranks,
        # "test": test_without_overlap_ranks,
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
#     result_df.to_csv(output_file)
    return result_df

def full_evaluation(output_file, model, n_distractors=10, plot=False):
    result_df = full_evaluation_df(model, n_distractors=n_distractors, plot=plot)
    result_df.to_csv(output_file, index=False)
    return result_df


def track_full_evaluation(output_file, model, n_distractors=10, plot=False, track_len=4):
    result_df = track_full_evaluation_df(model, n_distractors=n_distractors, plot=plot, track_len=track_len)
    result_df.to_csv(output_file, index=False)
    return result_df


def to_np_array(values, dim=128):
    return np.concatenate(list(values)).reshape(-1, dim)

def mAP_track_model_evaluation(track_model, track_len=5, timegap=15, timegap_unit="m", batch_size=64):
    
    # df preprocessing
    test_df = pd.read_csv(EVALUATION_FILES["test"])
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])
    
    # filter tracks by minimum len
    tracks = test_df.groupby("global_track_id").filter(lambda x: len(x) >= track_len)
    
    # sample track_len images per track
    tracks = tracks.groupby("global_track_id").sample(track_len).sort_values(["global_track_id", "datetime2"])
    
    tracks_ids = tracks["global_track_id"].values[::track_len]
    track_tag_id = tracks["track_tag_id"].values[::track_len]
    datetime = tracks["datetime2"].values[::track_len]
    filenames = tracks["filename"].values
    images = filename2image(filenames)
    predictions = track_model.predict(images.batch(track_len).batch(batch_size), verbose=True)
    tracks_emb = pd.DataFrame({"datetime": datetime, "track_tag_id":track_tag_id, "global_track_id": tracks_ids, "emb": list(predictions)})
    
    eval_tracks = len(tracks_emb)
    print("Evaluating {} tracks.".format(eval_tracks))
    
    # filtering those ids that at least have 2 tracks
    selected_track_tag_id = tracks_emb.track_tag_id.value_counts()[(tracks_emb.track_tag_id.value_counts() > 1)].index
    gtracks = tracks_emb[tracks_emb.track_tag_id.isin(selected_track_tag_id)].global_track_id.unique()
    
    APs = list()
    
    queries_num = 0
    for gtrack in gtracks:
        is_same_track = (tracks_emb.global_track_id == gtrack)
        query_row = tracks_emb[is_same_track].iloc[0]
        
        is_same_id = (query_row.track_tag_id == tracks_emb.track_tag_id)
        is_enough_timegap = np.abs(tracks_emb.datetime - query_row.datetime).astype('timedelta64[{}]'.format(timegap_unit)) > timegap
        gallery_df = tracks_emb[(is_enough_timegap & is_same_id & ~is_same_track) | ~is_same_id]
        if np.sum(gallery_df.track_tag_id == query_row.track_tag_id) == 0:
            continue
        gallery = to_np_array(gallery_df["emb"].values)
        labels = gallery_df.track_tag_id.values

        query_id = query_row.track_tag_id
        query = np.expand_dims(query_row.emb, axis=0)
        distances = tf.matmul(query, gallery.T)
        distances = np.squeeze(distances.numpy())

        binary_labels = (labels == query_id).astype(bool)

        precision, recall, thresholds = precision_recall_curve(binary_labels, distances)

        AP = auc(recall, precision)
        APs.append(AP)
        queries_num += 1
        
    print("Evaluating {} queries.".format(queries_num))
    return np.mean(APs)



def mAP_evaluation(model, timegap=15, timegap_unit="m", batch_size=256):
    test_df = pd.read_csv(EVALUATION_FILES["test"])
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])

    filenames = test_df["filename"].values
    images = filename2image(filenames)
    predictions = model.predict(images.batch(batch_size), verbose=True)
    test_df["emb"]  = list(predictions)

    gtracks = test_df.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    gtracks = gtracks.global_track_id.unique()

    APs = list()
    
    eval_tracks = len(gtracks)
    print("Evaluating {} tracks.".format(eval_tracks))
    
    queries_num = 0

    for gtrack in tqdm(gtracks):
        is_same_track = (test_df.global_track_id == gtrack)
        im_tracks = test_df[is_same_track]
        query_row = im_tracks.iloc[0]
        is_same_id = (query_row.track_tag_id == test_df.track_tag_id)
        is_enough_timegap = np.abs(test_df.datetime2 - query_row.datetime2).astype('timedelta64[{}]'.format(timegap_unit)) > timegap
        
        gallery_df = test_df[(is_enough_timegap & is_same_id & ~ is_same_track) | ~is_same_id]
        if np.sum(gallery_df.track_tag_id == query_row.track_tag_id) == 0:
            continue
            
        gallery = to_np_array(gallery_df["emb"].values)
        labels = gallery_df.track_tag_id.values
        for _, row in im_tracks.iterrows():
            query_id = row.track_tag_id
            query = np.expand_dims(row.emb, axis=0)
            distances = tf.matmul(query, gallery.T)
            distances = np.squeeze(distances.numpy())

            binary_labels = (labels == query_id).astype(bool)

            precision, recall, thresholds = precision_recall_curve(binary_labels, distances)

            AP = auc(recall, precision)
            APs.append(AP)
            queries_num += 1
    print("Evaluating {} queries.".format(queries_num))
    return np.mean(APs)


def mAP_average_full_track_evaluation(model):
    test_df = pd.read_csv(EVALUATION_FILES["test"])
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])

    filenames = test_df["filename"].values
    images = filename2image(filenames)
    predictions = model.predict(images.batch(256), verbose=True)
    test_df["emb"]  = list(predictions)

    gtracks = test_df.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    gtracks = gtracks.global_track_id.unique()

    APs = list()
    tracks_emb = test_df.groupby(["track_tag_id", "global_track_id"]).apply(lambda x: to_np_array(x.emb.values).mean(axis=0)).reset_index()
    tracks_emb.columns =  ["track_tag_id", "global_track_id", "emb"]
    for gtrack in tqdm(gtracks):
        query_row = tracks_emb[tracks_emb.global_track_id == gtrack].iloc[0]
        gallery_df = tracks_emb[tracks_emb.global_track_id != gtrack]
        gallery = to_np_array(gallery_df["emb"].values)
        labels = gallery_df.track_tag_id.values

        query_id = query_row.track_tag_id
        query = np.expand_dims(query_row.emb, axis=0)
        distances = tf.matmul(query, gallery.T)
        distances = np.squeeze(distances.numpy())

        binary_labels = (labels == query_id).astype(bool)

        precision, recall, thresholds = precision_recall_curve(binary_labels, distances)

        AP = auc(recall, precision)
        APs.append(AP)
    return np.mean(APs)


def cmc_track_model_evaluation(track_model, df, track_len=4, batch_size=64, dim=128):
    cddf = df.copy()
    cddf = cddf.sort_values(["iteration_id", "gallery_id", "image_id"])

    test_df = pd.read_csv(EVALUATION_FILES["test"])
    
    # sample track_len images per track
    tracks = test_df.groupby("global_track_id").sample(track_len, replace=True).sort_values(["global_track_id", "datetime2"])
    
    tracks_ids = tracks["global_track_id"].values[::track_len]
    track_tag_id = tracks["track_tag_id"].values[::track_len]
    datetime = tracks["datetime2"].values[::track_len]
    filenames = tracks["filename"].values
    images = filename2image(filenames)
    predictions = track_model.predict(images.batch(track_len).batch(batch_size), verbose=True)
    tracks_emb = pd.DataFrame({"global_track_id": tracks_ids , "emb": list(predictions)})

    query_gallery_size = cddf.image_id.max() + 1
    n_distractors = query_gallery_size - 2

    galleries_per_iteraration = cddf.gallery_id.max() + 1
    iterations = cddf.iteration_id.max() + 1
    total_galleries =  galleries_per_iteraration * iterations

    tracks_emb = cddf.merge(tracks_emb, how="left", on="global_track_id")
    tracks_emb = tracks_emb.sort_values(["iteration_id", "gallery_id", "image_id"])

    queries_emb = tracks_emb[tracks_emb.image_id == 0]
    galleries_emb = tracks_emb[tracks_emb.image_id != 0]

    queries_emb = to_np_array(queries_emb.emb.values, dim=dim)
    galleries_emb = to_np_array(galleries_emb.emb.values, dim)


    queries_emb = queries_emb.reshape(total_galleries, 1, dim)
    galleries_emb = galleries_emb.reshape(total_galleries, n_distractors + 1, dim)

    # Calucluate distance
    cos_dist = tf.matmul(queries_emb, galleries_emb, transpose_b=True).numpy()
    euclid_dist = -(cos_dist - 1)

    # Calculate Rank
    r = np.argmin(np.argsort(euclid_dist), axis=2)
    r = np.squeeze(r)

    ranks = np.zeros(n_distractors)
    for i in range(n_distractors):
        ranks[i] = np.mean(r < (i + 1))

    return ranks
    

def track_full_evaluation_df(model, n_distractors=10, plot=False, track_len=4):
    
    tsh_df = pd.read_csv(EVALUATION_FILES["track_test_same_hour"])
    tsh_df = tsh_df[tsh_df.image_id < n_distractors + 2]
    tsh_ranks = cmc_track_model_evaluation(model, tsh_df, track_len=track_len)
    if plot:
        plot_cmc(tsh_ranks)
        
    tddsh_df = pd.read_csv(EVALUATION_FILES["track_test_same_hour_diff_day"])
    tddsh_df = tddsh_df[tddsh_df.image_id < n_distractors + 2]
    tddsh_ranks = cmc_track_model_evaluation(model, tddsh_df, track_len=track_len)
    if plot:
        plot_cmc(tddsh_ranks)
        
    tdd_df = pd.read_csv(EVALUATION_FILES["track_test_diff_day"])
    tdd_df = tdd_df[tdd_df.image_id < n_distractors + 2]
    tdd_ranks = cmc_track_model_evaluation(model, tdd_df, track_len=track_len)
    if plot:
        plot_cmc(tdd_ranks)
        
    result_dict = {
        "test_same_hour": tsh_ranks,
        "test_different_day_same_hour": tddsh_ranks,
        "test_different_day": tdd_ranks
    }

    result_df = pd.DataFrame(result_dict)
#     result_df.to_csv(output_file)
    return result_df



def parse_track_model(filename):
    folder, model_folder = os.path.split(filename[:-1])
    config = model_folder.split("_")
    date = config[0]
    model_name = "_".join(config[-4:])
    agg = "_".join(config[-6:-4])
    dataset = "_".join(config[1:-6])
    augmentation = "augmentation" in dataset
    untagged = "_untagged_" in model_folder
    tagged = "_tagged_" in model_folder
    
    mAP_csv = os.path.join(filename + "mAP.csv")
    mAP_df = pd.read_csv(mAP_csv)
    min15, day1 = mAP_df.mAP.values
    
    
    cmc_csv = os.path.join(filename + "results.csv")
    cmc_df = pd.read_csv(cmc_csv)
    rank1_same_hour = cmc_df.loc[0, "test_same_hour"]
    rank3_same_hour = cmc_df.loc[2, "test_same_hour"]
    rank1_diff_day_same_hour = cmc_df.loc[0, "test_different_day_same_hour"]
    rank3_diff_day_same_hour = cmc_df.loc[2, "test_different_day_same_hour"]
    rank1_diff_day = cmc_df.loc[0, "test_different_day"]
    rank3_diff_day = cmc_df.loc[2, "test_different_day"]
    
    config_dict = {
        "date": date,
        "model_name": model_name,
        "agg": agg,
        "tagged": tagged,
        "untagged":untagged,
        "augmentation": augmentation,
        "dataset": dataset,
        "mAP15min": min15,
        "mAPday1":day1,
        "rank1_same_hour": rank1_same_hour,
        "rank3_same_hour": rank3_same_hour,
        "rank1_diff_day_same_hour": rank1_diff_day_same_hour,
        "rank3_diff_day_same_hour": rank3_diff_day_same_hour,
        "rank1_diff_day": rank1_diff_day,
        "rank3_diff_day": rank3_diff_day,
        "filename": filename
    }
    return config_dict

def parse_image_model(filename):
    folder, model_folder = os.path.split(filename[:-1])
    config = model_folder.split("_")
    date = config[0]
    model_name = "_".join(config[-4:])
    dataset = "_".join(config[1:-4])
    augmentation = "augmentation" in dataset
    untagged = "_untagged_" in model_folder
    tagged = "_tagged_" in model_folder
    
    # mAP_csv = os.path.join(filename + "mAP.csv")
    # mAP_df = pd.read_csv(mAP_csv)
    # min15, day1 = mAP_df.mAP.values
    
    
    cmc_csv = os.path.join(filename + "results.csv")
    cmc_df = pd.read_csv(cmc_csv)
    rank1_same_hour = cmc_df.loc[0, "test_same_hour"]
    rank3_same_hour = cmc_df.loc[2, "test_same_hour"]
    rank1_diff_day_same_hour = cmc_df.loc[0, "test_different_day_same_hour"]
    rank3_diff_day_same_hour = cmc_df.loc[2, "test_different_day_same_hour"]
    rank1_diff_day = cmc_df.loc[0, "test_different_day"]
    rank3_diff_day = cmc_df.loc[2, "test_different_day"]
    
    config_dict = {
        "date": date,
        "model_name": model_name,
        "agg": "image",
        "tagged": tagged,
        "untagged":untagged,
        "augmentation": augmentation,
        "dataset": dataset,
        # "mAP15min": min15,
        # "mAPday1":day1,
        "rank1_same_hour": rank1_same_hour,
        "rank3_same_hour": rank3_same_hour,
        "rank1_diff_day_same_hour": rank1_diff_day_same_hour,
        "rank3_diff_day_same_hour": rank3_diff_day_same_hour,
        "rank1_diff_day": rank1_diff_day,
        "rank3_diff_day": rank3_diff_day,
        "filename": filename
    }
    return config_dict

def parse_model_folder(filename):
    folder, file = os.path.split(filename)
    if "track" in file:
        return parse_track_model(filename)
    else:
        return parse_image_model(filename)

def load_benchmark_from_folder(folder, as_df=True):
    data = list()
    files = glob.glob(os.path.join(folder, "*/"))
    for file in files:
        try:
            data.append(parse_model_folder(file))
        except:
            print("Something worng with {}".format(file))
    
    if as_df:
        return pd.DataFrame(data)
    else:
        return data
        
    

def load_benchmark(folder_list):
    if type(folder_list) == str:
        return load_benchmark_from_folder(folder_list)
    elif type(folder_list) == list:
        data = list()
        for folder in folder_list:
            data += load_benchmark_from_folder(folder, as_df=False)
        return pd.DataFrame(data)

def topN_evaluation(model, N=1, timegap=15, timegap_unit="m", batch_size=256):
    test_df = pd.read_csv(EVALUATION_FILES["test"])
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])

    filenames = test_df["filename"].values
    images = filename2image(filenames)
    predictions = model.predict(images.batch(batch_size), verbose=True)
    test_df["emb"]  = list(predictions)

    gtracks = test_df.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    gtracks = gtracks.global_track_id.unique()

    ACCs = list()
    
    eval_tracks = len(gtracks)
    print("Evaluating {} tracks.".format(eval_tracks))
    
    queries_num = 0

    for gtrack in tqdm(gtracks):
        is_same_track = (test_df.global_track_id == gtrack)
        im_tracks = test_df[is_same_track]
        query_row = im_tracks.iloc[0]
        is_same_id = (query_row.track_tag_id == test_df.track_tag_id)
        is_enough_timegap = np.abs(test_df.datetime2 - query_row.datetime2).astype('timedelta64[{}]'.format(timegap_unit)) > timegap
        
        gallery_df = test_df[(is_enough_timegap & is_same_id & ~ is_same_track) | ~is_same_id]
        if np.sum(gallery_df.track_tag_id == query_row.track_tag_id) == 0:
            continue
            
#         return im_tracks, gallery_df
        gallery = to_np_array(gallery_df["emb"].values)
        labels = gallery_df.track_tag_id.values
        for _, row in im_tracks.iterrows():
            query_id = row.track_tag_id
            query = np.expand_dims(row.emb, axis=0)
            distances = tf.matmul(query, gallery.T)
            distances = np.squeeze(distances.numpy())
#             return distances, labels, query_id
            min_idx = np.argsort(distances)
            
            predicted_ids = labels[min_idx[-N:]]

            acc = np.any(predicted_ids == query_id)
            ACCs.append(acc)
            queries_num += 1
    print("Evaluating {} queries.".format(queries_num))
    return np.mean(ACCs)

def topN_track_model_evaluation(track_model, N=1, track_len=5, timegap=15, timegap_unit="m", batch_size=64):
    
    # df preprocessing
    test_df = pd.read_csv(EVALUATION_FILES["test"])
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])
    
    # filter tracks by minimum len
    tracks = test_df.groupby("global_track_id").filter(lambda x: len(x) >= track_len)
    
    # sample track_len images per track
    tracks = tracks.groupby("global_track_id").sample(track_len).sort_values(["global_track_id", "datetime2"])
    
    tracks_ids = tracks["global_track_id"].values[::track_len]
    track_tag_id = tracks["track_tag_id"].values[::track_len]
    datetime = tracks["datetime2"].values[::track_len]
    filenames = tracks["filename"].values
    images = filename2image(filenames)
    predictions = track_model.predict(images.batch(track_len).batch(batch_size), verbose=True)
    tracks_emb = pd.DataFrame({"datetime": datetime, "track_tag_id":track_tag_id, "global_track_id": tracks_ids, "emb": list(predictions)})
    
    eval_tracks = len(tracks_emb)
    print("Evaluating {} tracks.".format(eval_tracks))
    
    # filtering those ids that at least have 2 tracks
    selected_track_tag_id = tracks_emb.track_tag_id.value_counts()[(tracks_emb.track_tag_id.value_counts() > 1)].index
    gtracks = tracks_emb[tracks_emb.track_tag_id.isin(selected_track_tag_id)].global_track_id.unique()
    
    ACCs = list()
    
    queries_num = 0
    for gtrack in gtracks:
        is_same_track = (tracks_emb.global_track_id == gtrack)
        query_row = tracks_emb[is_same_track].iloc[0]
        
        is_same_id = (query_row.track_tag_id == tracks_emb.track_tag_id)
        is_enough_timegap = np.abs(tracks_emb.datetime - query_row.datetime).astype('timedelta64[{}]'.format(timegap_unit)) > timegap
        gallery_df = tracks_emb[(is_enough_timegap & is_same_id & ~is_same_track) | ~is_same_id]
        if np.sum(gallery_df.track_tag_id == query_row.track_tag_id) == 0:
            continue
        gallery = to_np_array(gallery_df["emb"].values)
        labels = gallery_df.track_tag_id.values

        query_id = query_row.track_tag_id
        query = np.expand_dims(query_row.emb, axis=0)
        distances = tf.matmul(query, gallery.T)
        distances = np.squeeze(distances.numpy())

        min_idx = np.argsort(distances)

        predicted_ids = labels[min_idx[-N:]]

        acc = np.any(predicted_ids == query_id)
        ACCs.append(acc)
        queries_num += 1
        
    print("Evaluating {} queries.".format(queries_num))
    return np.mean(ACCs)

def get_query_galleries(model, timegap=15, timegap_unit="m", batch_size=256):
    test_df = pd.read_csv("/home/jchan/beeid/notebooks/cmc_experiments/data/test_unnormalized.csv")
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])

    filenames = test_df["filename"].values
    images = filename2image(filenames)
    predictions = model.predict(images.batch(batch_size), verbose=True)
    test_df["emb"]  = list(predictions)

    gtracks = test_df.groupby("track_tag_id").filter(lambda x: len(x["global_track_id"].unique()) > 1)
    gtracks = gtracks.global_track_id.unique()

    query_galleries = list()
    
    eval_tracks = len(gtracks)
#     print("Evaluating {} tracks.".format(eval_tracks))
    
    queries_num = 0

    for gtrack in tqdm(gtracks):
        is_same_track = (test_df.global_track_id == gtrack)
        im_tracks = test_df[is_same_track]
        query_row = im_tracks.iloc[0]
        is_same_id = (query_row.track_tag_id == test_df.track_tag_id)
        is_enough_timegap = np.abs(test_df.datetime2 - query_row.datetime2).astype('timedelta64[{}]'.format(timegap_unit)) > timegap
        
        gallery_df = test_df[(is_enough_timegap & is_same_id & ~ is_same_track) | ~is_same_id]
        if np.sum(gallery_df.track_tag_id == query_row.track_tag_id) == 0:
            continue
            
        query_galleries.append((im_tracks, gallery_df))
    return query_galleries

def plot_query_gallery(query_gallery): 
    q, gallery = query_gallery
    q = q.iloc[0]
    query_id = q.track_tag_id
    query = np.expand_dims(q.emb, axis=0)
    distances = tf.matmul(query, to_np_array(gallery["emb"].values).T)
    distances = np.squeeze(distances.numpy())

    sort_indx = np.argsort(distances)
    qimage = io.imread(q.filename)
    gids = gallery.iloc[sort_indx[-10:]].track_tag_id.values
    gimages =  [io.imread(i) for i in gallery.iloc[sort_indx[-10:]].filename]

    fig, ax = plt.subplots(1, 11, figsize=(30, 20))
    ax[0].imshow(qimage)
    ax[0].set_title(f"query", fontsize=22)
    ax[0].set_xlabel(f"{query_id}", fontsize=22)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    for i in range(10):
        ax[i + 1].set_title(f"gallery", fontsize=22)
        color = "green" if gids[i] == query_id else "red"
        ax[i + 1].set_xlabel(f"{gids[i]}", fontsize=22, color=color)
        ax[i + 1].imshow(gimages[i])
        ax[i + 1].set_xticks([])
        ax[i + 1].set_yticks([])

def get_query_galleries_tracks(track_model, track_len=5, timegap=15, timegap_unit="m", batch_size=64):
    
    # df preprocessing
    test_df = pd.read_csv(EVALUATION_FILES["test"])
    test_df["datetime2"] = pd.to_datetime(test_df["datetime"])
    
    # filter tracks by minimum len
    tracks = test_df.groupby("global_track_id").filter(lambda x: len(x) >= track_len)
    
    # sample track_len images per track
    tracks = tracks.groupby("global_track_id").sample(track_len).sort_values(["global_track_id", "datetime2"])
    
    tracks_ids = tracks["global_track_id"].values[::track_len]
    track_tag_id = tracks["track_tag_id"].values[::track_len]
    datetime = tracks["datetime2"].values[::track_len]
    filenames = tracks["filename"].values
    repr_filenames = tracks["filename"].values[::track_len]
    images = filename2image(filenames)
    predictions = track_model.predict(images.batch(track_len).batch(batch_size), verbose=True)
    tracks_emb = pd.DataFrame({"datetime": datetime, "track_tag_id":track_tag_id,
                               "global_track_id": tracks_ids, "emb": list(predictions),
                              "filename": repr_filenames})
    
    eval_tracks = len(tracks_emb)
    print("Evaluating {} tracks.".format(eval_tracks))
    
    # filtering those ids that at least have 2 tracks
    selected_track_tag_id = tracks_emb.track_tag_id.value_counts()[(tracks_emb.track_tag_id.value_counts() > 1)].index
    gtracks = tracks_emb[tracks_emb.track_tag_id.isin(selected_track_tag_id)].global_track_id.unique()
    
    query_galleries = list()
    
    queries_num = 0
    for gtrack in gtracks:
        is_same_track = (tracks_emb.global_track_id == gtrack)
        query_row = tracks_emb[is_same_track].iloc[0]
        
        is_same_id = (query_row.track_tag_id == tracks_emb.track_tag_id)
        is_enough_timegap = np.abs(tracks_emb.datetime - query_row.datetime).astype('timedelta64[{}]'.format(timegap_unit)) > timegap
        gallery_df = tracks_emb[(is_enough_timegap & is_same_id & ~is_same_track) | ~is_same_id]
        if np.sum(gallery_df.track_tag_id == query_row.track_tag_id) == 0:
            continue
            
        query_galleries.append((query_row, gallery_df))
    return query_galleries


def plot_query_gallery_track(query_gallery): 
    q, gallery = query_gallery
    query_id = q.track_tag_id
    query = np.expand_dims(q.emb, axis=0)
    distances = tf.matmul(query, to_np_array(gallery["emb"].values).T)
    distances = np.squeeze(distances.numpy())

    sort_indx = np.argsort(distances)
    qimage = io.imread(q.filename)
    gids = gallery.iloc[sort_indx[-10:]].track_tag_id.values
    gimages =  [io.imread(i) for i in gallery.iloc[sort_indx[-10:]].filename]

    fig, ax = plt.subplots(1, 11, figsize=(30, 20))
    ax[0].imshow(qimage)
    ax[0].set_title(f"query", fontsize=22)
    ax[0].set_xlabel(f"{query_id}", fontsize=22)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    for i in range(10):
        ax[i + 1].set_title(f"gallery", fontsize=22)
        color = "green" if gids[i] == query_id else "red"
        ax[i + 1].set_xlabel(f"{gids[i]}", fontsize=22, color=color)
        ax[i + 1].imshow(gimages[i])
        ax[i + 1].set_xticks([])
        ax[i + 1].set_yticks([])
