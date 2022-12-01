import tensorflow as tf
from beeid.augmentation import gaussian_blur, random_erasing, color_jitter, color_drop
import tensorflow_addons as tfa
import numpy as np
from functools import partial
import pandas as pd
from sklearn.utils import shuffle

DATASET_FILENAMES = {
    "untagged" : "/home/jchan/beeid/notebooks/cmc_experiments/data/untagged_dataset.csv",
    "untagged_augmented" : "/home/jchan/beeid/notebooks/cmc_experiments/data/untagged_dataset_augmented.csv",
    "tagged": ["/home/jchan/beeid/notebooks/cmc_experiments/data/train_unnormalized.csv",
               "/home/jchan/beeid/notebooks/cmc_experiments/data/valid_unnormalized.csv"]
} 


random_erasing_black = partial(random_erasing, method="black")

def train_valid_split_df(df, train_frac=0.8):
    labels = df.label.unique()
    train_num = int(len(labels)*train_frac)
    rand_labels = np.random.permutation(len(labels))
    train_labels = rand_labels[:train_num]
    train_df = df[df.label.isin(train_labels)]
    valid_df = df[~df.label.isin(train_labels)]
    return train_df, valid_df

@tf.function 
def load_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img

@tf.function 
def merge_angle(image, angle, diff):
    angle_plane = tf.ones((*image.shape[:-1], 1), dtype=tf.float32) * angle
    diff_plane = tf.ones((*image.shape[:-1], 1), dtype=tf.float32) * diff
    image = tf.concat([image, angle_plane, diff_plane], axis=2)
    return image

def apply_rescale(dataset, rescale_factor=1, image_size=(512, 512)):
    target_size = [image_size[0]//rescale_factor, image_size[1]//rescale_factor]
    func = lambda x: tf.image.resize(x, target_size)
    dataset = dataset.map(func, num_parallel_calls=10)
    return dataset

# def get_abdomen(image):
#     return image[272:496, 144:368,:]
def get_abdomen(image):
    if image.shape[0] == 512:
        return image[272:496, 144:368,:]
    else:
        return image[16:240, 16:240,:]


def extract_filenames_and_labels(df, censored=True, label_column="track_tag_id"):
#     dataset_path = "/mnt/storage/work/jchan/normalized_uncensored_dataset/images/"
        
    file_path = list()
    labels = list()
    ids_list = list(df[label_column].unique())
    for i, row in df.iterrows():
        filename = row["filename"]
        y = ids_list.index(row[label_column])
        file_path.append(filename)
        labels.append(y)
    return file_path, labels

def load_tf_dataset(df, rescale_factor=1, augmentation=False, censored=True, label_column="track_tag_id"):
    
    filenames, labels = extract_filenames_and_labels(df, censored=censored, label_column=label_column)
#     classes = len(df.track_tag_id.unique())
        
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
#     labels = tf.one_hot(labels, classes)
    labels = tf.data.Dataset.from_tensor_slices(labels)
    
    images = filenames.map(load_image, num_parallel_calls=10)
    images = images.map(get_abdomen, num_parallel_calls=10)
    images = apply_rescale(images, rescale_factor=rescale_factor, image_size=(224, 224))
    
    if augmentation:
        images = images.map(gaussian_blur, num_parallel_calls=10)
        images = images.map(color_jitter, num_parallel_calls=10)
        images = images.map(color_drop, num_parallel_calls=10)
        images = images.map(random_erasing, num_parallel_calls=10)
        
    dataset = tf.data.Dataset.zip((images, labels))
    return dataset

def extract_filenames_and_labels_pairs(df, censored=True, label_column="track_tag_id"):        
    file_path_x = list()
    file_path_y = list()
    labels = list()
    ids_list = list(df[label_column].unique())
    for i, row in df.iterrows():
        filename_x = row["filename_x"]
        filename_y = row["filename_y"]
        y = ids_list.index(row[label_column])
        file_path_x.append(filename_x)
        file_path_y.append(filename_y)
        labels.append(y)
    return file_path_x, file_path_y, labels

def extract_filenames_and_labels_pairs_angle(df, censored=True, label_column="track_tag_id"):        
    file_path_x = list()
    file_path_y = list()
    angles = list()
    diffs = list()
    labels = list()
    ids_list = list(df[label_column].unique())
    for i, row in df.iterrows():
        filename_x = row["filename_x"]
        filename_y = row["filename_y"]
        y = ids_list.index(row[label_column])
        file_path_x.append(filename_x)
        file_path_y.append(filename_y)
        labels.append(y)
        angles.append(row["angle_y"] / 360.)
        diffs.append(row["diff_norm"])
    return file_path_x, angles, diffs, file_path_y, labels


def load_tf_track_dataset(df, track_len=5, rescale_factor=1, image_augmentation=False, augmentation=False, censored=True, label_column="track_tag_id"):
    
    filenames, labels = extract_filenames_and_labels(df, censored=censored, label_column=label_column)
#     classes = len(df.track_tag_id.unique())
        
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    labels = tf.data.Dataset.from_tensor_slices(labels[::track_len])
    
    images = filenames.map(load_image, num_parallel_calls=10)
    images = images.map(get_abdomen, num_parallel_calls=10)
    images = apply_rescale(images, rescale_factor=rescale_factor, image_size=(224, 224))
    
    if image_augmentation:
        images = images.map(gaussian_blur, num_parallel_calls=10)
        images = images.map(color_jitter, num_parallel_calls=10)
        images = images.map(color_drop, num_parallel_calls=10)
        images = images.map(random_erasing, num_parallel_calls=10)
        
    track_images = images.batch(track_len)
    
    if augmentation:
        track_images = track_images.map(track_gaussian_blur, num_parallel_calls=10)
        track_images = track_images.map(track_color_jitter, num_parallel_calls=10)
        track_images = track_images.map(track_color_drop, num_parallel_calls=10)
        track_images = track_images.map(track_random_erasing, num_parallel_calls=10)
        
    dataset = tf.data.Dataset.zip((track_images, labels))
    return dataset


def load_tf_pair_dataset(df, rescale_factor=1, augmentation=False, censored=True, label_column="track_tag_id"):
    
    filenames_x, filenames_y, labels = extract_filenames_and_labels_pairs(df, censored=censored, label_column=label_column)
        
    filenames_x = tf.data.Dataset.from_tensor_slices(filenames_x)
    filenames_y = tf.data.Dataset.from_tensor_slices(filenames_y)
    
    labels = tf.data.Dataset.from_tensor_slices(labels)
    
    images_x = filenames_x.map(load_image, num_parallel_calls=10)
    images_x = images_x.map(get_abdomen, num_parallel_calls=10)
    images_x = apply_rescale(images_x, rescale_factor=rescale_factor, image_size=(224, 224))
    
    images_y = filenames_y.map(load_image, num_parallel_calls=10)
    images_y = images_y.map(get_abdomen, num_parallel_calls=10)
    images_y = apply_rescale(images_y, rescale_factor=rescale_factor, image_size=(224, 224))
    
    if augmentation:
        images_x = images_x.map(gaussian_blur, num_parallel_calls=10)
        images_x = images_x.map(color_jitter, num_parallel_calls=10)
        images_x = images_x.map(color_drop, num_parallel_calls=10)
        images_x = images_x.map(random_erasing, num_parallel_calls=10)
        
        images_y = images_y.map(gaussian_blur, num_parallel_calls=10)
        images_y = images_y.map(color_jitter, num_parallel_calls=10)
        images_y = images_y.map(color_drop, num_parallel_calls=10)
        images_y = images_y.map(random_erasing, num_parallel_calls=10)
        
    dataset = tf.data.Dataset.zip((images_x, images_y, labels))
    return dataset



@tf.function 
def gaussian_blur(image, p=0.5, sigma_min=0.5, sigma_max=1.75):
    prob = np.random.random_sample()
    if prob < p:
        sigma = (sigma_max - sigma_min) * np.random.random_sample() + sigma_min
        image = tfa.image.gaussian_filter2d(image, (5, 5), sigma)
    return image

def filename2image(filenames, rescale_factor=4, image_size=(224, 224)):
    filenames = tf.data.Dataset.from_tensor_slices(filenames)
    images = filenames.map(load_image, num_parallel_calls=10)
    images = images.map(get_abdomen, num_parallel_calls=10)
    images = apply_rescale(images, rescale_factor=rescale_factor, image_size=image_size)
    return images


@tf.function 
def track_gaussian_blur(images, p=0.5, sigma_min=0.5, sigma_max=1.75):
    prob = np.random.random_sample()
    if prob < p:
        sigma = (sigma_max - sigma_min) * np.random.random_sample() + sigma_min
        images = tfa.image.gaussian_filter2d(images, (5, 5), sigma)
    return images


def track_color_jitter(images, p=0.5, s=0.1):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        images = tf.image.random_brightness(images, max_delta=0.8 * s)
        images = tf.image.random_contrast(images, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        images = tf.image.random_saturation(images, lower=1 - 0.8 * s, upper=1 + 0.8 * s)
        images = tf.image.random_hue(images, max_delta=0.2 * s)
        images = tf.clip_by_value(images, 0, 1)
    return images

def track_color_drop(images, p=0.2):
    rand_ = tf.random.uniform(shape=(), minval=0, maxval=1)
    if rand_ < p:
        images = tf.image.rgb_to_grayscale(images)
        images = tf.tile(images, [1, 1, 1, 3])
    return images

def track_random_erasing(images):
    images = tf.map_fn(random_erasing_black, images)
    return images


def get_dataset(dataset, augmentation=False):
    
    if dataset == "untagged" or dataset == "untagged_augmented":
        label_column = "label"
        dataset_filename = DATASET_FILENAMES[dataset]
        untagged_df = pd.read_csv(dataset_filename)
        train_df, valid_df = train_valid_split_df(untagged_df, train_frac=0.8)
        train_df = prepare_for_triplet_loss(train_df)
        valid_df = prepare_for_triplet_loss(valid_df)
        
    elif dataset == "tagged":
        label_column = "track_tag_id"
        train_csv, valid_csv = DATASET_FILENAMES[dataset]
        train_df = pd.read_csv(train_csv)
        train_df = prepare_for_triplet_loss(train_df, label=label_column)
        valid_df = pd.read_csv(valid_csv)
        valid_df = prepare_for_triplet_loss(valid_df, label=label_column)
        
    
    train_dataset, valid_dataset = load_dataset(train_df, valid_df, augmentation=augmentation, label_column="label", shuffle=False)
    return train_dataset, valid_dataset
        

def load_dataset(train_df, valid_df, augmentation=False, label_column="label", shuffle=True):
    
    train_dataset = load_tf_dataset(train_df, rescale_factor=4, augmentation=augmentation, label_column=label_column)
    valid_dataset = load_tf_dataset(valid_df, rescale_factor=4, label_column=label_column)
    
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))
    
    return train_dataset, valid_dataset


def prepare_for_triplet_loss(df, label="label"):
    sdf = df.sort_values(label)

    labels = sdf[label].values
    filename = sdf.filename.values
    
    if labels.shape[0] % 2:
        labels = labels[1:]
        filename = filename[1:]
        

    pair_labels = labels.reshape((-1, 2))
    pair_filename = filename.reshape((-1, 2))

    ridx = np.random.permutation(pair_labels.shape[0])

    labels = pair_labels[ridx].ravel()
    filename = pair_filename[ridx].ravel()

    tdf = pd.DataFrame({"filename":filename, "label":labels})
    return tdf

def prepare_for_triplet_loss_track(df, track_len=4,  repeats=10, label="label"):
    pairs = list()
    pair_labels = list()

    for i in range(repeats):

        ids = df[label].unique()
        shuffle(ids)

        A_df = df.groupby(label).sample(track_len, replace=True)
        A_df = A_df.set_index(label).loc[ids].reset_index()

        B_df = df.groupby(label).sample(track_len, replace=True)
        B_df = B_df.set_index(label).loc[ids].reset_index()

        A = A_df.filename.values
        B = B_df.filename.values
        A_label = A_df[label].values
        B_label = B_df[label].values

        pdf = np.hstack((A.reshape(-1, 1, track_len), B.reshape(-1, 1, track_len)))
        labels = np.dstack((A_label, B_label))
        
        pairs.append(pdf)
        pair_labels.append(labels)

    pair_df = np.vstack(pairs)
    pair_labels = np.vstack(pair_labels)
    df = pd.DataFrame({"filename": pair_df.ravel(), "label": pair_labels.ravel()})
    return df


def get_track_dataset(dataset, augmentation=False, track_len=4):
    
    if dataset == "untagged":
        label_column = "label"
        dataset_filename = DATASET_FILENAMES[dataset]
        untagged_df = pd.read_csv(dataset_filename)
        train_df, valid_df = train_valid_split_df(untagged_df, train_frac=0.8)
        train_df = prepare_for_triplet_loss_track(train_df, track_len=track_len, label=label_column)
        valid_df = prepare_for_triplet_loss_track(valid_df, track_len=track_len, label=label_column)
        
    elif dataset == "tagged":
        label_column = "track_tag_id"
        train_csv, valid_csv = DATASET_FILENAMES[dataset]
        train_df = pd.read_csv(train_csv)
        train_df = prepare_for_triplet_loss_track(train_df, label=label_column, track_len=track_len)
        valid_df = pd.read_csv(valid_csv)
        valid_df = prepare_for_triplet_loss_track(valid_df, label=label_column, track_len=track_len)
        
    
    train_dataset, valid_dataset = load_dataset_track(train_df, valid_df, track_len=track_len, augmentation=augmentation,  label_column="label", shuffle=False)
    return train_dataset, valid_dataset

def load_dataset_track(train_df, valid_df, track_len=4, augmentation=False, label_column="label", shuffle=True):
    
    train_dataset = load_tf_track_dataset(train_df, rescale_factor=4, track_len=track_len, augmentation=augmentation, label_column=label_column)
    valid_dataset = load_tf_track_dataset(valid_df, rescale_factor=4, track_len=track_len, label_column=label_column)
    
    if shuffle:
        train_dataset = train_dataset.shuffle(len(train_df))
        valid_dataset = valid_dataset.shuffle(len(valid_df))
    
    return train_dataset, valid_dataset


def load_tf_pair_dataset_with_yangle(df, rescale_factor=1, augmentation=False, censored=True, label_column="track_tag_id"):
    
    filenames_x, angles_y, diff_y, filenames_y, labels = extract_filenames_and_labels_pairs_angle(df, censored=censored, label_column=label_column)
        
    filenames_x = tf.data.Dataset.from_tensor_slices(filenames_x)
    filenames_y = tf.data.Dataset.from_tensor_slices(filenames_y)
    angles_y = tf.data.Dataset.from_tensor_slices(angles_y)
    diff_y = tf.data.Dataset.from_tensor_slices(diff_y)
    
    labels = tf.data.Dataset.from_tensor_slices(labels)
    
    images_x = filenames_x.map(load_image, num_parallel_calls=10)
    images_x = images_x.map(get_abdomen, num_parallel_calls=10)
    images_x = apply_rescale(images_x, rescale_factor=rescale_factor, image_size=(224, 224))
    images_x_angle = tf.data.Dataset.zip((images_x, angles_y, diff_y))
    images_x = images_x_angle.map(merge_angle, num_parallel_calls=10)
    
    images_y = filenames_y.map(load_image, num_parallel_calls=10)
    images_y = images_y.map(get_abdomen, num_parallel_calls=10)
    images_y = apply_rescale(images_y, rescale_factor=rescale_factor, image_size=(224, 224))
    
    if augmentation:
        images_x = images_x.map(gaussian_blur, num_parallel_calls=10)
        images_x = images_x.map(color_jitter, num_parallel_calls=10)
        images_x = images_x.map(color_drop, num_parallel_calls=10)
        images_x = images_x.map(random_erasing, num_parallel_calls=10)
        
        images_y = images_y.map(gaussian_blur, num_parallel_calls=10)
        images_y = images_y.map(color_jitter, num_parallel_calls=10)
        images_y = images_y.map(color_drop, num_parallel_calls=10)
        images_y = images_y.map(random_erasing, num_parallel_calls=10)
        
    dataset = tf.data.Dataset.zip((images_x, images_y, labels))
    return dataset