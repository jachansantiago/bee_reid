import json
import numpy as np
import tensorflow as tf


def read_json(path):
    with open(path, 'r') as fp:
        data = json.load(fp)
    return data

def save_json(path, data):
    with open(path, 'w') as outfile:
        json.dump(data, outfile, indent=4)
  
def model_memory_usage(model, *, batch_size: int):
    """
    Return the estimated memory usage of a given Keras model in bytes.
    This includes the model weights and layers, but excludes the dataset.

    The model shapes are multipled by the batch size, but the weights are not.

    Args:
        model: A Keras model.
        batch_size: The batch size you intend to run the model with. If you
            have already specified the batch size in the model itself, then
            pass `1` as the argument here.
    Returns:
        An estimate of the Keras model's memory usage in bytes.

    """
    default_dtype = tf.keras.backend.floatx()
    shapes_mem_count = 0
    internal_model_mem_count = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            internal_model_mem_count += model_memory_usage(
                layer, batch_size=batch_size
            )
        single_layer_mem = tf.as_dtype(layer.dtype or default_dtype).size
        print(layer.name)
        out_shape = layer.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        for s in out_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.trainable_weights]
    )
    non_trainable_count = sum(
        [tf.keras.backend.count_params(p) for p in model.non_trainable_weights]
    )

    total_memory = (
        batch_size * shapes_mem_count
        + internal_model_mem_count
        + trainable_count
        + non_trainable_count
    )
    return np.round(total_memory / (1024.0 ** 3), 3)


def sensitivity_map(model, sample, occlude_size=16, occlude_step=2):
    image_y, image_x, _ = sample.shape
    half_width = occlude_size//2
    occluded_images = []

    for y in range(half_width, image_y-half_width, occlude_step):
        for x in range(half_width, image_x-half_width, occlude_step):
            image = sample.copy()
            image = occlude_box(image,x,y,half_width,0.5)
            occluded_images.append(image)
    occluded_images = np.array(images)
    
    coordinates = [
        (y, x)
        for y in range(half_width, image_y-half_width, occlude_step)
        for x in range(half_width, image_x-half_width, occlude_step)
    ]
    
    predictions = model.predict(np.array(images), batch_size=32).astype(np.float64)
    original_embedding = model.predict(np.array([sample])).astype(np.float64)

    distances = np.matmul(predictions, original_embedding.T).flatten()

    sensitivities = np.zeros((image_y, image_x)) 

    for (y, x), s in zip(coordinates, distances):
        sensitivities[y-occlude_step//2:y+occlude_step//2, x-occlude_step//2:x+occlude_step//2] = 1-s
    sensitivities = np.where(sensitivities==0, np.min(sensitivities), sensitivities)
    sensitivities -= np.min(sensitivities)
    
    return sensitivities.reshape(image.shape[0:2])
    