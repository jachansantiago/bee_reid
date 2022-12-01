from beeid2.utils import sensitivity_map
import matplotlib.pyplot as plt
import tensorflow as tf
import io


def show_sensitivity_maps(model, dataset):
    data = dataset.shuffle(1000).batch(32)
    gen = iter(data)
    sample_batch = next(gen)

    fig, ax = plt.subplots(4, 8, figsize=(25, 8))
    ax = ax.ravel()
    for j in range(32):
        sample = sample_batch[0][j].numpy()
        ax[j].imshow(sample)
        ax[j].imshow(sensitivity_map(model, sample, occlude_size=8), alpha=0.4)
        ax[j].set_title("{}".format(sample_batch[1][j].numpy()))
        ax[j].set_xticks([])
        ax[j].set_yticks([])

        
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
        
    
def sensitivity_map_figure(model, sample_batch):
    figure, ax = plt.subplots(4, 8, figsize=(25, 8))
    ax = ax.ravel()
    for j in range(32):
        sample = sample_batch[0][j].numpy()
        ax[j].imshow(sample)
        ax[j].imshow(sensitivity_map(model, sample, occlude_size=8), alpha=0.4)
        ax[j].set_title("{}".format(sample_batch[1][j].numpy()))
        ax[j].set_xticks([])
        ax[j].set_yticks([])
    
    return figure

def log_sensitivity_map(epoch, logs, model, file_writer, sample_batch):
    
    figure = sensitivity_map_figure(model, sample_batch)
    sensitive_map_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer.as_default():
        tf.summary.image("Sensitivity Map", sensitive_map_image , step=epoch)

