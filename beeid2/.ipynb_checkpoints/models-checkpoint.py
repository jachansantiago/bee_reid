import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, Sequential, load_model
import tensorflow_addons as tfa

def simple_cnn(input_shape=(400, 200, 3), conv_blocks=2, latent_dim=256, l2_norm=True):
    model = Sequential(name="simple_cnn")
    model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(Conv2D(64, (3, 3), padding="same",))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    
    
    for _ in range(conv_blocks - 1):
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
        model.add(Conv2D(64, (3, 3), activation='relu', padding="same",))
        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(latent_dim))
    if l2_norm:
        model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    return model



def simple_cnnv2(input_shape=(400, 200, 3), conv_blocks=2, latent_dim=256, l2_norm=True, dropout=True):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, (7, 7), activation='relu', padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding="same")(x)

    for _ in range(conv_blocks - 1):
        xp = MaxPooling2D((2, 2))(x)
        x = BatchNormalization()(xp)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding="same")(x)
        x = Add()([xp, x])
        

    x = Flatten()(x)
    if dropout:
        x = Dropout(0.5)(x)
    x = Dense(latent_dim)(x)
    if dropout:
        x = Dropout(0.2)(x)
    if l2_norm:
        x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    model_name = "simplecnnv2_convb{conv_blocks}_dim_{latent_dim}".format(conv_blocks=conv_blocks, latent_dim=latent_dim)
    model = Model(inputs, x, name=model_name)
    return model


def ResNet50v2(input_shape=(400, 200, 3), latent_dim=256, weights=None):
    base_model = tf.keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, weights=weights)
        
    inputs = Input(input_shape)
    h = base_model(inputs, training=True)
    h = Flatten()(h)
    h = Dropout(0.5)(h)
    projection = Dense(latent_dim)(h)
     
    projection = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(projection)

    model = Model(inputs, projection, name="ResNetV2")
    return model


class ContrastiveLearning(tf.keras.Model):
    def __init__(self, base_model, temperature=0.01):
        super(ContrastiveLearning, self).__init__()
        self.backbone = base_model
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.valid_loss_tracker = tf.keras.metrics.Mean(name="valid_loss")
        self.temperature = temperature
        self.model_name = "ConstrastiveLearning"
        
    def call(self, data):
        x = data
        x = self.backbone(x)
        return x

    def train_step(self, data):
        x1, x2, y = data
        
        with tf.GradientTape() as tape:
            x1 = self(x1, training=True)
            x2 = self(x2, training=True)
            
            sim_matrix1 = tf.matmul(x1, x2, transpose_b=True)/ self.temperature
            sim_matrix2 = tf.transpose(sim_matrix1)
            
            loss1 = tfa.losses.npairs_loss(y_pred=sim_matrix1, y_true=y)
            loss2 = tfa.losses.npairs_loss(y_pred=sim_matrix2, y_true=y)
            loss = loss1 + loss2
        
        trainable_vars = self.trainable_weights
        gradients = tape.gradient(loss, trainable_vars)
        self.loss_tracker.update_state(loss)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": self.loss_tracker.result()}
    
    def test_step(self, data):
        x1, x2, y = data
        
        x1 = self(x1, training=False)
        x2 = self(x2, training=False)
            
        sim_matrix1 = tf.matmul(x1, x2, transpose_b=True)/ self.temperature
        sim_matrix2 = tf.transpose(sim_matrix1)
            
        loss1 = tfa.losses.npairs_loss(y_pred=sim_matrix1, y_true=y)
        loss2 = tfa.losses.npairs_loss(y_pred=sim_matrix2, y_true=y)
        loss = loss1 + loss2
        
        self.valid_loss_tracker.update_state(loss)
        
        return {"loss": self.valid_loss_tracker.result()}
    
    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker, self.valid_loss_tracker]
    
    
class TrackDistanceLoss(tf.keras.layers.Layer):
    def __init__(self, margin=0.2):
        super(TrackDistanceLoss, self).__init__()
        self.margin = margin

    def call(self, inputs):
        track_distances = - tf.matmul(inputs, inputs, transpose_b=True) + 1.0
        max_dists = tf.reduce_max(track_distances, axis=[1, 2])
        track_max_dist = tf.maximum(tf.reduce_mean(max_dists), self.margin)
        self.add_loss(track_max_dist)
        return inputs
    
class VectorMeanAttentionAggLayer(Layer):
    def __init__(self):
        super(VectorMeanAttentionAggLayer, self).__init__()
        
    def build(self, input_shape):
        self.w = self.add_weight(
            name='w',
            shape=(input_shape[-1]*2, input_shape[-1]),
            initializer="random_normal",
            trainable=True,
        )
        tf.print(self.w.shape)
        
    def call(self, inputs):
        means = tf.math.reduce_mean(inputs, axis=1)
        #these next two lines just take a matrix of size
        #batch_size*latent_dim and repeat the means to form
        #a tensor of size batch_size*track_size*latent_dim
        means = tf.expand_dims(means, axis=1)
        means = tf.repeat(means, repeats=[inputs.shape[1]], axis=1)
        #now we stick the means to the original values
        full_inputs = tf.keras.layers.concatenate([inputs, means], axis=2)
        attention_weights = tf.matmul(full_inputs, self.w)
        normalized_weights = tf.nn.softmax(attention_weights, axis=1)
        x = tf.math.multiply(normalized_weights, inputs)
        return tf.math.reduce_sum(x, axis=1)

MeanAggLayer = tf.keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))
AttentionAggLayer = VectorMeanAttentionAggLayer()

def TrackModel(backbone, name, agg_layer, track_len=4, margin=2.0):
    track_model = tf.keras.Sequential(name=name)
    track_model.add(tf.keras.layers.Input(shape=(track_len, 56, 56, 3)))
    track_model.add(tf.keras.layers.TimeDistributed(backbone, input_shape=(track_len, 56, 56, 3)))
    track_model.add(TrackDistanceLoss(margin=margin))
    track_model.add(agg_layer)
    track_model.add(tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    return track_model