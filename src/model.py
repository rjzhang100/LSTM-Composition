import tensorflow as tf 
from tensorflow import keras 
from matplotlib import pyplot as plt 

#Create an LSTM model using Keras Functional API
def build_model(seq_length: int, num_features: int, alpha: float, num_velocities: int, num_pitches: int):
    input_shape = (seq_length, num_features)
    inputs = tf.keras.Input(input_shape)
    x = tf.keras.layers.LSTM(num_pitches)(inputs)

    #Create an output dense layer per feature
    outputs = {
        'velocity': tf.keras.layers.Dense(num_velocities, name = 'velocity')(x),
        'pitch': tf.keras.layers.Dense(num_pitches, name = 'pitch')(x),
        'duration': tf.keras.layers.Dense(1, name = 'duration')(x),
        'step': tf.keras.layers.Dense(1, name = 'step')(x)
    }

    #Assign each output per feature an appropriate loss function 
    loss = {
        'velocity': tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        'pitch': tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
        'duration': tf.keras.losses.MeanSquaredLogarithmicError(),
        'step': tf.keras.losses.MeanSquaredError()
    }

    model = tf.keras.Model(inputs, outputs, name = 'Chad')
    model.compile(
        loss = loss, 
        loss_weights = {
            'velocity': 1,
            'pitch': 1,
            'duration': 0.5,
            'step': 1
        },
        optimizer = tf.keras.optimizers.Adam(learning_rate = alpha),
        metrics = ['accuracy']
    )
    return model 

#Train specified model with batched data
def train_model(model, batched_train_data, epochs: int):

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath='../training_checkpoints/ckpt_{epoch}',
            save_weights_only=True),
        tf.keras.callbacks.EarlyStopping(
            monitor='loss',
            patience=5,
            verbose=1,
            restore_best_weights=True),
    ]
    history = model.fit(
        batched_train_data,
        callbacks = callbacks,
        epochs = epochs,
    )
    return history 




        
