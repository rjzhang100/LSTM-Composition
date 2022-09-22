import tensorflow as tf
import numpy as np
import pretty_midi as midi
import matplotlib.pyplot as plt
import os 
from tensorflow import keras 
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import time
import pandas as pd
import model
import data_process 
import midi_utils 

#Define relevant constants
SEQ_LENGTH = 25
BATCH_SIZE = 64
NUM_PITCHES = 128 
NUM_VELOCITIES = 128

#Convert midi files to a numpy array 
notes_nparr = midi_utils.midi_to_notes("../data")
#Convert to a labelled and batched tensorflow dataset
flat_map = data_process.nparray_to_tfflatmap(notes_nparr, seq_length = SEQ_LENGTH)
train_data = data_process.label_and_batch(
                flat_map, 
                num_velocities = NUM_VELOCITIES,
                num_pitches = NUM_PITCHES,
                batch_size = BATCH_SIZE,
                shuffle_buffer_size = len(notes_nparr)
            )
#Build the network using Functional API
chad = model.build_model(
            seq_length = SEQ_LENGTH,
            num_features = notes_nparr.shape[1],
            alpha = 0.003,
            num_velocities = NUM_VELOCITIES,
            num_pitches = NUM_PITCHES
        )

#Train network and plot to show loss over epochs
history = model.train_model(chad, train_data, epochs = 100)
plt.plot(history.epoch, history.history['loss'], label = 'Total Loss')
plt.title('Total loss as a function of epochs')
plt.show()

#Predict notes using trained network
composition = midi_utils.generate_notes(
    starter_notes = notes_nparr[:SEQ_LENGTH][:],
    num_predictions = 250,
    model = chad,
    temperature = 2,
    num_velocities = NUM_VELOCITIES,
    num_pitches = NUM_PITCHES
)

#Write notes to midi
write_file = "../output/final_product.midi"
final_product = midi_utils.note_to_midi(composition, write_file, instrument_name = "Acoustic Grand Piano")



