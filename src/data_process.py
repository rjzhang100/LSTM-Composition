from random import shuffle
import pandas as pd 
import numpy as np 
import tensorflow as tf 
import pretty_midi as midi 
import os 


#Convert numpy array to tensorflow flat_map of training example datasets with one extra for label
def nparray_to_tfflatmap(in_arr: np.ndarray, seq_length: int):
    seq_length = seq_length + 1 #Increment by one to grab labels
    init_dataset = tf.data.Dataset.from_tensor_slices(in_arr)
    windows = init_dataset.window(seq_length, shift = 1, stride = 1, drop_remainder = True)
    map_func = lambda x : x.batch(seq_length, drop_remainder = True) #Mapping function for flat_map() 
    return windows.flat_map(map_func)

#Map over to label and batch the flat_map of datasets
def label_and_batch(
    in_flat_map, 
    num_velocities: int, 
    num_pitches: int,
    batch_size: int,
    shuffle_buffer_size: int):
    
    #Function for label map
    def label_make(x):
        input_data = x[:-1]
        label_value = x[-1]
        label_dict = {
            key:label_value[i] for i,key in enumerate(['velocity', 'pitch', 'duration', 'step'])
        }
        input_data = input_data / [num_velocities, num_pitches, 1, 1] #Normalization
        return input_data, label_dict
    

    unbatched_train_data = in_flat_map.map(label_make, num_parallel_calls = tf.data.AUTOTUNE)
    batched_train_data = (unbatched_train_data
                        .shuffle(shuffle_buffer_size)
                        .batch(batch_size, drop_remainder = True)
                        .cache()
                        .prefetch(tf.data.AUTOTUNE)
                        )
    return batched_train_data


    

    




    