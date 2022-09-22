import pandas as pd 
import numpy as np 
import tensorflow as tf
import pretty_midi as midi 
import os

#Convert raw midi files into a numpy ndarray of notes
def midi_to_notes(directory: str):
    #midi_data is a list of PrettyMIDI objects
    midi_data = []
    for filename in os.listdir(directory):
        midi_data.append(midi.PrettyMIDI(os.path.join(directory, filename)))
    #Pull notes from midi file
    velocities = []
    pitches = []
    starts = []
    ends = []
    durations = []
    steps = []
    #Iterate through all midi objects, pull notes and zip together in a list 
    for midi_obj in midi_data:
        piano = sorted(midi_obj.instruments[0].notes, key = lambda note: note.start)
        prev_start = piano[0].start
        for note in piano:
            velocities.append(note.velocity)
            pitches.append(note.pitch)
            starts.append(note.start)
            ends.append(note.end)
            durations.append(note.get_duration())
            steps.append(note.start - prev_start)
            prev_start = note.start


    note_data_lists = list(zip(velocities, pitches, starts, ends, durations, steps))
    note_data = pd.DataFrame(note_data_lists, columns = ['Velocities', 'Pitches', 'Starts', 'Ends', 'Durations', 'Steps'])
    note_data = note_data.drop('Starts', axis = 1)
    note_data = note_data.drop('Ends', axis = 1)
    note_data = note_data.to_numpy()
    return note_data  

#Convert a Pandas DataFrame of notes into a midi file 
def note_to_midi(notes: pd.DataFrame, out_file: str, instrument_name: str) -> midi.PrettyMIDI:
    ret_obj = midi.PrettyMIDI()
    instrument = midi.Instrument(midi.instrument_name_to_program(instrument_name))

    prev_start = 0
    for i, note in notes.iterrows():
        start = float(note['step']+ prev_start)
        end = float(start + note['duration'])
        note = midi.Note(
            velocity = int(note['velocity']),
            pitch = int(note['pitch']),
            start = start,
            end = end
        )
        instrument.notes.append(note)
        prev_start = start 
    
    ret_obj.instruments.append(instrument)
    ret_obj.write(out_file)
    return ret_obj 

#Predict notes based on a trained keras model
def predict_note(notes: np.array, model: tf.keras.Model, temperature: int) -> int:

    assert temperature > 0
    inputs = tf.expand_dims(notes, 0)

    predictions = model.predict(inputs)
    pitch_logits = predictions['pitch']
    velocity_logits = predictions['velocity']
    step = predictions['step']
    duration = predictions['duration']

    pitch_logits /= temperature
    velocity_logits /= temperature
    pitch = tf.random.categorical(pitch_logits, num_samples = 1)
    velocity = tf.random.categorical(velocity_logits, num_samples = 1)
    velocity = tf.squeeze(velocity, axis = -1)
    pitch = tf.squeeze(pitch, axis = -1) 
    step = tf.squeeze(step, axis = -1)
    duration = tf.squeeze(duration, axis = -1)

    return int(velocity), int(pitch),  float(step), float(duration)

def generate_notes(starter_notes: np.ndarray, 
    num_predictions: int, 
    model, 
    temperature, 
    num_velocities: int, 
    num_pitches: int):

    starter_notes_normed = starter_notes / np.array([num_velocities, num_pitches, 1,1])
    composition = [] 
    prev_start = 0

    for _ in range(num_predictions):
        velocity, pitch, step, duration = predict_note(starter_notes_normed, model, temperature) #Get predicted note from model 
        start = prev_start + step 
        end = start + duration 
        input_note = (velocity, pitch, step, duration) #Store note value as a tuple
        composition.append((*input_note, start, end)) #Append unpacked tuple, start and end data to composition list 
        np.delete(starter_notes_normed, 0, axis = 0) #Remove the first note of the sequence 
        np.append(starter_notes_normed, np.expand_dims(input_note, 0), axis = 0) #Append the newly predicted note to recycle 
        prev_start = start 

    composition = pd.DataFrame(composition, columns = ['velocity', 'pitch', 'step', 'duration', 'start', 'end'])
    return composition