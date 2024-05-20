import glob
from music21 import converter,note,chord,stream,tempo
import pandas as pd
import numpy as np
import collections
import seaborn as sns
import tensorflow as tf
import pickle

from matplotlib import pyplot as plt
from typing import Optional

filenames = glob.glob('midi_songs/**/*.mid*', recursive=True)

def midi_to_notes(midi_file: str) -> pd.DataFrame:
    piano_score = converter.parse(midi_file)

    piano_sounds = piano_score.parts[0].recurse()
    notes = collections.defaultdict(list)
    tmpo = piano_sounds.getElementsByClass('MetronomeMark')[0].getQuarterBPM()
    prev_start = 0
    curr_beat = 0

    for sound in piano_sounds:
        if isinstance(sound,note.Note):
            start = (float(sound.offset+ curr_beat))
            duration = (float(sound.duration.quarterLength))
            end = start+duration
            notes['pitch'].append(sound.pitch.ps)
            notes['start'].append(start)
            notes['end'].append(end)
            notes['duration'].append(duration)
            notes['step'].append(start-prev_start)
            prev_start = start
        elif isinstance(sound,chord.Chord):
            start = (float(sound.offset + curr_beat))
            duration = (float(sound.duration.quarterLength))
            end = start+duration
            for i_note in sound.pitches:
                notes['pitch'].append(i_note.ps)
                notes['start'].append(start)
                notes['end'].append(end)
                notes['duration'].append(duration)
                notes['step'].append(start-prev_start)
                prev_start = start
        elif isinstance(sound,stream.Measure):
            curr_beat = sound.offset
    
    theNotes= pd.DataFrame({name:np.array(values) for name,values in notes.items()})

    return theNotes,tmpo


def plot_piano_roll(notes:pd.DataFrame, count: Optional[int] = None):
    if count:
        title = f'First {count} notes'
    else:
        title = f'Whole track'
        count = len(notes['pitch'])
    plt.figure(figsize=(20, 4))
    plot_pitch = np.stack([notes['pitch'], notes['pitch']], axis=0)
    plot_start_stop = np.stack([notes['start'], notes['end']], axis=0)
    plt.plot(
        plot_start_stop[:, :count], plot_pitch[:, :count], color="b", marker=".")
    plt.xlabel('Time (s)')
    plt.ylabel('Pitch')
    _ = plt.title(title)

def plot_distributions(notes: pd.DataFrame, drop_percentile=2.5):
  plt.figure(figsize=[15, 5])
  plt.subplot(1, 3, 1)
  sns.histplot(notes, x="pitch", bins=20)

  plt.subplot(1, 3, 2)
  max_step = np.percentile(notes['step'], 100 - drop_percentile)
  sns.histplot(notes, x="step", bins=np.linspace(0, max_step, 21))
  
  plt.subplot(1, 3, 3)
  max_duration = np.percentile(notes['duration'], 100 - drop_percentile)
  sns.histplot(notes, x="duration", bins=np.linspace(0, max_duration, 21))

key_order = ['pitch','step','duration']

def create_sequences(
    dataset: tf.data.Dataset,
    seq_length: int,
    vocab_size: int = 128
) -> tf.data.Dataset:
    
    seq_length += 1

    windows = dataset.window(seq_length,shift=1,stride=1,
                             drop_remainder=True)
    
    sequences = windows.flat_map(lambda x: x.batch(seq_length,
                                                  drop_remainder=True))
    
    def scale_pitch(x):
        x = x/[vocab_size,1,1]
        return x
    
    def split_labels(sequence):
        inputs = sequence[:-1]
        labels_dense = sequence[-1]
        labels = {key:labels_dense[i] for i,key in enumerate(key_order)}

        return scale_pitch(inputs),labels
    
    return sequences.map(split_labels, num_parallel_calls=tf.data.AUTOTUNE)

def mse_pos_pressure(y_true: tf.Tensor, y_pred: tf.Tensor):
    mse = (y_true-y_pred)**2
    pos_pressure = 15*tf.minimum(tf.maximum(0.1-y_pred,0),2*abs(y_pred))
    return tf.reduce_mean(mse + pos_pressure)

def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    tmpo: float = 120) -> stream.Stream:

    new_stream = stream.Stream()
    new_stream.insert(0,tempo.MetronomeMark(number=tmpo))

    for i,note_info in notes.iterrows():
        start = note_info['start']
        duration = note_info['duration']
        pitch = note_info['pitch']
        
        new_note = note.Note(pitch)
        new_note.duration.quarterLength = duration        
        new_note.offset = start
        new_stream.insert(start,new_note)
    new_stream.write('midi',out_file)
    return new_stream

def predict_next_note(
    notes: np.ndarray, 
    model: tf.keras.Model, 
    temperature: float = 1.0) -> tuple[int, float, float]:
  """Generates a note as a tuple of (pitch, step, duration), using a trained sequence model."""

  assert temperature > 0

  # Add batch dimension
  inputs = tf.expand_dims(notes, 0)

  predictions = model.predict(inputs, verbose=0)
  pitch_logits = predictions['pitch']
  step = predictions['step']
  duration = predictions['duration']
 
  pitch_logits /= temperature

  pitch = tf.random.categorical(pitch_logits, num_samples=1)
  pitch = tf.squeeze(pitch, axis=-1)
  duration = tf.squeeze(duration, axis=-1)
  step = tf.squeeze(step, axis=-1)

  # `step` and `duration` values should be non-negative

  step+= tf.random.normal(shape=(1,),mean=0.05,stddev=0.3)
  step = tf.maximum(0, step)
  duration = tf.maximum(0, duration)

  return int(pitch), float(step), float(duration)


seq_length = 50
vocab_size = 128

input_shape = (seq_length, 3)
learning_rate = 0.001

inputs = tf.keras.Input(input_shape)
x = tf.keras.layers.LSTM(vocab_size)(inputs)

outputs = {
    'pitch': tf.keras.layers.Dense(vocab_size, name='pitch')(x),
    'step': tf.keras.layers.Dense(1, name='step')(x),
    'duration': tf.keras.layers.Dense(1, name='duration')(x)
}

loss = {
    'pitch': tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    'step': mse_pos_pressure,
    'duration': mse_pos_pressure,
}


model = tf.keras.Model(inputs, outputs)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,clipnorm=1.0)

model.compile(
    loss=loss,
    loss_weights={
        'pitch': 2.0,
        'step': 0.2,
        'duration':1.0,
    },
    optimizer=optimizer,
)

final_weights = 'final_weights.keras'


