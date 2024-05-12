import glob
from music21 import converter,instrument,note,chord,stream,meter,tempo
from music21.midi import MidiFile
import pandas as pd
import numpy as np
import collections
import seaborn as sns

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



def notes_to_midi(
    notes: pd.DataFrame,
    out_file: str,
    tmpo: float) -> stream.Stream:

    new_stream = stream.Stream()
    new_stream.insert(0,tempo.MetronomeMark(number=tmpo))



    for i,note_info in notes.iterrows():
        start = note_info['start']
        duration = note_info['duration']
        pitch = note_info['pitch']
        
        if i < 20:
            print(start,duration,pitch)
        
        new_note.duration.quarterLength = duration
        new_note = note.Note(pitch)
        new_note.offset = start
        new_stream.append(new_note)
    new_stream.write('midi',out_file)
    return new_stream

sample_file = filenames[10]
notes,tmpo = midi_to_notes(sample_file)
print(sample_file)

# plot_piano_roll(notes,40)
# plt.show()

new_stream = notes_to_midi(notes,'output.midi',tmpo)

for n in new_stream.flat[:20]:
    print(n)
    print(n.offset)