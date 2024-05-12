import glob
from music21 import converter,instrument,note,chord
from music21.midi import MidiFile
import pandas as pd
import numpy as np
import collections

from matplotlib import pyplot as plt
from typing import Optional

filenames = glob.glob('midi_songs/**/*.mid*', recursive=True)
sample_midi = filenames[0]

score = converter.parse(sample_midi)

stuff=score.parts[0].recurse()

print(stuff.getElementsByClass('TimeSignature')[0])


for i in range(30):
    print(stuff[i])
    print(stuff[i].offset)


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
