import glob
from music21 import converter,instrument,note,chord
from music21.midi import MidiFile
import pandas as pd
import numpy as np
import collections

from matplotlib import pyplot as plt
from typing import Optional

filenames = glob.glob('midi_songs/**/*.mid*', recursive=True)

for i,f in enumerate(filenames):
    sounds = converter.parse(f).parts[0].recurse()
    tmpo = sounds.getElementsByClass('MetronomeMark')[0].getQuarterBPM()
    
    if tmpo != 120 or i % 100 ==0:
        print(f)
        print(tmpo)