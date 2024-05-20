from utils import *

filenames = glob.glob('midi_songs/**/*.mid*', recursive=True)

num_files = len(filenames)

all_notes = []
for i,f in enumerate(filenames[:num_files]):
    notes,tmpo = midi_to_notes(f)
    all_notes.append(notes)
    print(f"Readings files {i+1}/{num_files}")

with open('notes.pkl', 'wb') as file:
    pickle.dump(all_notes, file)