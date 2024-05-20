from utils import *

sample_file = filenames[100]
raw_notes,tmpo = midi_to_notes(sample_file)

temperature = 3.0
num_predictions = 100

sample_notes = np.stack([raw_notes[key] for key in key_order], axis=1)

input_notes = (
    sample_notes[:seq_length] / np.array([vocab_size, 1, 1]))

fuck = 'checkpoints/ckpt_25.weights.h5'
model.load_weights(fuck)

num_outputs = 10

for i in range(num_outputs):
    generated_notes = []
    prev_start = 0
    for _ in range(num_predictions):
        pitch, step, duration = predict_next_note(input_notes, model, temperature)
        start = prev_start + step
        end = start + duration
        input_note = (pitch, step, duration)
        generated_notes.append((*input_note, start, end))
        input_notes = np.delete(input_notes, 0, axis=0)
        input_notes = np.append(input_notes, np.expand_dims(input_note, 0), axis=0)
        prev_start = start

    generated_notes = pd.DataFrame(
        generated_notes, columns=(*key_order, 'start', 'end'))

    out_file = f'outputs/output--{i}.midi'

    out_pm = notes_to_midi(
        notes=generated_notes, out_file=out_file)
    
    print(f'Generated {i+1} outputs')
