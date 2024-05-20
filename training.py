from utils import *

all_notes = []

with open('notes.pkl', 'rb') as file:
    all_notes = pickle.load(file)

all_notes = pd.concat(all_notes)
n_notes = len(all_notes)
train_notes = np.stack([all_notes[key] for key in key_order],axis=1)
notes_ds = tf.data.Dataset.from_tensor_slices(train_notes)

seq_ds = create_sequences(notes_ds, seq_length, vocab_size)

batch_size = 64
buffer_size = n_notes - seq_length
train_ds = (seq_ds
            .shuffle(buffer_size)
            .batch(batch_size, drop_remainder=True)
            .cache()
            .prefetch(tf.data.experimental.AUTOTUNE))



callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./checkpoints/ckpt_{epoch}.weights.h5',
        save_weights_only=True),
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        verbose=1,
        restore_best_weights=True),
]

epochs = 50

history = model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
)

model.save_weights(final_weights)

plt.plot(history.epoch, history.history['loss'], label='total loss')
plt.show()
