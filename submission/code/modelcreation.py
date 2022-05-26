import time                             # time is used to seed the random shuffler from NumPy.
from tensorflow import keras            # Keras is our tool of choice for model creation and saving.
from keras import layers
from keras import optimizers            # Optimizers were crucial for this project, as choosing the right one improved
# our accuracy by 50%.
import fft
import numpy as np
import os                               # Used to read our dataset.

full_data = []
full_labels = []


counter = 0

drawobj = None

for dir in os.walk('dataset'):
    if len(dir[0].split('\\')) == 2:
        label = dir[0].split('\\')[1].split('-')[0]
        label_list = np.array([0.0, 0.0, 0.0, 0.0])                # The output template, for example: [0 0 1 0] for 2
        label_list[int(label)] = 1.0
        counter += 1
        check = np.load(dir[0] + "\\RadarIfxAvian_00\\radar.npy")  # This line is used to load the NumPy file from the
        # radar recorder
        check = check / 4095.0                                     # Data normalisation, making the values 0 to 1
        print(f"Importing file {counter}...")
        screen = np.abs(np.squeeze(fft.range_doppler_fft(check)))  # Getting the Doppler range data from the raw rec.
        for i in screen:
            full_data.append(i[16:48, 0:32])                       # Cutting the data for efficiency
            full_labels.append(label_list)

keras.backend.set_floatx('float64')                                # Using float64 values for added precision

model = keras.Sequential(                                          #
    [
        layers.Dense(1024, activation="relu", name="layer1", input_shape=(32, 32)),
        layers.Flatten(),
        layers.Dense(650, activation="relu", name="layer2"),
        layers.Dense(4, activation='sigmoid', name="layer3"),
    ]
)

seed = int(time.time())

full_data = np.array(full_data)
full_labels = np.array(full_labels)

print(full_data.shape)
print(full_data.dtype)
print(full_labels.shape)

[print(i.shape, i.dtype) for i in model.inputs]

np.random.seed(seed)
np.random.shuffle(full_data)
np.random.seed(seed)
np.random.shuffle(full_labels)

opti = optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=opti,
              loss='categorical_crossentropy',
              run_eagerly=True,
              metrics=['accuracy'])

cutoff = int(len(full_data) * 0.8)

# Train model
model.fit(full_data[:cutoff], full_labels[:cutoff],
          epochs=15,
          verbose=1,
          validation_data=(full_data[cutoff:], full_labels[cutoff:]))

score = model.evaluate(full_data[cutoff:], full_labels[cutoff:], verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# model.predict(full_data[0:1])
model.save('radar_model_fit')

