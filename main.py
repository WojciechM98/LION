import DataPreprocessing
from Models import build_model, prediction_preprocessing, prediction
import os
from tqdm import tqdm
import json
import tensorflow as tf

print('Creating mp3 dataset and csv file.')
prepro = DataPreprocessing.FileManagement()
prepro.data_manipulation()

print('Creating plots for LIKE category.')
prepro = DataPreprocessing.Plot()
files = os.listdir('songs/like')
# prepro.delete_plots(['predictions'])
try:
    with open('process_info.json', 'r') as data:
        info = json.load(data)
except FileNotFoundError:
    dictionary = {'like': 'mp3', 'dislike': 'mp3'}
    with open('process_info.json', 'w') as data:
        json.dump(dictionary, data)
    with open('process_info.json', 'r') as data:
        info = json.load(data)

if info['like'] != 'mp3':
    file_index = files.index(info['like'])
    files = files[file_index:]

for file in tqdm(files):
    signal, sr = prepro.load_mp3(f'songs/like/{file}')
    signal, sr, time_stample = prepro.select_signal_from_specs(signal, sr)
    if signal is False:
        print('Breaking from loop...')
        info['like'] = file
        with open('process_info.json', 'w') as data:
            json.dump(info, data)
        break
    prepro.add_time_stample(file, time_stample)
    prepro.batch_plot(plots=[prepro.plot_spec, prepro.plot_mel, prepro.plot_mfccs, prepro.plot_chroma], signal=signal,
                      sr=sr, fname=file, save='save')
    prepro.df.to_csv(prepro.data_file, index=False)
print('Successfully ended plotting for class LIKE.')

print('Creating plots for DISLIKE category.')
files = os.listdir('songs/dislike')
if info['dislike'] != 'mp3':
    file_index = files.index(info['dislike'])
    files = files[file_index:]
column = 'length'
for file in tqdm(files):
    index = prepro.df[prepro.df['working name'] == file].index[0]
    time = prepro.df.loc[index, column]
    start = round(time / 2)
    duration = round(time / 2) + 15
    signal, sr = prepro.load_mp3(f'songs/dislike/{file}', start=start, duration=duration)
    prepro.batch_plot(plots=[prepro.plot_spec, prepro.plot_mel, prepro.plot_mfccs, prepro.plot_chroma], signal=signal,
                      sr=sr, fname=file, save='save')
prepro.df.to_csv(prepro.data_file, index=False)
with open('process_info.json', 'w') as data:
    json.dump(info, data)
print('Successfully ended plotting for class DISLIKE.')

print('Shuffling csv file and creating training set for model training.')
shuffled_df = prepro.shuffle_df()
(X_train_specs, X_train_mels, X_train_mfccs, X_train_chromas), y_train = prepro.get_X_y(shuffled_df)
print(f'Spectrograms: {X_train_specs.shape}\nMels: {X_train_mels.shape}\nMFCCs: {X_train_mfccs.shape}\n'
      f'Chromas: {X_train_chromas.shape}\nOutput data: {y_train.shape}')

print('Building CNN model.')
model = build_model(prepro.img_size)
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
model.summary()
model.fit([X_train_specs, X_train_mels, X_train_mfccs, X_train_chromas],
          y=y_train, validation_split=0.2, epochs=10)

print('Single prediction')
# prepro.delete_plots(['predictions'])
prediction_preprocessing(prepro)
prediction(prepro, model)
