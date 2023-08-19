from tqdm import tqdm
import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, GlobalAveragePooling2D, Dropout
from keras import layers


def prediction_preprocessing(prepro_instance):
    pred_songs_path = 'songs/prediction'
    print('Creating plots.')
    songs = os.listdir(pred_songs_path)
    for song in songs:
        signal, sr = prepro_instance.load_mp3(os.path.join(pred_songs_path, song))
        print(f'Loading song: {song}\n')
        signal_slices = []
        # Creating list with start times of each slice, because cannot use np.ndarray with .index()
        signal_time = []
        # Creating list with sliced song with step of 15 seconds
        for step in range(0, round(signal.shape[0] / sr), 15):
            signal_time.append(step)
        step_n = round(len(signal) / len(signal_time))
        for sig in range(step_n, len(signal), step_n):
            signal_slices.append(signal[sig:sig + step_n])
        for _ in tqdm(range(len(signal_time)-1)):
            signal = np.array(signal_slices[_])
            prepro_instance.batch_plot(plots=[prepro_instance.plot_spec, prepro_instance.plot_mel,
                                              prepro_instance.plot_mfccs, prepro_instance.plot_chroma],
                                       signal=signal, sr=sr, fname=f'{song}-{_}', save='prediction')


def prediction(prepro_instance, model):
    img_size = prepro_instance.img_size

    # Generating converted images to arrays for model prediction
    def group_file_names(files):
        names = {}
        name = "name"
        for file in files:
            split_name = file.rsplit('-')[0]
            if split_name != name:
                names[split_name] = {'specs': [],
                                     'mels': [],
                                     'mfccs': [],
                                     'chromas': []}
        return names

    print('Creating datasets.')
    pred_paths = prepro_instance.paths['predictions']
    print(pred_paths)
    files = os.listdir(pred_paths['specs'])
    file_names = group_file_names(files)
    for fig_name in pred_paths:
        print(fig_name)
        path = pred_paths[fig_name]
        files = os.listdir(path)
        print(f'Converting {fig_name} to arrays.\n')
        for fname in file_names:
            for plot_file in tqdm(files):
                if plot_file.rsplit('-')[0] == fname:
                    plot_path = os.path.join(path, plot_file)
                    img = prepro_instance.resize_normalize(plot_path, img_size)
                    file_names[fname][fig_name].append(img)
            file_names[fname][fig_name] = np.array(file_names[fname][fig_name])
            # print(f"-- Spectrograms: {file_names[fname]['specs'].shape}\n-- Mels: {file_names[fname]['mels'].shape}\n"
            #       f"-- MFCCs: {file_names[fname]['mfccs'].shape}\n-- Chromas: {file_names[fname]['chromas'].shape}")
            # print(f"\n-- Spectrograms: {type(file_names[fname]['specs'])}\n-- Mels: {type(file_names[fname]['mels'])}\n"
            # f"-- MFCCs: {type(file_names[fname]['mfccs'])}\n-- Chromas: {type(file_names[fname]['chromas'])}")

    print('Predicting...')
    for fname in file_names:
        print("loop")
        result = model.predict([file_names[fname]['specs'], file_names[fname]['mels'],
                                file_names[fname]['mfccs'], file_names[fname]['chromas']])
        result = result.tolist()
        result = [round(number[0], 2) for number in result]
        print(result)
    print('Predicting Done!')

def build_model(img_size):
    # Input layer for spectrograms
    input_specs = Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_specs)
    x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = layers.Dropout(0.5)(x)
    # x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(x)
    # x = layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    x = Model(inputs=input_specs, outputs=x)

    #Input layer for mel spectograms
    input_mels = Input(shape=(img_size[0], img_size[1], 3))
    y = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_mels)
    y = layers.MaxPool2D(pool_size=(2, 2), strides=2)(y)
    y = layers.Dropout(0.5)(y)
    # y = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(y)
    # y = layers.MaxPool2D(pool_size=(2, 2), strides=2)(y)
    y = Model(inputs=input_mels, outputs=y)

    # Input layer for mfccs
    input_mfccs = Input(shape=(img_size[0], img_size[1], 3))
    m = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_mfccs)
    m = layers.MaxPool2D(pool_size=(2, 2), strides=2)(m)
    m = layers.Dropout(0.5)(m)
    # m = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(m)
    # m = layers.MaxPool2D(pool_size=(2, 2), strides=2)(m)
    m = Model(inputs=input_mfccs, outputs=m)

    # Input layer for chromas
    input_chromas = Input(shape=(img_size[0], img_size[1], 3))
    c = layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu')(input_chromas)
    c = layers.MaxPool2D(pool_size=(2, 2), strides=2)(c)
    c = layers.Dropout(0.5)(c)
    # c = layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu')(c)
    # c = layers.MaxPool2D(pool_size=(2, 2), strides=2)(c)
    c = Model(inputs=input_chromas, outputs=c)

    #Concatenate two layers together
    combined = layers.concatenate([x.output, y.output, m.output, c.output])

    z = layers.Flatten()(combined)
    z = layers.Dense(128, activation='relu')(z)
    z = layers.Dense(64, activation='relu')(z)
    z = layers.Dense(1, activation='sigmoid')(z)

    model = Model(inputs=[x.input, y.input, m.input, c.input], outputs=z)
    return model
