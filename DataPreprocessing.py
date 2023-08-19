from pprint import pp
import os
import pandas as pd
from tqdm import tqdm
import string
import random
import librosa
import librosa.display
import librosa.feature
import librosa.effects
import numpy as np
import matplotlib.pyplot as plt
import math
from inspect import signature
from typing import List, Dict, Any
import cv2
import mplcursors

class FunctionCaller:
    @staticmethod
    def call_functions(func):
        def wrapper(*args, **kwargs):
            sig = signature(func)
            params = sig.parameters
            target_args = {}

            for i, arg_value in enumerate(args):
                param_name = list(params.keys())[i]
                target_args[param_name] = arg_value

            for arg_name, arg_value in kwargs.items():
                if arg_name in params:
                    target_args[arg_name] = arg_value

            return func(**target_args)
        return wrapper


class FileManagement:
    def __init__(self):
        self.songs_path = 'songs'
        self.data_file = 'dataset.csv'
        self.path_cols = ['specs', 'mels', 'mfccs', 'chromas']
        self.img_size = (150, 150)
        try:
            self.df = pd.read_csv(self.data_file)
        except FileNotFoundError:
            self.df = pd.DataFrame(columns=['working name', 'song name', 'length', 'class', 'specs', 'mels', 'mfccs', 'chromas', 'time stample'])

    @staticmethod
    def id_generator(size=6):
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=size))

    def check_orig_fname(self, fname=''):
        index = self.df.set_index('working name')
        exist = index['song name'].get(fname)
        return exist

    @staticmethod
    def resize_normalize(path, img_size=()):
        img = cv2.imread(path)
        res = cv2.resize(img, dsize=img_size, interpolation=cv2.INTER_CUBIC)
        img_norm = cv2.normalize(res, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        return img_norm

    def get_X_y(self, df):
        X_img = {
            'specs': [],
            'mels': [],
            'mfccs': [],
            'chromas': [],
        }
        y = []
        df.reset_index(inplace=True)
        for column in self.path_cols:
            y.clear()
            for path in df[column]:
                img = self.resize_normalize(path, img_size=self.img_size)
                X_img[column].append(img)
                label = df.loc[df[column] == path, 'class'].iloc[0]
                y.append(label)
            X_img[column] = np.array(X_img[column])
        y = np.array(y)
        return (X_img['specs'], X_img['mels'], X_img['mfccs'], X_img['chromas']), y

    def shuffle_df(self):
        shuffled_df = self.df.sample(frac=1)
        return shuffled_df

    def add_time_stample(self, fname='', time_stample=None):
        index = self.df[self.df['working name'] == fname].index[0]
        column = 'time stample'
        num1, num2 = time_stample[0], time_stample[1]
        self.df.loc[index, column] = f'{num1},{num2}'

    def data_manipulation(self):
        directories = os.listdir(self.songs_path)
        files = []
        entry_files = []
        for dire in directories:
            entry_files = os.listdir(f'{self.songs_path}/{dire}')
            for file in entry_files:
                if dire == 'dislike':
                    files.append((file, '0'))
                elif dire == 'like':
                    files.append((file, '1'))
                else:
                    raise Exception('Folder "dislike" or "like" not found.')

        for tup in tqdm(files):
            file, _class = tup
            if file not in self.df.iloc[:]['working name'].values.tolist():
                self.df.at[files.index(tup), 'song name'] = file.replace(' (320 kbps)', '')
                self.df.at[files.index(tup), 'working name'] = f'{self.id_generator()}.mp3'
                self.df.at[files.index(tup), 'class'] = _class
                if _class == '1':
                    path = f'{self.songs_path}/like'
                elif _class == '0':
                    path = f'{self.songs_path}/dislike'
                signal, rate = librosa.load(f'{path}/{file}')
                self.df.at[files.index(tup), 'length'] = float("{:.2f}".format(signal.shape[0] / rate))
                os.renames(f'{path}/{file}', f"{path}/{self.df.at[files.index(tup), 'working name']}")
            else:
                print(f"--- File: {file} is already preprocessed. \nOriginal name: {self.check_orig_fname(file)}")
        self.df.to_csv(self.data_file, index=False)
        print(self.df)

# TODO 2 make another plots if needed
#  https://librosa.org/doc/latest/auto_examples/plot_display.html#sphx-glr-download-auto-examples-plot-display-py
class Plot(FileManagement):
    def __init__(self):
        super().__init__()
        self.hop_length = 512
        self.n_fft = 512
        self.base_path = './data'
        directories = [name for name in os.listdir(self.base_path)]
        paths = {key: os.listdir(os.path.join(self.base_path, key).replace("\\", "/")) for key in directories}
        for name in paths:
            paths_dict = {key: os.path.join(self.base_path, name, key).replace("\\", "/") for key in paths[name]}
            paths[name] = paths_dict
        self.paths = paths

    def load_mp3(self, path='', start=0.0, duration=None):
        signal, sr = librosa.load(path, offset=start, duration=duration)
        split = librosa.effects.split(signal, frame_length=self.hop_length * 4, hop_length=self.hop_length)
        signal = signal[split[0][0]:split[0][1]]
        return signal, sr

    @staticmethod
    def save_plot(plt, fname='', path=''):
        split_name = fname.rsplit('.mp3')
        name = split_name[0]+split_name[1]
        created_path = f'{path}/{name}.png'
        plt.axis('off')
        plt.savefig(created_path, bbox_inches='tight')
        plt.close()
        return created_path

    def delete_plots(self, directories=None):
        """For now directories: plots, predictions"""
        if directories is None:
            directories = []
        for folder in directories:
            print(f'* Deleting plots from "{folder}" folder...')
            path = os.path.join(self.base_path, folder).replace('\\', '/')
            for walk in os.walk(path):
                for dir_name in walk[1]:
                    plot_folder_path = os.path.join(path, dir_name)
                    for name in os.walk(plot_folder_path):
                        for file in name[2]:
                            os.remove(os.path.join(plot_folder_path, file))
        print('* Plots deleted.')

    # Spectogram
    def to_decibels(self, signal):
        # Perform short time Fourier Transformation of signal and take absolute value of results
        stft = np.abs(librosa.stft(signal, hop_length=self.hop_length, n_fft=self.n_fft))
        # Convert to dB
        dB = librosa.amplitude_to_db(stft, ref=np.max)
        return dB

    # Function to plot the converted audio signal
    @FunctionCaller.call_functions
    def plot_spec(self, signal, sr, save='show', fname='', chose_length=False):
        """save = prediction, save or None"""
        y_axis = 'log'
        figsize = (15, 15)
        if chose_length is not False:
            figsize = (30, 10)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        spec = librosa.display.specshow(self.to_decibels(signal), sr=sr, x_axis='time', y_axis=y_axis, ax=ax,
                                        hop_length=self.hop_length)
        # ax.set(title=f'Spectrogram of {fname}')
        # fig.colorbar(spec, format='%+2.0f dB')
        if save == 'prediction':
            self.save_plot(plt, fname=fname, path=self.paths['predictions']['specs'])
        elif save == 'save':
            created_path = self.save_plot(plt, fname=fname, path=self.paths['plots']['specs'])
            row = self.df[self.df['song name'] == fname].index.values[0]
            self.df.at[row, 'specs'] = created_path
        else:
            plt.show()
        if chose_length is not False:
            return plt, fig, ax
        return plt

    # Env mask for Mels
    @staticmethod
    def env_mask(wav, threshold):
        wav = np.abs(wav)
        mask = wav > threshold
        return wav[mask]

    @FunctionCaller.call_functions
    def plot_mel(self, signal, save='show', fname=''):
        signal = self.env_mask(signal, 0.005)
        # Create signal for mel spectrogram
        signal_spec = librosa.feature.melspectrogram(y=signal)
        g = librosa.amplitude_to_db(signal_spec)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        spec = librosa.display.specshow(g, ax=ax, x_axis='time', y_axis='mel')
        # ax.set(title=f'Mel Spectrogram of signal {fname}')
        # plt.colorbar(spec, format='%+2.0f dB')
        if save == 'prediction':
            self.save_plot(plt, fname=fname, path=self.paths['predictions']['mels'])
        elif save == 'save':
            created_path = self.save_plot(plt, fname=fname, path=self.paths['plots']['mels'])
            row = self.df[self.df['song name'] == fname].index.values[0]
            self.df.at[row, 'mels'] = created_path
        else:
            plt.show()
        return plt

    # Mel Frequency Cepstral Coefficients (MFCCs)
    @FunctionCaller.call_functions
    def plot_mfccs(self, signal, save='show', fname=''):
        # Create mfcc signal for mfcc plot
        MFCC = librosa.feature.mfcc(y=signal)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        spec = librosa.display.specshow(MFCC, x_axis='time', ax=ax)
        # ax.set(title=f'MFCCs of signal - {fname}')
        # plt.colorbar(spec)
        if save == 'prediction':
            self.save_plot(plt, fname=fname, path=self.paths['predictions']['mfccs'])
        elif save == 'save':
            created_path = self.save_plot(plt, fname=fname, path=self.paths['plots']['mfccs'])
            row = self.df[self.df['song name'] == fname].index.values[0]
            self.df.at[row, 'mfccs'] = created_path
        else:
            plt.show()
        return plt

    # Pitch and Chromagrams
    @FunctionCaller.call_functions
    def plot_chroma(self, signal, sr, save='show', fname=''):
        # Create signal for chroma plot
        signal_chroma = librosa.feature.chroma_stft(y=signal, sr=sr, n_fft=self.n_fft)
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        spec = librosa.display.specshow(signal_chroma, y_axis='chroma', x_axis='time', ax=ax)
        # ax.set(title=f'Signal Chromagram - {fname}')
        # fig.colorbar(spec, ax=ax)
        if save == 'prediction':
            self.save_plot(plt, fname=fname, path=self.paths['predictions']['chromas'])
        elif save == 'save':
            created_path = self.save_plot(plt, fname=fname, path=self.paths['plots']['chromas'])
            row = self.df[self.df['song name'] == fname].index.values[0]
            self.df.at[row, 'chromas'] = created_path
        else:
            plt.show()
        return plt

    def plot_centralspec(self):
        pass

    def batch_plot(self, plots: List[Any], signal: np.array, sr: int, save, fname=''):
        for plot in plots:
            plt = plot(signal=signal, sr=sr, fname=fname, hop_length=self.hop_length, save=save)
            plt.cla()
            plt.clf()
            plt.close()

    def select_signal_from_specs(self, signal, sr):
        vert_line_time = 0
        vert_line = None
        exit_flag = None
        exit_flag = False

        # Function to update the plot when the lines are moved
        def update_lines(vert_line):
            vert_line.set_ydata([0, 1])
            plt.draw()

        # Function to handle cursor selection on the plot
        def on_select(event):
            nonlocal vert_line
            nonlocal vert_line_time
            if event.name != 'button_press_event':
                return
            if event.inaxes != ax:
                return
            x = event.xdata
            if vert_line is None:
                vert_line = ax.axvline(x, color='g', linestyle='--')
                vert_line_time = x
                update_lines(vert_line)

        def on_key(event):
            if event.key == ' ':
                plt.close('all')
            if event.key == 'enter':
                global exit_flag
                exit_flag = True
                plt.close('all')

        plt, fig, ax = self.plot_spec(signal=signal, sr=sr, save='prediction', chose_length=True)
        mplcursors.cursor(hover=True)

        plt.gcf().canvas.mpl_connect('button_press_event', on_select)
        plt.gcf().canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        if exit_flag is True:
            return False, False, False

        time_start = vert_line_time
        time_end = vert_line_time + 15
        signal_start = round(time_start * sr)
        signal_end = round(time_end * sr)
        signal = signal[signal_start:signal_end]
        print(f'Start time: {time_start}, End time: {time_end}')
        plt.close()
        return signal, sr, (time_start, time_end)
