# LION - Like it or not
<hr/>
LION - python based app that automatically collects your favourite songs of the same genre, turns into dataset and train CNN model to predict if the random song of the same genre will suit your preferences.

## How does LION works
LION has two main modules:
- Data preprocessing,
- Model creation and prediction handler.

### Data preprocessing
DataPreprocessing module has two classes: **FileManagement** and **Plot** which inherits functions and variables from first class. Function **data_manipulation** create info csv file with necessary informations for further preprocessing, process liked and disliked sets of mp3 audio files and converts them to array of signal and sample rate. Then with **batch_plot** function four figures: spectograms, mels, mfccs and chromas are created and saved in train directory. There are other fuctions like **delete_plots** for removing fugures for training and/or prediction directory. Function **shuffle_df** shuffle created dataframe for training model.

### Model creation and prediction handler
Models module has two functions. First function called **build_model** create CNN model with four X inputs (each for different figures) and one y output with binary outcome. Second function **prediction** create figures from prediction folder with mp3 files, then converts those plots to numpy.ndarrays for model prediction.

### Model is still in process of building and testing to get better results.
