# LION - Like it or not
<hr/>
LION - python based app that automatically collects your favourite songs of the same genre, turns into dataset and train CNN model to predict if the random song of the same genre will suit your preferences.

## How does LION works
LION has two main modules:
- Data preprocessing,
- Model creation and prediction handler.

### Data preprocessing
DataPreprocessing module has two classes: **FileManagement** and **Plot** which inherits functions and variables from first class. Function **data_manipulation** create csv file with necessary informations for further preprocessing, process liked and disliked sets of mp3 audio files and converts them to array of signal and sample rate. Then with **batch_plot** function four graphs: spectograms, mels, mfccs and chromas are created and saved in train directory. There are other fuctions like **delete_plots** for removing graphs for training and/or prediction directory. Function **shuffle_df** shuffle created dataframe for training model.

### Model creation and prediction handler
Models module has three functions. First function called **prediction_preprocessing** create plots (just like **batch_plots**) from mp3 file. Signal is cut into 15 seconds sections. Every section then is plotted into graphs. Second function **prediction** converts created plots to numpy.ndarrays for model prediction. Third function **build_model** create CNN model with four X inputs (each for different graphs) and one y output with binary outcome.

### Model is still in process of building and testing to get better results.
