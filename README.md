# Automated Audio Source Extraction From Recordings Using Deep Neural Networks
This repository contains an updated code version, related to my diploma thesis "[Automated Audio Source Extraction From Recordings Using Deep Neural Networks](https://angelosbousis.azurewebsites.net/api/PdfFile?filePath=~%2Ffile%2Fthesis_angelos_bousis.pdf)"(in Greek) supervised by prof. [Nikolaos Mitianoudis](https://utopia.duth.gr/nmitiano/index.html). This repo covers the case of Singing Voice Separation problem. The main difference between this and my initial implementation is the creation of the dataset. 

## Deep Learning Frameworks and Libraries
For this implementation I used Tensorflow v2.4.1 and Keras Functional API.

## Output examples
Here are some output examples extracted using my pre-trained model which is provided in this repository. The music tracks are the following 

[Ιουλία Καραπατάκη - Μπιρ Αλλάχ (Σαν βγαίνει ο Χότζας στο τζαμί) (exact time interval 90-120 sec.)](https://www.youtube.com/watch?v=nv2rp5JCWj0) - [Vocals](https://drive.google.com/file/d/195HOyaQi12PSyn3J7ry-Bx7IEKuBjDfE/view?usp=sharing) - [Accompaniment](https://drive.google.com/file/d/1--TvTstFaiiHsO5zYySpYlGQy-5W5TAC/view?usp=sharing)    
[Villagers of Ioannina City - Perdikomata (exact time interval 360-390 sec.)](https://www.youtube.com/watch?v=MsCB4iocPJE) - [Vocals](https://drive.google.com/file/d/1-JkdoGPFZ5hy31A6OGR58o1mzWJfC3j5/view?usp=sharing) - [Accompaniment](https://drive.google.com/file/d/1-GNxHFLwEoq1GabRHUZXmWUqKftQxN2z/view?usp=sharing) 
[Boston - More Than A Feeling 45-75 sec.](https://www.youtube.com/watch?v=oR4uKcvQbGQ) - [Vocals](https://drive.google.com/file/d/1-EHxr9P_uSxp3YU9pnBZVcFJQDzJldzZ/view?usp=sharing) -  [Accompaniment](https://drive.google.com/file/d/1-7i-TyArhaqc4_fPO9JTMjeizufP4Og3/view?usp=sharing)    
[Porcupine Tree - Trains (exact time interval 70-100 sec.)](https://www.youtube.com/watch?v=0UHwkfhwjsk) - [Vocals](https://drive.google.com/file/d/1-2kXKGhOGIGsA0_fMqNGdaDp_DhtzrKn/view?usp=sharing) - [Accompaniment](https://drive.google.com/file/d/1-024Sv4KaZupkpTupty80E7c7pl07HWZ/view?usp=sharing)  
[TCTS - Not Ready For Love (feat. Maya B) (exact time interval 30-60 sec)](https://www.youtube.com/watch?v=kQY6dzXLBnI) = [Vocals](https://drive.google.com/file/d/1-LA2KWoOSfxzTPEe0cUdulYoEhrpVJAK/view?usp=sharing) - [Accompaniment](https://drive.google.com/file/d/1-L7lJbzqmcu1blbqnPKVfEoDYA1RnVww/view?usp=sharing)

Note that the current pre-trained model is not fully optimized yet. It can be further improved and lead to better results.


## Installation
Use Python 3.8 interpreter in order to get librosa version 8.0 library work smoothly. Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the following packages. See requirements.txt for the exact package versions.
```bash
pip install -Iv tensorflow-gpu==2.4.1
pip install -Iv librosa==0.8
pip install -Iv numpy==1.20.1
pip install -Iv musdb==0.40.0
```
An easier approach is to use [Anaconda](https://www.anaconda.com/) environment. If you use anaconda, install the following
```bash
conda install -c conda-forge librosa
conda install -c conda-forge ffmpeg
conda install -c conda-forge musdb
conda install -c anaconda tensorflow-gpu 
```

## Dataset
First, you have to download [MUSDB18](https://sigsep.github.io/datasets/musdb.html) dataset. I used the predefined split, 84 tracks for training and 16 tracks for validation.

## Method
This method borrows much from these two papers [1](https://pdfs.semanticscholar.org/41f0/973c0777f6da3b47fa035aa0bc071c8f02f8.pdf?_ga=2.214795095.1524381320.1605807192-246561933.1601815407), [2](https://arxiv.org/pdf/1812.01278.pdf). 

## Preprocessing
I randomly cut 30 second chunks, downsample to 22.05kHz and apply Short Time Fourier Transform to time domain signals. Then I feed the model with 9 overlapping magnitude spectograms. Their overlap is 1 frame.

## Architecture
2D Convolutional layers with (3x12) kernels are used for feature extraction. Max Pooling layers for downsampling the frequency dimension(a process similar to MFCCs extraction). Dropout layers and Early Stopping for regularization and Dense layers, which have been successfully used for music source separation. The output is a soft mask, thus the final layer is activated by a sigmoid function.

![Model's Architecture](https://github.com/gelobs/Automated-Audio-Source-Extraction-From-Recordings-Using-Deep-Neural-Networks/blob/main/img/architecture.png?raw=true)

## Training
The model is trained to minimize the binary cross-entropy of between the ideal binary mask and the estimated source in the time-frequency domain for the validation set. The ideal binary mask is created by comparing mixture's and vocals' magnitude spectrograms. 

## Postprocessing
Reconstruction of the 9 spectrogram frames, application of the estimated soft masks to magnitude spectrogram and element-wise multiplication(Hadamard product) with the original phase. Thresholds are applied to vocal and accompaniment soft masks to further improve separation. I recommend 0.1-0.2 for vocals and 0.75-0.85 for accompaniment, depending on track. Then I apply inverse Short Time Fourier Transform to reconstruct the signals to time domain, and upsample to 44.1kHz.

In Google Colaboratory jupyter notebook "postprocessing_yt_no_out.ipynb", in "Estimating soft masks" section, I have commented out the following
```python
......
# Estimating soft masks
vocals_soft_mask = np.zeros((freq_bins, time_bins))
# predictions = model.predict(input)
for i in range(time_bins):
    temp_soft_mask = np.transpose(np.reshape(model.predict(input[i:i+1,:]), (num_frames, freq_bins)))
    # temp_soft_mask = np.transpose(np.reshape(predictions[i,:],(num_frames, freq_bins)))
......
```
This is slower method, but is useful when trying to source separate tracks with over 30 sec. duration, because the process then becomes more memory intense. The faster equivalent method is to comment and uncomment the following lines(in the same section)
```python
......
# Estimating soft masks
vocals_soft_mask = np.zeros((freq_bins, time_bins))
predictions = model.predict(input)
for i in range(time_bins):
    # temp_soft_mask = np.transpose(np.reshape(model.predict(input[i:i+1,:]), (num_frames, freq_bins)))
    temp_soft_mask = np.transpose(np.reshape(predictions[i,:],(num_frames, freq_bins)))
......
```
This works well and fast for small durations, with the same results, but it will probably run your system out of memory for tracks with over 30 sec. duration.

## Usage for Deep Learning station owners and Requirements
To execute the code you should have a GPU with at least 12GB memory , e.g. NVIDIA TITAN Xp. The related code is in "Python" folder. When training a model, the code assumes that the data is stored as .stem.mp4 files with folder structure ../Dataset/musdb18/train/source.stem.mp4 and ../Dataset/musdb18/test/source.stem.mp4.

To train model run
```bash
python train.py
```

Use "tracks_in_batch" parameter according to your memory restrictions.
 
## Usage for Google Colaboratory users
I mostly worked my diploma thesis using Google Colaboratory. You can find the related code in "Google Colaboratory" folder. You can separate tracks from YouTube using [YoutubeDL](https://github.com/ytdl-org/youtube-dl). I recommend to save MUSDB18 dataset to your Google Drive, then mount it 

```python
from google.colab import drive
drive.mount('/gdrive')
```
and copy MUSDB18 dataset to the root of Colab's virtual machine
```python
!gsutil -m cp -r "path-to-musdb18-on-gdrive" "/root"
```
Currently if you have a pro subscription at Google Colaboratory, you will be able to train your model up to 24 hours without disconnecting, whereas for users without pro subscription the limit is 6 hours. You should save your model after each epoch on Google Drive using the following Keras Callback
```python
checkpoint_path = "your-model-saving-path-to-gdrive/model.h5"
tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='val_binary_crossentropy',verbose=1, save_best_only=True, mode='min')
```
Use "steps per epoch" parameter according to the training time limit.

## Important note
When loading a model with
```python
model = tf.keras.models.load_model("path-to-model", compile=False)
```
the Tensorflow versions should match, i.e. if you use Tensorflow v2.4.1 when training the model and try to load model using Tensorflow v2.2. you will possibly get an error. The pre-built model is created with Tensorflow v2.4.1.

## Acknowledgements
This repository borrows many ideas from the following
[CNN-with-IBM-for-Singing-Voice-Separation](https://github.com/EdwardLin2014/CNN-with-IBM-for-Singing-Voice-Separation) and [blind-audio-source-separation-cnn](https://github.com/ivasique/blind-audio-source-separation-cnn)
