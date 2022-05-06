from numpy.lib.function_base import i0
import torch.utils.data as data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torchaudio
import random
import torch
import math
import os

geforce_rtx_3060_xc = 0

if torch.cuda.is_available():
    torch.cuda.set_device(geforce_rtx_3060_xc)

def print_spectrogram(spectrogram, labels, word_list):
    """ Prints spectrogram to screen. Used for debugging. 
        Input(s): Tensor of dimensions [n_batch x 1 x n_mel x 101]
        Output(s): None
    """
    np_spectrogram = (spectrogram.view(spectrogram.shape[0], spectrogram.shape[2], spectrogram.shape[1], 101)).numpy()
    fig, axs = plt.subplots(8, int(spectrogram.shape[0]/8))
    for i in range(8):
        for j in range(int(spectrogram.shape[0]/8)):
            axs[i, j].imshow(np_spectrogram[8*i + j, 0])
            axs[i, j].set_title(word_list[labels[8*i + j]])
            axs[i, j].axis('off')
    plt.show()
    

def split_dataset(root_dir, word_list, n_samples=2000, split_pct=[0.8, 0.1, 0.1]):
    """ Generates a list of paths for each sample and splits them into training, validation and test sets.

        Input(s): 
            - root_dir (string): Path where to find the dataset. Folder structure should be:
                                 -> ROOT_DIR
                                     -> yes
                                        -> {yes samples}.wav
                                     -> no
                                        -> {no samples}.wav
                                     -> etc.
            - word_list (list of strings): List of all words need to train the network on ('unknown' and 'silence')
                                           should be added to this list.
            - n_samples (int): Number of samples to use for each word. This limit was set to add new words to train.
                               Default is 2000.
            - split_pct (list of floats): Sets proportions of the dataset which are allocated for training, validation
                                          and testing respectively. Default is 80% training, 10% validation & 10% testing.
        Output(s):

                                         
    """

    unknown_list = []

    train_set = []
    dev_set = []
    test_set = []

    n_train = int(n_samples*split_pct[0])
    n_dev = int(n_samples*split_pct[1])

    available_words = os.listdir(root_dir)
    for i, word in enumerate(available_words):

        if (word in word_list):
            temp_set = [(root_dir + word + "/" + wav_file, word) for wav_file in os.listdir(root_dir + word) \
                        if wav_file.endswith(".wav")]
            # If word samples are insufficient, re-use same data multiple times.
            # This isn't ideal since validation/test sets might contain data from the training set.
            if (len(temp_set) < n_samples):
                temp_set *= math.ceil(n_samples/len(temp_set))
            temp_set = temp_set[:n_samples]
            random.shuffle(temp_set)
            train_set += temp_set[:n_train]
            dev_set += temp_set[n_train:n_train+n_dev]
            test_set += temp_set[n_train+n_dev:]

        elif ((word != "_background_noise_") and ("unknown" in word_list)): # Adding unknown words
            temp_set = [(root_dir + word + "/" + wav_file, "unknown") for wav_file in os.listdir(root_dir + word) \
                        if wav_file.endswith(".wav")]
            unknown_list += temp_set

    # Adding unknown category
    if ("unknown" in word_list):
        random.shuffle(unknown_list)
        unknown_list = unknown_list[:n_samples]
        train_set += unknown_list[:n_train]
        dev_set += unknown_list[n_train:n_train+n_dev]
        test_set += unknown_list[n_train+n_dev:]

    # Adding silence category
    if ("silence" in word_list):
        temp_set = [(root_dir + "_background_noise_" + "/" + wav_file, "silence") for wav_file in os.listdir(root_dir \
                    + "_background_noise_") if wav_file.endswith(".wav")]
        if (len(temp_set) < n_samples):
            temp_set *= math.ceil(n_samples/len(temp_set))
        temp_set = temp_set[:n_samples]
        train_set += temp_set[:n_train]
        dev_set += temp_set[n_train:n_train+n_dev]
        test_set += temp_set[n_train+n_dev:]
    
    # Shuffling dataset
    random.shuffle(train_set)
    random.shuffle(dev_set)
    random.shuffle(test_set)

    return train_set, dev_set, test_set


class SpeechDataset(data.Dataset):

    def __init__(self, data_list, dataset_type, transforms, word_list, is_noisy=True, is_shift=True, sample_length=16000):
        """ types include [TRAIN, DEV, TEST] """
        self.data_list = data_list
        self.dataset_type = dataset_type
        self.is_noisy = is_noisy
        self.is_shift = is_shift
        self.sample_length = sample_length
        self.transforms = transforms
        self.word_list = word_list
    
    def shift_audio(self, audio_data, max_shift=160):
        """ Shifts audio.
        """
        shift_val = random.randint(-max_shift, max_shift)
        zero_fill = torch.zeros(audio_data.shape[0], abs(shift_val))
        if (shift_val < 0):
            audio_data = torch.cat((audio_data[:, 0:(audio_data.shape[1]+shift_val)], zero_fill), 1)
        else:
            audio_data = torch.cat((zero_fill, audio_data[:, shift_val:]), 1)
        return audio_data

    def load_data(self, data_element):
        """ Loads audio, shifts data and adds noise. """
        wav_data = torchaudio.load(data_element[0])[0]

        # Background noise used for silence needs to be shortened to 1 second.
        if (data_element[1] == "silence"):
            slice_idx = random.randint(0, wav_data.view(-1).shape[0] - self.sample_length - 1)
            amplitude = random.random()
            out_data = amplitude*wav_data[:, slice_idx:slice_idx+self.sample_length]
        else:
            out_data = 1*wav_data

        # Pad smaller audio files with zeros to reach 1 second (16_000 samples)
        if (out_data.shape[1] < 16000):
            out_data = F.pad(out_data, pad=(0, (16000-out_data.shape[1])), mode='constant', value=0)
        
        # Adds audio shift (upto 100 ms)
        if self.is_shift:
            out_data = self.shift_audio(out_data)

        # Add random noise
        if self.is_noisy:
            out_data += 0.01*torch.randn(out_data.shape)
            
        return (out_data, data_element[1])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        cur_element = self.load_data(self.data_list[idx])
        cur_element = (cur_element[0], self.word_list.index(cur_element[1]))
        return self.transforms(cur_element)
        

class AudioPreprocessor(): 
    
    def __init__(self):
        self.spectrogram = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=480,
                hop_length=160,
                f_min=0,
                f_max=8000,
                n_mels=40,
                window_fn=torch.hann_window
            ),
            torchaudio.transforms.AmplitudeToDB()
        )

    def __call__(self, data):
        o_data = self.spectrogram(data[0])
        # Set Filters as channels
        o_data = o_data.view(o_data.shape[1], o_data.shape[0], o_data.shape[2]) 
        return o_data, data[1]

if __name__ == "__main__":
    # Test example
    root_dir = "../../datasets/kws_mcu_dataset/"
    word_list = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
    ap = AudioPreprocessor()
    train, dev, test = split_dataset(root_dir, word_list)
    # Dataset
    train_data = SpeechDataset(train, "train", ap, word_list)
    dev_data = SpeechDataset(dev, "train", ap, word_list)
    test_data = SpeechDataset(test, "train", ap, word_list)
    # Dataloaders
    train_dataloader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    dev_dataloader = data.DataLoader(dev_data, batch_size=64, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=64, shuffle=True)
    # Dataloader tests
    train_spectrogram, train_labels = next(iter(train_dataloader))
    print_spectrogram(train_spectrogram, train_labels, word_list)





