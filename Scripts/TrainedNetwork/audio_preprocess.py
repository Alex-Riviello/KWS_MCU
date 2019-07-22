import torch.utils.data as data
import matplotlib.pyplot as plt
from chainmap import ChainMap
from scipy.io import wavfile
from enum import Enum
import numpy as np
import librosa
import hashlib
import random
import torch
import sys
import os
import re

class DatasetType(Enum):
    TRAIN = 0
    DEV = 1
    TEST = 2

class SimpleCache(dict):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit
        self.n_keys = 0

    def __setitem__(self, key, value):
        if key in self.keys():
            super().__setitem__(key, value)
        elif self.n_keys < self.limit:
            self.n_keys += 1
            super().__setitem__(key, value)
        return value

class SpeechDataset(data.Dataset):
    LABEL_SILENCE = "__silence__"
    LABEL_UNKNOWN = "__unknown__"

    def __init__(self, data, set_type, config):
        super().__init__()
        self.audio_files = list(data.keys())
        self.set_type = set_type
        self.audio_labels = list(data.values())
        config["bg_noise_files"] = list(filter(lambda x: x.endswith("wav"), config.get("bg_noise_files", [])))
        self.bg_noise_audio = [librosa.core.load(file, sr=16000)[0] for file in config["bg_noise_files"]]
        self.unknown_prob = config["unknown_prob"]
        self.silence_prob = config["silence_prob"]
        self.noise_prob = config["noise_prob"]
        self.input_length = config["input_length"]
        self.timeshift_ms = config["timeshift_ms"]
        self._audio_cache = SimpleCache(config["cache_size"])
        self._file_cache = SimpleCache(config["cache_size"])
        n_unk = len(list(filter(lambda x: x == 1, self.audio_labels)))
        self.n_silence = int(self.silence_prob * (len(self.audio_labels) - n_unk))
        self.audio_processor = AudioPreprocessor(n_mels=config["n_mels"], n_dct_filters=config["n_dct_filters"], hop_ms=10)

    def collate_fn(self, data):
        x = None
        y = []
        for audio_data, label in data:
            audio_tensor = self.audio_processor.compute_log_spectrum(audio_data)
            x = audio_tensor if x is None else torch.cat((x, audio_tensor), 0) 
            y.append(label)
        return x, torch.tensor(y)

    def _timeshift_audio(self, data):
        shift = (16000 * self.timeshift_ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(data, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def load_audio(self, example, silence=False):
        if silence:
            example = "__silence__"
        if random.random() < 0.7:
            try:
                return self._audio_cache[example]
            except KeyError:
                pass
        in_len = self.input_length
        if self.bg_noise_audio:
            bg_noise = random.choice(self.bg_noise_audio)
            a = random.randint(0, len(bg_noise) - in_len - 1)
            bg_noise = bg_noise[a:a + in_len]
        else:
            bg_noise = np.zeros(in_len)

        if silence:
            data = np.zeros(in_len, dtype=np.float32)
        else:
            file_data = self._file_cache.get(example)
            data = librosa.core.load(example, sr=16000)[0] if file_data is None else file_data
            self._file_cache[example] = data
        data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
        if self.set_type == DatasetType.TRAIN:
            data = self._timeshift_audio(data)

        if random.random() < self.noise_prob or silence:
            a = random.random() * 0.1
            data = np.clip(a * bg_noise + data, -1, 1)

        self._audio_cache[example] = data
        return data

    @classmethod
    def splits(cls, config):
        folder = config["data_folder"]
        wanted_words = config["wanted_words"]
        unknown_prob = config["unknown_prob"]
        train_pct = config["train_pct"]
        dev_pct = config["dev_pct"]
        test_pct = config["test_pct"]

        words = {word: i + 2 for i, word in enumerate(wanted_words)} # words 2 to X are in the list -> words = {'command': d, 'random': 3} use words["command"]
        words.update({cls.LABEL_SILENCE:0, cls.LABEL_UNKNOWN:1}) # words 0 and 1 are silence and unknown (cls is to get class instance defined prior)
        sets = [{}, {}, {}] # Train, dev & test
        unknowns = [0] * 3
        bg_noise_files = []
        unknown_files = []

        # Parsing classes
        for folder_name in os.listdir(folder):
            path_name = os.path.join(folder, folder_name) # Folder of a single word
            is_bg_noise = False
            if os.path.isfile(path_name): # If the path exists, continue
                continue
            if folder_name in words:
                label = words[folder_name]
            elif folder_name == "_background_noise_":
                is_bg_noise = True
            else: # All other words are unknown
                label = words[cls.LABEL_UNKNOWN]

            # Parsing individual audio files
            for filename in os.listdir(path_name):

                wav_name = os.path.join(path_name, filename)
                # Fill the noise list with noise files (_background_noise_)
                if is_bg_noise and os.path.isfile(wav_name):
                    bg_noise_files.append(wav_name)
                    continue            
                # Fill the unknown list with unknown (aka not-used keywords)
                elif label == words[cls.LABEL_UNKNOWN]:
                    unknown_files.append(wav_name)
                    continue
                if config["group_speakers_by_id"]:
                    hashname = re.sub(r"_nohash_.*$", "", filename) # Removes _nohash_0 from name
                max_no_wavs = 2**27 - 1
                bucket = int(hashlib.sha1(hashname.encode()).hexdigest(), 16)
                bucket = (bucket % (max_no_wavs + 1)) * (100. / max_no_wavs)
                if bucket < dev_pct:
                    tag = DatasetType.DEV
                elif bucket < test_pct + dev_pct:
                    tag = DatasetType.TEST
                else:
                    tag = DatasetType.TRAIN
                sets[tag.value][wav_name] = label
                

        for tag in range(len(sets)):
            unknowns[tag] = int(unknown_prob * len(sets[tag]))
        random.shuffle(unknown_files)
        a = 0
        for i, dataset in enumerate(sets):
            b = a + unknowns[i]
            unk_dict = {u: words[cls.LABEL_UNKNOWN] for u in unknown_files[a:b]}
            dataset.update(unk_dict)
            a = b

        train_cfg = ChainMap(dict(bg_noise_files=bg_noise_files), config)
        test_cfg = ChainMap(dict(bg_noise_files=bg_noise_files, noise_prob=0), config)
        datasets = (cls(sets[0], DatasetType.TRAIN, train_cfg), cls(sets[1], DatasetType.DEV, test_cfg),
                cls(sets[2], DatasetType.TEST, test_cfg))
        return datasets

    def __getitem__(self, index):
        if index >= len(self.audio_labels):
            return self.load_audio(None, silence=True), 0
        return self.load_audio(self.audio_files[index]), self.audio_labels[index]

    def __len__(self):
        return len(self.audio_labels) + self.n_silence

class AudioPreprocessor(object):
    def __init__(self, sr=16000, n_dct_filters=40, n_mels=40, f_max=8000, f_min=0, n_fft=480, hop_ms=10): #f_max = 4000, f_min = 20
        super().__init__()
        self.n_mels = n_mels
        self.dct_filters = librosa.filters.dct(n_dct_filters, n_mels)
        self.sr = sr
        self.f_max = f_max if f_max is not None else sr // 2
        self.f_min = f_min
        self.n_fft = n_fft
        self.hop_length = sr // 1000 * hop_ms

    def compute_variation(self, data, threshold):
        ref_val = data[0,:,0]
        new_x = torch.zeros((data.shape[0], data.shape[1], data.shape[2]))
        one_mask = torch.ones(self.n_mels)
        for x in range(data.shape[2]-1):
            temp_val = data[0,:,x+1] - ref_val
            d_up = torch.ge(temp_val, threshold).type(torch.FloatTensor)
            d_down = torch.le(temp_val, -threshold).type(torch.FloatTensor)
            # Update of reference
            ref_val = (one_mask-(d_up+d_down))*ref_val + (d_up+d_down)*data[0,:,x+1]
            # Creation of new vector
            new_x[:,:,x] = (d_up - d_down)  
        return new_x

    def splitValues(self, data):
        neg_data = torch.lt(data, 0).type(torch.FloatTensor)
        pos_data = torch.gt(data, 0).type(torch.FloatTensor)
        new_data = torch.cat((2*pos_data-1, 2*neg_data-1), 0)
        return new_data.view(1, 2, 40, 101)

    def quantize_linear(self, data, n_bits):
        data = np.floor(data)
        data = data.int()
        data = data >> (8-n_bits)
        data = data.float()
        return data

    def compute_log_spectrum(self, data):
        data = librosa.feature.melspectrogram(
            data,
            sr=self.sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            fmin=self.f_min,
            fmax=self.f_max)
        data[data > 0] = np.log(data[data > 0])
        data = np.array(data, order="F").reshape(1, self.n_mels, 101).astype(np.float32)
        delta = data.max() - 20.0
        data = data - delta
        data[data<0] = 0
        data = data*12.75
        data = torch.from_numpy(data)
        # data = self.compute_variation(data, 12)
        #data = self.splitValues(data) # Comment this for one channel
        #data = self.quantize_linear(data, 8)
        #data = data/255.0

        #data[data > 0] = np.log(data[data > 0])
        #data = [np.matmul(self.dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        #data = np.array(data, order="F").reshape(1, 101, 40).astype(np.float32)
        #data = torch.from_numpy(data)
        data = data-128.0
        return data


if __name__ == "__main__":
    wav_file =  "../../../Datasets/GoogleVoiceSearchDataset/sheila/0f7dc557_nohash_0.wav"  # 20 
    #wav_file =  "../../../Datasets/GoogleVoiceSearchDataset/up/0d53e045_nohash_1.wav"  #20 ish
    #wav_file =  "../../../Datasets/GoogleVoiceSearchDataset/seven/0fa1e7a9_nohash_0.wav" #23
    fs, data = wavfile.read(wav_file)
    data = data.astype(float)
    AP = AudioPreprocessor()
    melSpect = AP.compute_log_spectrum(data)
    plt.imshow(melSpect[0]) #, cmap=plt.get_cmap("summer"))
    plt.show()

