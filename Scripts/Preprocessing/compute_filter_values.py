import matplotlib.pyplot as plt
import numpy as np

class FilterCreator():
    """Computes Mel FilterBanks."""
    
    def __init__(self):
        self.sample_rate_ = 16000
        self.num_mel_ = 40
        self.fft_size_ = 512
        self.num_bins_ = self.fft_size_//2 + 1
        self.mel_filterbank_ = np.zeros([self.num_mel_, self.num_bins_])
        self.ham_window_ = np.hamming(self.fft_size_)
        self.text_values = np.zeros((self.mel_filterbank_.shape[0], 32))
        self.text_indices = np.zeros(self.mel_filterbank_.shape[0])

    def lin2mel(self, freq):
        return 2595*np.log10(1+freq/700)

    def mel2lin(self, mel):
        return (10**(mel/2595)-1)*700 

    def make_mel_filterbank(self):
		""" Function taken from EDX Speech Recognition Course by Microsoft. """
        lo_mel = self.lin2mel(0) # Base freq = 0 Hz
        hi_mel = self.lin2mel(self.sample_rate_//2) # Max Freq = 8000 Hz (typically)
        mel_freqs = np.linspace(lo_mel, hi_mel, self.num_mel_+2) # Equally spaced out
        bin_width = self.sample_rate_/self.fft_size_ # typically 31.25 Hz, bin[0]=0 Hz, bin[1]=31.25 Hz,..., bin[256]=8000 Hz
        mel_bins = np.floor(self.mel2lin(mel_freqs)/bin_width) #index values of the bins
        
        for i in range(0,self.num_mel_):
            left_bin = int(mel_bins[i])
            center_bin = int(mel_bins[i+1])
            right_bin = int(mel_bins[i+2])
            up_slope = 1/(center_bin-left_bin)
            for j in range(left_bin, center_bin):
                self.mel_filterbank_[i, j] = (j - left_bin)*up_slope
            down_slope = -1/(right_bin-center_bin)
            for j in range(center_bin, right_bin):
                self.mel_filterbank_[i, j] = (j-right_bin)*down_slope
    
    def extract_segment(self):
        position = 0
        for i in range(self.mel_filterbank_.shape[0]):
            for j in range(self.mel_filterbank_.shape[1]):
                if self.mel_filterbank_[i, j] != 0:
                    self.text_indices[i] = j
                    position = j
                    break
            self.text_values[i, :] = self.mel_filterbank_[i, position:position+32]

    def write_to_file(self):
        """40 x 1"""
        file1 = open("mel_indices.txt", "w")
        data_string = ""
        data_string += "int FILTER_INDICES[40] = {"
        for i in range(self.text_indices.shape[0]):
            data_string += str(int(self.text_indices[i]))
            if i != (self.text_indices.shape[0] - 1):
                data_string += ", "
            else:
                data_string += "};"
        file1.write(data_string)
        file1.close() 

        """40 x 32"""
        file2 = open("mel_filters.txt", "w")
        data_string = ""
        data_string += "float FILTERBANK[1280] = \n{"
        for i in range(self.text_values.shape[0]):
            for j in range(self.text_values.shape[1]):
                data_string += "{0:.2f}".format(self.text_values[i, j])
                if j != (self.text_values.shape[1]-1):
                    data_string += ", "
                elif i != (self.text_values.shape[0]-1):
                    data_string += ",\n"
                else:
                    data_string += "\n"
        data_string += "};"
        file2.write(data_string)
        file2.close() 
    
if __name__ == '__main__':
    MEL_FILTER = FilterCreator()
    MEL_FILTER.make_mel_filterbank()
    MEL_FILTER.extract_segment()
    MEL_FILTER.write_to_file()


