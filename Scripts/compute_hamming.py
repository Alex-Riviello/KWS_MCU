import matplotlib.pyplot as plt
import numpy as np

class HammingWindowMaker():
    """Computes Hamming Window."""
    
    def __init__(self):
        self.window_size_ = 480
        self.ham_window_ = np.hamming(self.window_size_)

    def write_to_file(self):
        """400 x 1"""
        file1 = open("hamming.txt", "w")
        data_string = "float HAMMING_WINDOW[480] = \n{"
        for i in range(self.window_size_):
            data_string += "{0:.2f}".format(self.ham_window_[i])
            if ((i%20) == 0) and (i != 0) and (i != (self.window_size_-1)):
                data_string += ",\n"
            elif i != self.window_size_-1:
                data_string += ", "
            else:
                data_string += "};"
        file1.write(data_string)
        file1.close() 
    
if __name__ == '__main__':
    HAMMING_WINDOW = HammingWindowMaker()
    HAMMING_WINDOW.write_to_file()


