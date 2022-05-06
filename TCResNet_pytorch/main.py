import torch.utils.data as data
import speech_dataset as sd
import utility as util
import model as md

TRAIN = True

ROOT_DIR = "../../datasets/kws_mcu_dataset/"
WORD_LIST = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
NUM_EPOCH = 60

if __name__ == "__main__":

    model = md.TCResNet8(1, 40, len(WORD_LIST))

    if TRAIN :
        util.train(model, ROOT_DIR, WORD_LIST, NUM_EPOCH)   
    else:
        train, dev, test = sd.split_dataset(ROOT_DIR, WORD_LIST)
        ap = sd.AudioPreprocessor()
        test_data = sd.SpeechDataset(test, "train", ap, WORD_LIST)
        test_dataloader = data.DataLoader(test_data, batch_size=64, shuffle=True)
        util.evaluate_testset(model, test_dataloader)

    