import torch
import model
import utility as util
import model as md
import audio_preprocess as ap
import torch.utils.data as data

if __name__ == "__main__":

    TRAIN = True

    config = dict(no_cuda=False, n_epochs=30, lr=[0.01, 0.001, 0.0001, 0.0001, 0.00001, 0.000001], schedule=[3000, 6000, 8000, 8500, 9500], batch_size=64, dev_every=1, seed=0,
        use_nesterov=False, gpu_no=0, cache_size=32768, momentum=0.9, weight_decay=0.001, 
        group_speakers_by_id=True, silence_prob=0.1, noise_prob=0.8, n_dct_filters=40, input_length=16000,
        n_mels=40, timeshift_ms=100, unknown_prob=0.1, train_pct=80, dev_pct=10, test_pct=10,
        wanted_words=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"], 
        data_folder="../../../Datasets/GoogleVoiceSearchDataset", output_file="quantizedSpeech")

    """
    config = dict(no_cuda=False, n_epochs=26, lr=[0.01, 0.001, 0.0001, 0.0001, 0.00001], schedule=[3000, 6000, 8000, 8500], batch_size=64, dev_every=1, seed=0,
        use_nesterov=False, gpu_no=0, cache_size=32768, momentum=0.9, weight_decay=0.00001, 
        group_speakers_by_id=True, silence_prob=0.1, noise_prob=0.8, n_dct_filters=14, input_length=16000,
        n_mels=40, timeshift_ms=100, unknown_prob=0.1, train_pct=80, dev_pct=10, test_pct=10,
        wanted_words=["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"], 
        data_folder="../../../Datasets/GoogleSpeechCommands_v2", output_file="quantizedSpeech")
    """

    if TRAIN :
        util.train(config)   
    else:
        train_set, dev_set, test_set = ap.SpeechDataset.splits(config)
        test_loader = data.DataLoader(
            test_set,
            batch_size=min(len(test_set), 16),
            shuffle=True,
            collate_fn=test_set.collate_fn)
        model = md.TCResNet8()
        model.load(config["output_file"])
        util.evaluate(config, model, test_loader)
    
 