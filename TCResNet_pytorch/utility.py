import torch
import torch.nn as nn
import torch.utils.data as data
import speech_dataset as sd

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...') 


def evaluate_testset(model, test_dataloader):
    # Final test
    test_loss = 0.0
    test_correct = 0.0
    model.load()
    model.eval()

    criterion = nn.CrossEntropyLoss()
        
    for batch_idx, (audio_data, labels) in enumerate(test_dataloader):

        if train_on_gpu:
            model.cuda()
            audio_data = audio_data.cuda()
            labels = labels.cuda()

        output = model(audio_data)
        loss = criterion(output, labels)
        test_loss += loss.item()*audio_data.size(0)
        batch_accuracy = (torch.sum(torch.argmax(output, 1) == labels).item())/audio_data.shape[0]

        print("Test step #{} - Loss : {:.2f} - Accuracy : {:.2f}".format(batch_idx, loss, batch_accuracy))

        test_correct += torch.sum(torch.argmax(output, 1) == labels).item()

    test_loss = test_loss/len(test_dataloader.dataset)
    test_accuracy = 100.0*(test_correct/len(test_dataloader.dataset))
    print("================================================")
    print(" FINAL ACCURACY : {:.2f}% - TEST LOSS : {:.2f}".format(test_accuracy, test_loss))
    print("================================================")


def train(model, root_dir, word_list, num_epoch):
    """
    Trains TCResNet. TODO: Complete.
    """

    # Enable GPU training
    if train_on_gpu:
        model.cuda()
    
    # Loading dataset
    ap = sd.AudioPreprocessor() # Computes Log-Mel spectrogram
    train_files, dev_files, test_files = sd.split_dataset(root_dir, word_list)

    train_data = sd.SpeechDataset(train_files, "train", ap, word_list)
    dev_data = sd.SpeechDataset(dev_files, "dev", ap, word_list)
    test_data = sd.SpeechDataset(test_files, "test", ap, word_list)

    train_dataloader = data.DataLoader(train_data, batch_size=64, shuffle=True)
    dev_dataloader = data.DataLoader(dev_data, batch_size=64, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=64, shuffle=True)

    # Setting training parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=20, gamma=0.1)

    # Training
    step_idx = 0
    valid_accuracy = 0
    previous_valid_accuracy = 0

    for epoch in range(num_epoch):

        train_loss = 0.0
        valid_loss = 0.0
        valid_correct = 0.0

        # Training (1 epoch)
        model.train()

        for batch_idx, (audio_data, labels) in enumerate(train_dataloader):

            if train_on_gpu:
                audio_data = audio_data.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            output = model(audio_data)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()*audio_data.size(0)
            batch_accuracy = float(torch.sum(torch.argmax(output, 1) == labels).item())/float(audio_data.shape[0])

            if (batch_idx%10 == 0):
                print("Train step #{} - Loss: {:.2f} - Accuracy: {:.2f}".format(step_idx, loss, batch_accuracy))

            step_idx += 1

        # Validation (1 epoch)
        model.eval()
        
        for batch_idx, (audio_data, labels) in enumerate(dev_dataloader):

            if train_on_gpu:
                audio_data = audio_data.cuda()
                labels = labels.cuda()

            output = model(audio_data)
            loss = criterion(output, labels)
            valid_loss += loss.item()*audio_data.size(0)
            batch_accuracy = (torch.sum(torch.argmax(output, 1) == labels).item())/audio_data.shape[0]

            print("Dev step #{} - Loss: {:.2f} - Accuracy: {:.2f}".format(batch_idx, loss, batch_accuracy))

            valid_correct += torch.sum(torch.argmax(output, 1) == labels).item()

        # Loss statistics
        train_loss = train_loss/len(train_dataloader.dataset)
        valid_loss = valid_loss/len(dev_dataloader.dataset)
        valid_accuracy = 100.0*(valid_correct/len(dev_dataloader.dataset))
        print("============================================================================")
        print(" EPOCH #{} - ACCURACY : {:.2f}% - TRAIN LOSS : {:.2f} - VALIDATION LOSS : {:.2f}".format(epoch, valid_accuracy, train_loss, valid_loss))
        print("============================================================================")
        
        if (valid_accuracy > previous_valid_accuracy):
            previous_valid_accuracy = valid_accuracy
            print("Saving current model...")
            model.save()
       
        # Update scheduler (for decaying learning rate)
        scheduler.step()

    # Final test
    evaluate_testset(model, test_dataloader)
    
