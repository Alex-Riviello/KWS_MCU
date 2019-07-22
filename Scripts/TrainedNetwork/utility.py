import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
import audio_preprocess
import model as md

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')    

def manhattan_regularizer(model, alpha):
    r_loss = 0
    for W in model.parameters():
        r_loss += torch.norm(torch.abs(alpha-torch.abs(W)))
    return r_loss

def print_eval(name, scores, labels, loss, end="\n"):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss))
    return accuracy

def evaluate(config, model=None, test_loader=None):
    if not test_loader:
        _, _, test_set = audio_preprocess.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=len(test_set))
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        scores = model(model_in)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        total += model_in.size(0)
    print("final test accuracy: {}".format(sum(results) / total))

def train(config):

    # Obtain audio files via a SpeechDataset class (MFCCs not yet calculated)
    train_set, dev_set, test_set = audio_preprocess.SpeechDataset.splits(config)
    model = md.TCResNet8()
    if train_on_gpu:
        model.cuda()

    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][0], 
                nesterov=config["use_nesterov"], weight_decay=config["weight_decay"], momentum=config["momentum"])
    criterion = nn.CrossEntropyLoss()

    schedule_steps = config["schedule"]
    schedule_steps.append(np.inf)
    sched_idx = 0
    max_acc = 0
    validation_curve = []

    train_loader = data.DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True, drop_last=True,
        collate_fn=train_set.collate_fn)
    dev_loader = data.DataLoader(
        dev_set,
        batch_size=min(len(dev_set), 16),
        shuffle=True,
        collate_fn=dev_set.collate_fn)
    test_loader = data.DataLoader(
        test_set,
        batch_size=min(len(test_set), 16),
        shuffle=True,
        collate_fn=test_set.collate_fn)
    step_no = 0

    for epoch_idx in range(config["n_epochs"]):
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()
            if train_on_gpu:
                model_in = model_in.cuda()
                labels = labels.cuda()

            model_in = Variable(model_in, requires_grad=False)
            scores = model(model_in)
            labels = Variable(labels, requires_grad=False)
            loss = criterion(scores, labels)
            loss.backward()
            optimizer.step()
            step_no += 1
            if step_no > schedule_steps[sched_idx]:
                sched_idx += 1
                print("changing learning rate to {}".format(config["lr"][sched_idx]))
                optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                    nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
            print_eval("train step #{}".format(step_no), scores, labels, loss)

        if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
            model.eval()
            accs = []
            for model_in, labels in dev_loader:
                model_in = Variable(model_in, requires_grad=False)
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)
                loss_numeric = loss.item()
                accs.append(print_eval("dev", scores, labels, loss))
            avg_acc = np.mean(accs)
            print("final dev accuracy: {}".format(avg_acc))
            validation_curve.append(1.0 - avg_acc)

            if avg_acc > max_acc:
                print("saving best model...")
                max_acc = avg_acc
                model.save(config["output_file"])

    evaluate(config, model, test_loader)
    
    plt.plot(np.arange(1, config["n_epochs"]+1), np.asarray(validation_curve))
    plt.show()
 