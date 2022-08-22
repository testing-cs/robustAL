import argparse
import torch
import torch.nn as nn
from lenet5 import LeNet5
import numpy as np
from utils_svhn import vgg8
from utils import evaluate_standard, get_loaders, save_best, selectLoader
from config import hyperparameters

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataName', default='mnist', type=str, choices=['mnist', 'svhn', 'fashion'], help="The dataset name")
    parser.add_argument('--ite', default=0, type=int, help="The repetition ID of experiments")
    return parser.parse_args()


def main():
    args = get_args()
    dataName = args.dataName
    if dataName == "mnist" or dataName == "fashion":
        model = LeNet5().to(device)
        learning_rate = 0.001
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        modelName = "lenet5"
    elif dataName == "svhn":
        model = vgg8().to(device)
        learning_rate = 0.001
        opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
        modelName = "VGG8"
    else:
        print("wrong data")
        return
    criterion = nn.CrossEntropyLoss()
    save_model = f"initial-{args.ite}.h5"
    parameters = hyperparameters(dataName, modelName)
    train_loader, train_data, _, _, val_loader = get_loaders(parameters.data_dir, parameters.batch_size, dataName)
    save_model_name = parameters.save_model_root + save_model
    candidate_index = list(np.arange(len(train_data)))
    current_stage = -1
    sub_train_loader, candidate_index = selectLoader(train_data, model, parameters.num_initial, "random", parameters.batch_size, candidate_index, class_num=parameters.class_num, dataName=dataName, epsilon=None)
    # Training
    best_acc = 0
    for epoch in range(parameters.epochs):
        model.train()
        for X, y in sub_train_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        val_acc = evaluate_standard(val_loader, model)
        if val_acc >= best_acc:
            checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'metric_best': val_acc,
                    'epoch_best': epoch,
                    'candidate_index': candidate_index,
                    'current_stage': current_stage
                }
            best_acc = val_acc
            save_best(checkpoint, save_model_name)


if __name__ == "__main__":
    main()
