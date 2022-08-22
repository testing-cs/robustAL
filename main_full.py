import argparse
import os
import torch
import torch.nn as nn
from lenet5 import LeNet5
from utils_svhn import vgg8
import torchattacks
from utils import evaluate_standard, evaluate_adv, get_loaders, save_best
from config import hyperparameters
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='standard', type=str, choices=['adv', 'standard'], help="perform adversarial training or standard training")
    parser.add_argument('--dataName', default='mnist', type=str, choices=['mnist', 'svhn', 'fashion'], help="name of datasets")
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
    save_model = f"full-{args.train}-{args.ite}.pt"
    parameters = hyperparameters(dataName, modelName)
    train_loader, _, _, _, val_loader = get_loaders(parameters.data_dir, parameters.batch_size, dataName)
    save_model_name = parameters.save_model_root + save_model

    # Training
    best_acc = 0
    for epoch in range(parameters.epochs):
        model.train()
        if args.train == "standard":
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        elif args.train == "adv":
            atk = torchattacks.PGD(model, eps=parameters.epsilon, alpha=parameters.alpha, steps=parameters.attack_iters, random_start=False)
            for X, y in train_loader:
                X, y = X.to(device), y.to(device)
                adv_data = atk(X, y)
                output_adv = model(adv_data)
                loss = criterion(output_adv, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            print("wrong training type")
        if args.train == "standard":
            val_acc = evaluate_standard(val_loader, model)
            if val_acc >= best_acc:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'metric_best': val_acc,
                    'epoch_best': epoch,
                    'current_stage': 0,
                    'candidate_index': 0
                }
                best_acc = val_acc
                save_best(checkpoint, save_model_name)
        elif args.train == "adv":
            val_attack_acc = evaluate_adv(val_loader, model, "pgd", dataName)
            if val_attack_acc >= best_acc:
                checkpoint = {
                    'state_dict': model.state_dict(),
                    'optimizer': opt.state_dict(),
                    'metric_best': val_attack_acc,
                    'epoch_best': epoch,
                    'current_stage': 0,
                    'candidate_index': 0
                }
                best_acc = val_attack_acc
                save_best(checkpoint, save_model_name)


if __name__ == "__main__":
    main()
