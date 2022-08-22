import os
import argparse
import torch
import torch.nn as nn
import gc
import numpy as np
gc.collect()
torch.cuda.empty_cache()
from lenet5 import LeNet5
from utils_svhn import vgg8
import torchattacks
from utils import evaluate_standard, evaluate_adv, get_loaders, selectLoader, save_best, load_best
from config import hyperparameters
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default='standard', type=str, choices=['adv', 'standard'])
    parser.add_argument('--dataName', default='svhn', type=str, choices=['mnist', 'svhn', 'fashion'])
    parser.add_argument('--metric', default="random", type=str, choices=["random", "bald", "dfal", "entropy", "gini", "entropyDrop", "lc", "margin", "mcp", "egl", "kcenter", "dre"], help="name of acquisition functions")
    parser.add_argument('--ite', default=0, type=int, help="The iteration ID of experiments")
    return parser.parse_args()


def main():
    args = get_args()
    dataName = args.dataName
    metric = args.metric
    ite = args.ite
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
    parameters = hyperparameters(dataName, modelName)
    criterion = nn.CrossEntropyLoss()
    train_loader, train_data, _, _, val_loader = get_loaders(parameters.data_dir, parameters.batch_size, dataName)
    initial_model = parameters.save_model_root + "initial-{0}.h5".format(ite)
    model, opt, _, _, candidate_index, _ = load_best(initial_model, model, opt)
    stage_num = int(np.ceil((parameters.budget - parameters.num_initial) / parameters.num_label))
    # Training
    for stage in range(stage_num):
        save_model = parameters.save_model_root + f"al-{args.train}-{args.metric}-{stage}-{args.ite}.pt"
        sub_train_loader, candidate_index = selectLoader(train_data, model, parameters.num_label, metric, parameters.batch_size, candidate_index, class_num=parameters.class_num, modelName=modelName, bin_num=50)
        best_acc = 0
        for epoch in range(parameters.epochs):
            model.train()
            if args.train == "standard":
                for X, y in sub_train_loader:
                    X, y = X.to(device), y.to(device)
                    output = model(X)
                    loss = criterion(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            elif args.train == "adv":
                atk = torchattacks.PGD(model, eps=parameters.epsilon, alpha=parameters.alpha, steps=parameters.attack_iters, random_start=False)
                for X, y in sub_train_loader:
                    X, y = X.to(device), y.to(device)
                    adv_data = atk(X, y)
                    output_adv = model(adv_data)
                    loss = criterion(output_adv, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            if args.train == "standard":
                val_acc = evaluate_standard(val_loader, model)
                if val_acc >= best_acc:
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'metric_best': val_acc,
                        'epoch_best': epoch,
                        'candidate_index': candidate_index,
                        'current_stage': stage
                    }
                    best_acc = val_acc
                    save_best(checkpoint, save_model)
            elif args.train == "adv":
                val_attack_acc = evaluate_adv(val_loader, model, "pgd", dataName)
                if val_attack_acc >= best_acc:
                    checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': opt.state_dict(),
                        'metric_best': val_attack_acc,
                        'epoch_best': epoch,
                        'candidate_index': candidate_index,
                        'current_stage': stage
                    }
                    best_acc = val_attack_acc
                    save_best(checkpoint, save_model)
        model, opt, _, _, _, _ = load_best(save_model, model, opt)


if __name__ == "__main__":
    main()