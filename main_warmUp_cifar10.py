import argparse
import apex.amp as amp
import torch
import torch.nn as nn
import numpy as np
from utils import evaluate_standard, selectLoader, get_loaders, save_best
from utils_cifar10 import getModel
from config import hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr-schedule', default='cyclic', type=str, choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.2, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
                        help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
                        help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    parser.add_argument('--master-weights', action='store_true',
                        help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')
    parser.add_argument('--ite', default=0, type=int, help="The repetition ID of experiments")
    parser.add_argument('--modelID', default=0, type=int, help="The ID of models")
    return parser.parse_args()


def main():
    args = get_args()
    dataName = "cifar10"
    modelID = args.modelID
    model, modelName = getModel(modelID)
    save_model = f"initial-{args.ite}.h5"
    parameters = hyperparameters(dataName, modelName)
    train_loader, train_data, _, _, val_loader = get_loaders(parameters.data_dir, parameters.batch_size, dataName)
    save_model_name = parameters.save_model_root + save_model

    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)
    amp_args = dict(opt_level=args.opt_level, loss_scale=args.loss_scale, verbosity=False)
    if args.opt_level == 'O2':
        amp_args['master_weights'] = args.master_weights
    model, opt = amp.initialize(model, opt, **amp_args)
    criterion = nn.CrossEntropyLoss()

    lr_steps = parameters.epochs * len(train_loader)
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
                                                      step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)

    current_stage = -1
    candidate_index = list(np.arange(len(train_data)))
    # Training
    sub_train_loader, candidate_index = selectLoader(train_data, model, parameters.num_initial, "random", parameters.batch_size, candidate_index, class_num=parameters.class_num, dataName=dataName, epsilon=None)
    best_acc = 0
    for epoch in range(parameters.epochs):
        model.train()
        for X, y in sub_train_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            opt.zero_grad()
            with amp.scale_loss(loss, opt) as scaled_loss:
                scaled_loss.backward()
            opt.step()
            scheduler.step()
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