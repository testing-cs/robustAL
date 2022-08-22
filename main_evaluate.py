import argparse
import logging
import torch
import os
from utils import evaluate_standard, evaluate_adv, get_loaders, load_best
from utils_cifar10 import getModel
from lenet5 import LeNet5
from utils_svhn import vgg8
from config import hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='al', type=str, choices=['al', 'full', 'initial'], help="model head")
    parser.add_argument('--train', default='standard', type=str, choices=['adv', 'standard'], help="perform adversarial training or standard training")
    parser.add_argument('--dataName', default='svhn', type=str, choices=['mnist', 'svhn', 'cifar10', 'fashion'], help="name of datasets")
    parser.add_argument('--attack', default='auto', type=str, choices=['auto', 'square', 'pgd'], help="name of adversarial attacks")
    parser.add_argument('--modelID', default=0, type=int, help="only used for cifar10")
    parser.add_argument('--metric', default="mcp", type=str, choices=["random", "bald", "dfal", "entropy", "gini", "entropyDrop", "lc", "margin", "mcp", "egl", "kcenter", "dre"], help="name of acquisition functions")
    parser.add_argument('--ite', default=0, type=int, help="The repetition ID of experiments")
    return parser.parse_args()


def main():
    global _
    args = get_args()
    type = args.type
    dataName = args.dataName
    modelID = args.modelID
    attack = args.attack
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
    elif dataName == "cifar10":
        model, modelName = getModel(modelID)
        opt = torch.optim.SGD(model.parameters(), lr=0.2, momentum=0.9, weight_decay=5e-4)
    else:
        print("wrong data")
        return
    if type == "initial":
        save_name = f"initial-{args.attack}-{args.ite}.txt"
    elif type == "full":
        save_name = f"full-{args.train}-{args.attack}-{args.ite}.txt"
    else:
        save_name = f"al-{args.train}-{args.metric}-{args.attack}-{args.ite}.txt"
    parameters = hyperparameters(dataName, modelName)
    _, _, test_loader, _, _ = get_loaders(parameters.data_dir, parameters.batch_size, dataName)
    logfile = parameters.save_log_root_test + save_name
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=logfile,
        filemode='w',
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.INFO)
    logger.info(args)
    logger.info('Stage \t Epoch \t TestAcc\t TestRobust')
    model.eval()
    if type == "full":
        model_path = parameters.save_model_root + f"fll-{args.train}-{args.ite}.pt"
        model, _, best_epoch, _, _, _ = load_best(model_path, model, opt)
        test_acc = evaluate_standard(test_loader, model)
        test_attack_acc = evaluate_adv(test_loader, model, attack, dataName)
        logger.info('%d \t %d \t %.4f \t %.4f', 1, best_epoch, test_acc, test_attack_acc)
    elif type == "al":
        stage_num = int((parameters.budget - parameters.num_initial) / parameters.num_label)
        for stage in range(stage_num):
            model_path = parameters.save_model_root + f"al-{args.train}-{args.metric}-{stage}-{args.ite}.pt"
            model, _, best_epoch, _, _, _ = load_best(model_path, model, opt)
            test_acc = evaluate_standard(test_loader, model)
            test_attack_acc = evaluate_adv(test_loader, model, attack, dataName)
            logger.info('%d \t %d \t %.4f \t %.4f', stage, best_epoch, test_acc, test_attack_acc)
    else:
        model_path = parameters.save_model_root + "initial-{0}.h5".format(ite)
        model, _, best_epoch, _, _, _ = load_best(model_path, model, opt)
        test_acc = evaluate_standard(test_loader, model)
        test_attack_acc = evaluate_adv(test_loader, model, attack, dataName)
        logger.info('%d \t %d \t %.4f \t %.4f', 1, best_epoch, test_acc, test_attack_acc)


if __name__ == "__main__":
    main()