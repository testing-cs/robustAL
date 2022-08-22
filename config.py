import os


class hyperparameters():
    def __init__(self, dataName, modelName):
        if dataName == "mnist":
            self.class_num = 10
            self.batch_size = 100
            self.epochs = 10
            self.epsilon = 0.3
            self.alpha = 0.01
            self.attack_iters = 40
            self.num_initial = 200
            self.num_label = 200
            self.budget = 5000
        elif dataName == "svhn":
            self.class_num = 10
            self.batch_size = 100
            self.epochs = 20
            self.epsilon = 8 / 255
            self.alpha = 2 / 255
            self.attack_iters = 7
            self.num_initial = 1000
            self.num_label = 500
            self.budget = 10000
        elif dataName == "cifar10":
            self.class_num = 10
            self.batch_size = 100
            self.epochs = 10
            self.epochs_retrain = 40
            self.epsilon = 8 / 255
            self.alpha = 2 / 255
            self.attack_iters = 7
            self.num_initial = 1000
            self.num_label = 500
            self.budget = 25000
        elif dataName == "fashion":
            self.class_num = 10
            self.batch_size = 100
            self.epochs = 10
            self.epsilon = 0.3
            self.alpha = 0.01
            self.attack_iters = 40
            self.num_initial = 200
            self.num_label = 200
            self.budget = 6000
        self.data_dir = "data".format(dataName)
        self.save_model_root = "{0}/{1}/savedM/".format(dataName, modelName)
        self.save_log_root_test = "{0}/{1}/savedLT/".format(dataName, modelName)
        if not os.path.isdir(self.save_model_root):
            os.makedirs(self.save_model_root)
        if not os.path.isdir(self.save_log_root_test):
            os.makedirs(self.save_log_root_test)