# DRE: Density-Based Data Selection With Entropy for Adversarial-Robust Deep Learning Models

This is the implementation for the project of DRE.

## Problem definition

Integrate adversarial training into active learning to produce accurate and adversarial robust deep neural networks. 

## Dependency

- python 3.6.10
- torch 1.6.0
- torchattacks 2.14.2
- torchvision 0.7.0
- foolbox 3.3.0
- scikit-learn 0.23.2
- apex **please refer to [NVIDA/apex](https://github.com/NVIDIA/apex.git) for the installation**

## Download the dataset

Description of Dataset: 

- MNIST, Fashion-MNIST, CIFAR-10: loaded from the corresponding dataset of TorchVision. 

- SVHN: download the "train_32x32.mat, test_32x32.mat" from the [site](http://ufldl.stanford.edu/housenumbers/), then take the first 50,000 and 10,000 from each file for training and testing, respectively.

## How to use

### Training models without active learning
To obtain models using adversarial training:

```
python main_full.py --dataName mnist --train adv --ite 0 
```

### Training models with active learning

To obtain initial models before starting active learning:

```
python main_warmUp.py --dataName mnist --ite 0
```

To perform robust active learning using the random selection as the acquisition function:

```
python main_al.py --dataName mnist --train adv --metric random --ite 0
```

### Evaluate the accuracy and adversarial robustness of trained models

```
python main_evaluate.py --type al --train adv --dataName mnist --attack pgd --metric random --ite 0
```

**[Notice] Be careful with the saving path in `config.py`.**

## Reference
More experimental results can be found at our [companion site](https://sites.google.com/view/robust-al/home).

If you use this project, please consider citing us:
<pre><code>
@article{guo2022dre,
  title={DRE: density-based data selection with entropy for adversarial-robust deep learning models},
  author={Guo, Yuejun and Hu, Qiang and Cordy, Maxime and Papadakis, Michail and Le Traon, Yves},
  journal={Neural Computing and Applications},
  pages={1--18},
  year={2022},
  publisher={Springer},
  doi={10.1007/s00521-022-07812-2}
}
</code></pre>

## Contact
Please contact Yuejun Guo (yuejun.guo@uni.lu; yuejun.guo@yahoo.com) if you have further questions or want to contribute.
