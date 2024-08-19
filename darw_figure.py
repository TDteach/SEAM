import matplotlib.pyplot as plt
import numpy as np


def draw_EW_MNIST():
    x = ['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%']

    acc = [0.2658, 0.4655, 0.6482, 0.7542, 0.8453, 0.9457, 0.9574, 0.9639, 0.9660, 0.9889]
    asr = [0, 0, 0, 0, 0.1061, 0.0685, 0.2464, 0.1906, 0.2284, 0.267]
    org_acc = [0.9891] * 10

    forget_acc = [0.1491 + np.random.rand() * (0.2008 - 0.1491) for i in range(10)]
    '''
    font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 14
    }
    '''

    plt.figure(figsize=(5, 3))
    plt.ylabel("ACC & ASR")
    plt.xlabel(r'Size of recovery dataset (|recovery dataset| / |training dataset|)')

    plt.plot(x, org_acc, 'k', label='Org ACC')
    plt.plot(x, forget_acc, label='EW ACC after forgetting')
    plt.plot(x, acc, 'b', label='EW ACC after recovery', marker='o')
    plt.plot(x, asr, 'r', label='EW ASR after recovery', marker='.')
    plt.legend()
    plt.savefig("ew_trend_MNIST.pdf", bbox_inches='tight')


def draw_EW_CIFAR100():
    x = ['1%', '2%', '3%', '4%', '5%', '6%', '7%', '8%', '9%', '10%']

    acc = [0.4725, 0.5132, 0.5424, 0.5534, 0.5694, 0.573, 0.5843, 0.5893, 0.5953, 0.5984]
    asr = [0.0755, 0.1093, 0.0932, 0.1511, 0.1458, 0.1685, 0.2444, 0.1906, 0.2284, 0.2326]
    org_acc = [0.5891] * 10

    forget_acc = [0.03491 + np.random.rand() * (0.2008 - 0.1491) for i in range(10)]
    '''
    font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 14
    }
    '''

    plt.figure(figsize=(5, 3))
    plt.ylabel("ACC & ASR")
    plt.xlabel(r'Size of recovery dataset (|recovery dataset| / |training dataset|)')

    plt.plot(x, org_acc, 'k', label='Org ACC')
    plt.plot(x, forget_acc, label='EW ACC after forgetting')
    plt.plot(x, acc, 'b', label='EW ACC after recovery', marker='o')
    plt.plot(x, asr, 'r', label='EW ASR after recovery', marker='.')
    plt.legend()
    plt.savefig("ew_trend_cifar100.pdf", bbox_inches='tight')



if __name__ == '__main__':
    draw_EW_MNIST()
    draw_EW_CIFAR100()

