# @akjgskwi
import numpy as np
import argparse
import chainer
import random
import datetime
import os
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
from chainer import serializers, iterators, optimizers, Variable
from chainer.dataset import concat_examples
from chainer.datasets import mnist
from chainer.cuda import to_cpu


class MLP(chainer.Chain):

    def __init__(self, n_units=100, n_out=10):
        super(MLP, self).__init__(
            l1=L.Linear(None, n_units),
            l2=L.Linear(n_units, n_units),
            l3=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(seed)


reset_seed(0)
model = MLP()
gpu_id = -1

# parser settings
parser = argparse.ArgumentParser(description='Chainer MNIST Training')
parser.add_argument('--batchsize', default=128, type=int,
                    help='batch size(default: 128)')
parser.add_argument('--epoch', default=100, type=int,
                    help='total epochs to run(default: 100)')
parser.add_argument('--alpha', default=1., type=float,
                    help='mixup interpolation coefficient (default: 1)')
parser.add_argument('--decay', default=1e-4, type=float,
                    help='weight decay(default: 1e-4)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='learning rate(default: 0.1)')
args = parser.parse_args()


def mixup_data(x, t, alpha):
    '''Returns mixed inputs, pairs of labels, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.shape[0]
    idx = np.random.permutation(batch_size)
    mixed_x = lam * x + (1 - lam) * x[idx]
    t_a, t_b = t, t[idx]
    return mixed_x, t_a, t_b, lam


def mixup_criterion(pred, t_a, t_b, lam):
    val = lam * F.softmax_cross_entropy(pred, t_a) \
            + (1 - lam) * F.softmax_cross_entropy(pred, t_b)
    return val


def adjust_lr(optimizer, epoch):
    """scheduling learning rate"""
    lr = args.lr
    if epoch >= 30:
        lr /= 10
    if epoch >= 60:
        lr /= 10
    optimizer.lr = lr


def train(model):
    dir_path = 'mnist_mixup_' + str(datetime.date.today())
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    train, test = mnist.get_mnist(withlabel=True, ndim=1)
    train_iter = iterators.SerialIterator(train, args.batchsize, shuffle=False)
    test_iter = iterators.SerialIterator(test, args.batchsize,
                                         repeat=False, shuffle=False)

    optimizer = optimizers.MomentumSGD(lr=args.lr, momentum=0.9)
    optimizer.use_cleargrads()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(rate=args.decay))
    result = []
    train_losses = []
    while train_iter.epoch < args.epoch:
        train_batch = train_iter.next()
        x, t = concat_examples(train_batch, gpu_id)
        x, t_a, t_b, lam = mixup_data(x, t, args.alpha)
        x = Variable(x)
        t_a = Variable(t_a)
        t_b = Variable(t_b)
        y = model(x)
        loss = mixup_criterion(y, t_a, t_b, lam)
        train_losses.append(to_cpu(loss.data))

        model.cleargrads()
        loss.backward()
        optimizer.update()
        if train_iter.is_new_epoch:
            if train_iter.epoch % 5 == 0:
                print('epoch:{: 02d} train_loss:{: .04f}'.format(
                    train_iter.epoch, float(np.mean(train_losses))), end=' ')

            adjust_lr(optimizer, train_iter.epoch)
            test_losses = []
            test_accuracies = []
            while True:
                test_batch = test_iter.next()
                x_test, t_test = concat_examples(test_batch, gpu_id)

                y_test = model(x_test)

                loss_test = F.softmax_cross_entropy(y_test, t_test)
                test_losses.append(to_cpu(loss_test.data))

                accuracy = F.accuracy(y_test, t_test)
                accuracy.to_cpu()
                test_accuracies.append(accuracy.data)

                if test_iter.is_new_epoch:
                    test_iter.epoch = 0
                    test_iter.current_position = 0
                    test_iter.is_new_epoch = False
                    test_iter._pushed_position = None
                    break
            if train_iter.epoch % 5 == 0:
                print('val_loss:{: .04f} val_accuracy:{: .04f}'.format(
                    np.mean(test_losses), np.mean(test_accuracies)))
            result.append([train_iter.epoch,
                          float(np.mean(train_losses)), np.mean(test_losses)])
            serializers.save_npz('mnist_mixup_' + str(datetime.date.today()) +
                                 '/epoch-' + str(train_iter.epoch), model)

            train_losses.clear()

    result = np.asarray(result)
    epoch = result[:, 0]
    tr_loss = result[:, 1]
    te_loss = result[:, 2]

    plt.plot(epoch, tr_loss, color='blue', label="main/loss", marker="x")
    plt.plot(epoch, te_loss,
             color='orange', label="validation/main/loss", marker="x")
    plt.legend()
    plt.savefig("mnist_mixup_" + str(datetime.date.today()) + "/loss.png")


train(model)
