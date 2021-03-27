import numpy as np
import pickle
import matplotlib.pyplot as plt


def plotCurve(res_fn, fig_fn):
    save_res = pickle.load(open(res_fn, 'rb'))

    dev_epoch_loss = save_res['dev_epoch_loss']
    train_epoch_loss = save_res['train_epoch_loss']
    dev_acc = save_res['dev_acc']
    train_acc = save_res['train_acc']

    train_argmin_loss = np.argmin(train_epoch_loss)
    train_min_loss = train_epoch_loss[train_argmin_loss]
    dev_argmin_loss = np.argmin(dev_epoch_loss)
    dev_min_loss = dev_epoch_loss[dev_argmin_loss]


    train_argmax_acc = np.argmax(train_acc)
    train_max_acc = train_acc[train_argmax_acc]
    dev_argmax_acc = np.argmax(dev_acc)
    dev_max_acc = dev_acc[dev_argmax_acc]

    fig = plt.figure(figsize=(20, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(range(len(dev_epoch_loss)), dev_epoch_loss, c='red', label='dev')
    ax1.plot(range(len(train_epoch_loss)), train_epoch_loss, c='blue', label='train')
    ax1.plot(dev_argmin_loss, dev_min_loss, 'ks', c='red')
    ax1.plot(train_argmin_loss, train_min_loss, 'ks', c='blue')
    ax1.annotate(text="{:.4f}".format(train_min_loss), xy=(train_argmin_loss, train_min_loss))
    ax1.annotate(text="{:.4f}".format(dev_min_loss), xy=(dev_argmin_loss, dev_min_loss))
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('Epoch Average Loss')
    ax1.legend(loc='best')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(range(len(dev_acc)), dev_acc, c='red', label='dev')
    ax2.plot(range(len(train_acc)), train_acc, c='blue', label='train')
    ax2.plot(dev_argmax_acc, dev_max_acc, 'ks', c='red')
    ax2.plot(train_argmax_acc, train_max_acc, 'ks', c='blue')
    ax2.annotate(text="{:.2f}".format(train_max_acc), xy=(train_argmax_acc, train_max_acc))
    ax2.annotate(text="{:.2f}".format(dev_max_acc), xy=(dev_argmax_acc, dev_max_acc))
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('Epoch Accuracy')
    ax2.legend(loc='best')

    plt.savefig(fig_fn)
    plt.show()

if __name__ == '__main__':
    plotCurve('./res/SGD_norm_linear.pkl', './fig/SGD_norm_linear.png')