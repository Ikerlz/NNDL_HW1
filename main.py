#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/4/30 15:54
# @Author : Iker Zhe
# @Version：V 0.1
# @File : utils.py
# @desc :
from model import ThreeLayerNeuralNetwork
import argparse
from utils import *
import pandas as pd


def para_find(lr_cand, hidden_cand, reg_cand, batch_size_cand, activation_cand, data_path='', epochs=100, lr_decay_rate=0.99, lr_strategy='step_decay', save_path='./model_save', early_stopping=True, patience=5):
    test_acc = 0
    best_para = {'lr': None, 'hidden': None, 'reg': None, 'batch_size': None, 'activation': None}

    df = pd.DataFrame(columns=['lr', 'hidden', 'reg', 'batch_size', 'activation', 'acc'])

    for lr in lr_cand:
        for hidden in hidden_cand:
            for reg in reg_cand:
                for batch_size in batch_size_cand:
                    for activation in activation_cand:
                        net = ThreeLayerNeuralNetwork(input_size=28 * 28, hidden_sizes=hidden, output_size=10,
                                                       activation_function=activation)
                        print(f'Testing the accuracy with lr {lr}, hidden {hidden}, reg {reg}, batch_size {batch_size}, activation {activation}')
                        best_tmp = train(net=net, data_path=data_path, epochs=epochs, batch_size=batch_size, initial_lr=lr, lr_decay_rate=lr_decay_rate, lambda_L2=reg, task='para_find', lr_strategy=lr_strategy, save_path=save_path, early_stopping=early_stopping, patience=patience)
                        df = pd.concat([df, pd.DataFrame({'lr': [lr], 'hidden': [hidden], 'reg': [reg], 'batch_size': [batch_size], 'activation': [activation], 'acc': [best_tmp]})], ignore_index=True)
                        df.to_csv(os.path.join(save_path, 'acc_res.csv'), index=False)
                        if best_tmp > test_acc:
                            test_acc = best_tmp
                            best_para['lr'] = lr
                            best_para['hidden'] = hidden
                            best_para['reg'] = reg
                            best_para['batch_size'] = batch_size
                            best_para['activation'] = activation



    return best_para

def train(net, data_path='', epochs=100, batch_size=128, initial_lr=1e-3, lr_decay_rate=0.99, lambda_L2=0.1, task='train', lr_strategy='step_decay', save_path='./model_save', early_stopping=True, patience=5):
    x_train, t_train = load_mnist(data_path, kind='train')
    x_test, t_test = load_mnist(data_path, kind='t10k')
    # TODO: Normalization
    x_train = x_train / 255
    x_test = x_test / 255
    classes = len(np.unique(t_train))
    idx = list(range(x_train.shape[0]))
    np.random.shuffle(idx)
    train_loss, train_acc = [], []
    test_loss, test_acc = [], []

    best_acc = 0
    best_epoch = 0
    no_improvement_count = 0

    # check the save path

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for e in range(epochs):
        loss_e = []
        acc_e = []

        if lr_strategy == 'time_based_decay':
            lr = time_based_decay(initial_lr=initial_lr, epoch=e, decay_rate=lr_decay_rate)
        elif lr_strategy == 'step_decay':
            lr = step_decay(initial_lr=initial_lr, epoch=e, drop=lr_decay_rate, epochs_drop=20)
        elif lr_strategy == 'exponential_decay':
            lr = exponential_decay(initial_lr=initial_lr, epoch=e, decay_rate=lr_decay_rate)
        elif lr_strategy == 'cosine_decay':
            lr = cosine_decay(initial_lr=initial_lr, epoch=e, max_epochs=epochs)
        else:
            raise ValueError(
                "The lr_strategy should be time_based_decay, step_decay, exponential_decay or the cosine_decay, but {} is given".format(
                    lr_strategy))

        # training dataset

        for i in range(0, x_train.shape[0], batch_size):
            X = x_train[idx[i:i + batch_size]]
            y = one_hot(classes, t_train[idx[i:i + batch_size]])
            out = net.forward_pass(X)
            loss_e.append(cross_entropy_loss(out, y))
            acc_e.append(calc_acc(out, y))
            net.backpropagate(X, y, out, lr, lambda_L2)

        train_loss.append(np.mean(loss_e))
        train_acc.append(np.mean(acc_e))

        # testing dataset

        test_out = net.forward_pass(x_test)
        test_y = one_hot(classes, t_test)
        test_loss.append(cross_entropy_loss(test_out, test_y))
        test_acc.append(calc_acc(test_out, test_y))
        if task == 'train':
            print(f'---------epoch {e + 1}---------')
            print(f'train loss {train_loss[-1]}')
            print(f'test loss {test_loss[-1]}')
            print(f'train acc {train_acc[-1]}')
            print(f'test acc {test_acc[-1]}')

        if task == 'train':

            # Save current epoch model parameters
            net.save_model(os.path.join(save_path, f'model_epoch_{e}.npy'))
            if e > 0:
                os.remove(os.path.join(save_path, f'model_epoch_{e-1}.npy'))

            # Check for early stopping
            if early_stopping:
                if test_acc[-1] > best_acc:
                    net.save_model(os.path.join(save_path, f'best_model_epoch_{e}.npy'))
                    # Delete previous best model parameters
                    if e > 0:
                        os.remove(os.path.join(save_path, f'best_model_epoch_{best_epoch}.npy'))
                    best_acc = test_acc[-1]
                    best_epoch = e
                    no_improvement_count = 0
                    # Save current best model parameters
                    # net.save_model(os.path.join(save_path, 'model_best.npy'))
                else:
                    no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f'Early stopping at epoch {e}, best epoch: {best_epoch}')
                    break
        elif task == 'para_find':
            if early_stopping:
                if test_acc[-1] > best_acc:
                    best_acc = test_acc[-1]
                    best_epoch = e
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                if no_improvement_count >= patience:
                    print(f'Early stopping at epoch {e}, best epoch: {best_epoch}')
                    break
        else:
            raise ValueError("The task should be train or para_find but {} is given".format(task))
    if task == 'train':
        loss_acc = pd.DataFrame({
            "TrainLoss": train_loss,
            "TestLoss": test_loss,
            "TrainAcc": train_acc,
            "TestAcc": test_acc
        })
        loss_acc.to_csv(os.path.join(save_path, "epochs{}_batch_size{}_lambdaL2{}.csv".format(epochs, batch_size, lambda_L2)), index=False)

    elif task == 'para_find':
        return best_acc


def test(test_data_path='', model_weight_path=''):
    x_test, t_test = load_mnist(test_data_path, kind='t10k')
    x_test = x_test / 255
    test_net = ThreeLayerNeuralNetwork()
    test_net.load_model(path=model_weight_path)
    test_out = test_net.forward_pass(x_test)
    test_y = one_hot(10, t_test)
    test_loss = cross_entropy_loss(test_out, test_y)
    test_acc = calc_acc(test_out, test_y)
    print("===== The Testing Accuracy is {} and the Testing Loss is {}! =====".format(np.round(test_acc, 4), np.round(test_loss, 4)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters for running the model.')

    parser.add_argument('--task', type=str, default='test', choices=['train', 'test', 'param_find'],
                        help='Task to be executed (choices: train, test, param_find)')

    # for testing
    parser.add_argument('--data_path', type=str, default='', help='Path to data')
    parser.add_argument('--model_weight_path', type=str, default='', help='Path to the trained model weights')

    # for training
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden', type=str, default="20, 10, 5", help="The sizes of the hidden layers")
    parser.add_argument('--activation', type=str, default='relu', help="The activation function")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--initial_lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.99, help='Learning rate decay rate')
    parser.add_argument('--lambda_L2', type=float, default=0.1, help='L2 regularization parameter')
    parser.add_argument('--lr_strategy', type=str, choices=['step_decay', 'time_based_decay', 'exponential_decay', 'cosine_decay'], default='step_decay',
                        help='Learning rate strategy')
    parser.add_argument('--save_path', type=str, default='./train_model_save', help='Path to save model')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')

    # for param_find
    parser.add_argument('--lr_cand', type=str, default=[0.01, 0.001], help="The candidate list of the initial learning rate")
    parser.add_argument('--hidden_cand', type=str, default=[[20, 20, 5], [5, 10, 20]], help="The candidate list of the sizes of the hidden layers")
    parser.add_argument('--reg_cand', type=str, default=[0.01, 0.001],
                        help="The candidate list of the tuning parameter for the L2 regularization")
    parser.add_argument('--batch_size_cand', type=str, default=[128, 256], help="The candidate list of the batch size")
    parser.add_argument('--activation_cand', type=str, default=['relu', 'tanh'], help="The candidate list of the activation functions")

    args = parser.parse_args()


    if args.task == 'test':
        test(test_data_path=args.data_path, model_weight_path=args.model_weight_path)
    elif args.task == 'train':
        hidden = parse_list(args.hidden)
        net = ThreeLayerNeuralNetwork(input_size=28 * 28, hidden_sizes=hidden, output_size=10, activation_function=args.activation)
        train(net, data_path=args.data_path, epochs=args.epochs, batch_size=args.batch_size, initial_lr=args.initial_lr,
              lr_decay_rate=args.lr_decay_rate, lambda_L2=args.lambda_L2, task='train', lr_strategy=args.lr_strategy,
              save_path=args.save_path, early_stopping=args.early_stopping, patience=args.patience)
    else:
        hidden_cand = parse_list_of_lists(args.hidden_cand)
        batch_size_cand = parse_list(args.batch_size_cand, val_name='batch_size_cand')
        activation_cand = parse_list(args.activation_cand, val_name='activation_cand')
        lr_cand = parse_list(args.lr_cand, val_name='lr_cand')
        reg_cand = parse_list(args.reg_cand, val_name='reg_cand')
        para_find(lr_cand=lr_cand, hidden_cand=hidden_cand, reg_cand=reg_cand,
                  batch_size_cand=batch_size_cand, activation_cand=activation_cand,
                  data_path=args.data_path, epochs=args.epochs, lr_decay_rate=args.lr_decay_rate,
                  lr_strategy=args.lr_strategy, save_path=args.save_path,
                  early_stopping=args.early_stopping, patience=args.patience)



