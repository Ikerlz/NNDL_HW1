#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/4/30 15:54 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : utils.py
# @desc :

import math
import os
import gzip
import numpy as np


# activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - x ** 2


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# learning rate decay

def time_based_decay(initial_lr, epoch, decay_rate):
   return initial_lr / (1 + decay_rate * epoch)

def step_decay(initial_lr, epoch, drop, epochs_drop):
   return initial_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))

def exponential_decay(initial_lr, epoch, decay_rate):
   return initial_lr * math.exp(-decay_rate * epoch)

def cosine_decay(initial_lr, epoch, max_epochs):
   return 0.5 * initial_lr * (1 + math.cos(math.pi * epoch / max_epochs))


# Loss function for classification
def cross_entropy_loss(y_pred, y_true, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)  # To avoid the log(0)
    return - np.mean(y_true * np.log(y_pred))

# data loader

def load_mnist(path, kind='train'):

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

# To onehot code
def one_hot(classes, y):
    encode = np.eye(classes)[y]
    return encode


# functions for training

def calc_acc(out, y):
    labels = np.argmax(y, axis=1)
    preds = np.argmax(out, axis=1)
    acc = np.mean(labels == preds)
    return acc

# for parsing
def parse_list_of_lists(input_string):
    elements = input_string.split(',')
    return [parse_list(x) for x in elements]

def parse_list(input_string, val_name="hidden"):
    elements = input_string.split(' ')
    if val_name in ['lr_cand', 'reg_cand']:
        return [float(x) for x in elements]
    elif val_name in ['activation_cand', 'hidden_cand']:
        return elements
    else:
        return [int(x) for x in elements]
