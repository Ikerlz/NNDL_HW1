#!/usr/bin/env python
# _*_ coding: utf-8 _*_
# @Time : 2024/4/30 16:39 
# @Author : Iker Zhe 
# @Version：V 0.1
# @File : model.py
# @desc :

from utils import *


class ThreeLayerNeuralNetwork:
    def __init__(self, input_size=28*28, hidden_sizes=None, output_size=10, activation_function='relu'):
        if hidden_sizes is None:
            hidden_sizes = [20, 10, 5]
        self.activation_functions = {
            'relu': (relu, relu_derivative),
            'sigmoid': (sigmoid, sigmoid_derivative),
            'tanh': (tanh, tanh_derivative),
            'leaky-relu': (leaky_relu, leaky_relu_derivative)
        }
        self.activate_type = activation_function
        self.activation_function, self.activation_derivative = self.activation_functions[self.activate_type]

        self.weights1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2. / input_size)
        self.bias1 = np.zeros(hidden_sizes[0])
        self.weights2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2. / hidden_sizes[0])
        self.bias2 = np.zeros(hidden_sizes[1])
        self.weights3 = np.random.randn(hidden_sizes[1], hidden_sizes[2]) * np.sqrt(2. / hidden_sizes[1])
        self.bias3 = np.zeros(hidden_sizes[2])
        self.weights4 = np.random.randn(hidden_sizes[2], output_size) * np.sqrt(2. / hidden_sizes[2])
        self.bias4 = np.zeros(output_size)

        self.params = {
            'W1': self.weights1,
            'b1': self.bias1,
            'W2': self.weights2,
            'b2': self.bias2,
            'W3': self.weights3,
            'b3': self.bias3,
            'W4': self.weights4,
            'b4': self.bias4
        }


    def softmax(self, h):
        exp_h = np.exp(h - np.max(h, axis=-1, keepdims=True))
        return exp_h / np.sum(exp_h, axis=-1, keepdims=True)

    def forward_pass(self, X):
        self.layer1 = self.activation_function(np.dot(X, self.weights1) + self.bias1)
        self.layer2 = self.activation_function(np.dot(self.layer1, self.weights2) + self.bias2)
        self.layer3 = self.activation_function(np.dot(self.layer2, self.weights3) + self.bias3)
        output = self.softmax(np.dot(self.layer3, self.weights4) + self.bias4)
        return output

    def backpropagate(self, X, y, output, learning_rate, lambda_L2):
        batch_num = X.shape[0]

        # Error in output
        error = (output - y) / batch_num

        # Calculate deltas
        delta4 = error
        d_weights4 = np.dot(self.layer3.T, delta4) + lambda_L2 * self.weights4
        d_bias4 = np.sum(delta4, axis=0)

        delta3 = np.dot(delta4, self.weights4.T) * self.activation_derivative(self.layer3)
        d_weights3 = np.dot(self.layer2.T, delta3) + lambda_L2 * self.weights3
        d_bias3 = np.sum(delta3, axis=0)

        delta2 = np.dot(delta3, self.weights3.T) * self.activation_derivative(self.layer2)
        d_weights2 = np.dot(self.layer1.T, delta2) + lambda_L2 * self.weights2
        d_bias2 = np.sum(delta2, axis=0)

        delta1 = np.dot(delta2, self.weights2.T) * self.activation_derivative(self.layer1)
        # delta1 = np.dot(delta2, self.weights2.T) * self.layer1
        d_weights1 = np.dot(X.T, delta1) + lambda_L2 * self.weights1
        d_bias1 = np.sum(delta1, axis=0)

        # print(d_weights1)

        # Update weights and biases
        self.weights4 -= learning_rate * d_weights4
        self.bias4 -= learning_rate * d_bias4

        self.weights3 -= learning_rate * d_weights3
        self.bias3 -= learning_rate * d_bias3

        self.weights2 -= learning_rate * d_weights2
        self.bias2 -= learning_rate * d_bias2

        self.weights1 -= learning_rate * d_weights1
        self.bias1 -= learning_rate * d_bias1

    def save_model(self, path):
        save_dict = {
            'model_name': 'ThreeLayerNet',
            'params': self.params,
            'activation_function': self.activate_type
        }
        np.save(path, save_dict)

    def load_model(self, path):
        to_fit = np.load(path, allow_pickle=True).item()
        self.activation_function, self.activation_derivative = self.activation_functions[to_fit['activation_function']]
        self.params = to_fit['params']
        self.weights1 = self.params['W1']
        self.weights2 = self.params['W2']
        self.weights3 = self.params['W3']
        self.weights4 = self.params['W4']
        self.bias1 = self.params['b1']
        self.bias2 = self.params['b2']
        self.bias3 = self.params['b3']
        self.bias4 = self.params['b4']

