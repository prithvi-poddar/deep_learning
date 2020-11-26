#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 18:46:45 2020

@author: prithvi
"""

import numpy as np
import random
import matplotlib.pyplot as plt

class sigmoid(object):
    
    @staticmethod
    def act_fn(z):
        z = np.array(z, dtype=np.float128)
        return 1.0/(1.0+np.exp(-z))
    
    @staticmethod
    def act_fn_prime(z):
        z = np.array(z, dtype=np.float128)
        return (1.0/(1.0+np.exp(-z)))*(1-(1.0/(1.0+np.exp(-z))))
    
class tanh(object):
    
    @staticmethod
    def act_fn(z):
        z = np.array(z, dtype=np.float128)
        return np.tanh(z)
    
    @staticmethod
    def act_fn_prime(z):
        z = np.array(z, dtype=np.float128)
        return 1-(np.tanh(z))**2
    
class relu(object):
    
    @staticmethod
    def act_fn(z):
        return np.maximum(0,z)
    
    @staticmethod
    def act_fn_prime(z):
        z[z<=0] = 0
        z[z>0] = 1
        return z

class QuadraticCost(object):
    
    def __init__(self, activation):
        self.activation = activation

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

    def delta(self, z, a, y):
        """Return the error delta from the output layer."""
        return (a-y) * self.activation.act_fn_prime(z)
    
class CrossEntropyCost(object):
    
    def __init__(self, activation):
        self.activation = activation

    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y*np.log(a)-(1-y)*np.log(1-a)))

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a-y)

    
class Linear_Network(object):
    
    def __init__(self, size, weight_initializer = 'default', cost = 'CrossEntropy', activation = 'sigmoid'):
        self.num_layers = len(size)
        self.size = size
        
        if weight_initializer == 'default':
            self.default_weight()
        elif weight_initializer == 'large':
            self.large_weight()
        
        if (activation == 'sigmoid'):
            self.activation = sigmoid()
        elif (activation == 'tanh'):
            self.activation = tanh()
        elif (activation == 'relu'):
            self.activation = relu()
        
        if (cost == 'CrossEntropy'):
            self.cost = CrossEntropyCost(self.activation)
            
        elif (cost == 'Quadratic'):
            self.cost = QuadraticCost(self.activation)
        
    def default_weight(self):
        self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.size[:-1], self.size[1:])]
        
    def large_weight(self):
        self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])]
        
    def feedforward(self, a, return_activations = False):
        """return_activations = False -> returns the output of the network given the weights and biases and input 'a'
        return_activations = True -> returns the activations and weighted inputs of all the layers"""
        if return_activations==False:
            for w,b in zip(self.weights, self.biases):
                a = self.activation.act_fn(np.dot(w,a)+b)
            return a
        else:
            activation = a
            activations = [a]
            zs=[]
            for w,b in zip(self.weights, self.biases):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = self.activation.act_fn(z)
                activations.append(activation)
                
            return activations, zs
        
    def SGD(self, training_data, epochs, batch_size, eta, lmbda = 0.0, test_data=None, show_eval = True, show_graph = True, return_accuracies = False):
        n = len(training_data)
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        epoch = []
        for i in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            
            for batch in batches:
                self.update_batch(batch, eta, lmbda, n)
                
            
            if test_data!=None:
                train_loss = self.evaluate_loss(training_data)
                train_accuracy = self.evaluate_train_accuracy(training_data)
                test_accuracy = self.evaluate_test_accuracy(test_data)
                if show_eval == True:
                    print ("Epoch "+str(i+1))
                    print ("Training loss: "+str(train_loss))
                    print ("Training accuracy: "+str(train_accuracy)+"%")
                    print ("Test accuracy: "+str(test_accuracy)+"%")
                    print ()
                train_losses.append(train_loss)
                train_accuracies.append(train_accuracy)
                test_accuracies.append(test_accuracy)
                epoch.append(i+1)
            else:
                print("Epoch "+str(i+1)+"complete")
                
        if test_data != None and show_graph == True:
            plt.figure()
            plt.subplot(2, 1, 1)
            self.plot_loss(epoch, train_losses)
            plt.subplot(2, 1, 2)
            self.plot_accuracies(epoch, train_accuracies, test_accuracies)

        if return_accuracies == True:
            return np.mean(train_accuracies[-5:]), np.mean(test_accuracies[-5:])

                
    def update_batch(self, batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [((1-eta*(lmbda/n))*w)-((eta/len(batch))*nw) for w, nw in zip(self.weights, nabla_w)]
        
        self.biases = [b-((eta/len(batch))*nb) for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """calculates the gradient of the cost with respect to the weights and biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations, zs = self.feedforward(x, return_activations = True)
        """first, delta of the output layer"""
        #eqn1
        delta = self.cost.delta(zs[-1], activations[-1], y) #get the cost specific delta
        #eqn 3 and 4
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation.act_fn_prime(z)
            #eqn 2
            delta = (np.dot(self.weights[-l+1].transpose(),delta))*sp 
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w
    
    def evaluate_test_accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return ((sum(int(x == y) for (x, y) in results))/len(data))*100
    
    def evaluate_train_accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        return ((sum(int(x == y) for (x, y) in results))/len(data))*100
    
    def evaluate_loss(self, data):
        losses = []
        for (x, y) in data:
            loss = np.mean(self.cost.fn(self.feedforward(x),y))
            losses.append(loss)
        return np.mean(losses)
    
    @staticmethod
    def plot_loss(epoch, losses):
        plt.plot(epoch, losses)
        plt.title("Training loss")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.show()
        
    @staticmethod
    def plot_accuracies(epoch, train_accuracies, test_accuracies):
        plt.plot(epoch, train_accuracies, label='training accuracy')
        plt.plot(epoch, test_accuracies, label='testing accuracy')
        plt.legend()
        plt.title("Training and testing accuracies")
        plt.xlabel("Epochs")
        plt.ylabel("accuracy %")
        plt.ylim(0, 100)
        plt.show()


class Linear_Network_CrossValidation(object):
    
    def __init__(self, size, weight_initializer = 'default', cost = 'CrossEntropy', activation = 'sigmoid'):
        self.num_layers = len(size)
        self.size = size
        
        if weight_initializer == 'default':
            self.default_weight()
        elif weight_initializer == 'large':
            self.large_weight()
        
        if (activation == 'sigmoid'):
            self.activation = sigmoid()
        elif (activation == 'tanh'):
            self.activation = tanh()
        elif (activation == 'relu'):
            self.activation = relu()
        
        if (cost == 'CrossEntropy'):
            self.cost = CrossEntropyCost(self.activation)
            
        elif (cost == 'Quadratic'):
            self.cost = QuadraticCost(self.activation)
        
    def default_weight(self):
        self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(self.size[:-1], self.size[1:])]
        
    def large_weight(self):
        self.biases = [np.random.randn(y, 1) for y in self.size[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(self.size[:-1], self.size[1:])]
        
    def feedforward(self, a, return_activations = False):
        """return_activations = False -> returns the output of the network given the weights and biases and input 'a'
        return_activations = True -> returns the activations and weighted inputs of all the layers"""
        if return_activations==False:
            for w,b in zip(self.weights, self.biases):
                a = self.activation.act_fn(np.dot(w,a)+b)
            return a
        else:
            activation = a
            activations = [a]
            zs=[]
            for w,b in zip(self.weights, self.biases):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = self.activation.act_fn(z)
                activations.append(activation)
                
            return activations, zs
        
    def SGD(self, training_data, epochs, k, eta, lmbda = 0.0):
        n = len(training_data)
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        epoch = []
        for i in range(epochs):
            random.shuffle(training_data)
            #Cross validation
            batch_size = int(n/k)
            
            batches = [training_data[k:k+batch_size] for k in range(0, n, batch_size)]
            tr_loss = []
            tr_acc = []
            te_acc = []
            
            for batch in batches:
                train = batch[:int(0.8*batch_size)]
                test = batch[int(0.8*batch_size):]
                self.update_batch(train, eta, lmbda, n)
                tr_loss.append(self.evaluate_loss(train))
                tr_acc.append(self.evaluate_train_accuracy(train))
                te_acc.append(self.evaluate_test_accuracy(test))
                
            train_loss = np.mean(tr_loss)
            train_accuracy = np.mean(tr_acc)
            test_accuracy = np.mean(te_acc)
            print ("Epoch "+str(i+1))
            print ("Training loss: "+str(train_loss))
            print ("Training accuracy: "+str(train_accuracy)+"%")
            print ("Test accuracy: "+str(test_accuracy)+"%")
            print ()
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)
            epoch.append(i+1)
            
                
        
        plt.figure()
        plt.subplot(2, 1, 1)
        self.plot_loss(epoch, train_losses)
        plt.subplot(2, 1, 2)
        self.plot_accuracies(epoch, train_accuracies, test_accuracies)
                
    def update_batch(self, batch, eta, lmbda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        for x, y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            
        self.weights = [((1-eta*(lmbda/n))*w)-((eta/len(batch))*nw) for w, nw in zip(self.weights, nabla_w)]
        
        self.biases = [b-((eta/len(batch))*nb) for b, nb in zip(self.biases, nabla_b)]
        
    def backprop(self, x, y):
        """calculates the gradient of the cost with respect to the weights and biases"""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activations, zs = self.feedforward(x, return_activations = True)
        """first, delta of the output layer"""
        #eqn1
        delta = self.cost.delta(zs[-1], activations[-1], y) #get the cost specific delta
        #eqn 3 and 4
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation.act_fn_prime(z)
            #eqn 2
            delta = (np.dot(self.weights[-l+1].transpose(),delta))*sp 
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return nabla_b, nabla_w
    
    def evaluate_test_accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        return ((sum(int(x == y) for (x, y) in results))/len(data))*100
    
    def evaluate_train_accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        return ((sum(int(x == y) for (x, y) in results))/len(data))*100
    
    def evaluate_loss(self, data):
        losses = []
        for (x, y) in data:
            loss = np.mean(self.cost.fn(self.feedforward(x),y))
            losses.append(loss)
        return np.mean(losses)
    
    def evaluate_validation_accuracy(self, data):
        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return ((sum(int(x == y) for (x, y) in results))/len(data))*100

    @staticmethod
    def plot_loss(epoch, losses):
        plt.plot(epoch, losses)
        plt.title("Training loss")
        plt.xlabel("Epochs")
        plt.ylabel("loss")
        plt.show()
        
    @staticmethod
    def plot_accuracies(epoch, train_accuracies, test_accuracies):
        plt.plot(epoch, train_accuracies, label='training accuracy')
        plt.plot(epoch, test_accuracies, label='testing accuracy')
        plt.legend()
        plt.title("Training and testing accuracies")
        plt.xlabel("Epochs")
        plt.ylabel("accuracy %")
        plt.ylim(0, 100)
        plt.show()
            
