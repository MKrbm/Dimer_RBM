from ..abstract_optimizer import AbstractOptimizer
import numpy as np


class Sgd(AbstractOptimizer):
    def __init__(self, learning_rate, l2reg=0, decay_factor=1.0, N = 100):

        self._learning_rate = learning_rate
        self._l2reg = l2reg
        self._decay_factor = decay_factor
        self._eta = learning_rate

        self.N = N

        self.iter = 0

        self.linear_decay = False

        self.sigmoid_decay = False

        if self._decay_factor == 'sigmoid decay':
            self.sigmoid_decay = True

        if learning_rate <= 0:
            raise ValueError("Invalid learning rate.")
        if l2reg < 0:
            raise ValueError("Invalid L2 regularization.")
        # if decay_factor < 1:
        #     raise ValueError("Invalid decay factor.")

    def update(self, grad, pars):
        
        if self.sigmoid_decay:
            self.iter += 1
            self._eta = self._learning_rate * self.sigmoid(self.iter)
            print('eta = ', self._eta)

        else:
            self._eta *= self._decay_factor
        # print(self._eta)
        pars -= (grad + self._l2reg * pars) * self._eta
        return pars

    def reset(self):
        self._eta = self._learning_rate
    
    def sigmoid(self,x):

        return 1/(1 + np.exp((x-self.N/2)/(self.N/10)))

    def __repr__(self):
        if self._learning_rate != self._eta:
            lr = "learning_rate[base]={}, learning_rate[current]={}".format(
                self._learning_rate, self._eta
            )
        else:
            lr = "learning_rate={}".format(self._learning_rate)
        return ("Sgd({}, l2reg={}, decay_factor={})").format(
            lr, self._l2reg, self._decay_factor
        )
