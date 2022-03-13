import numpy as np
from numpy import abs
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv("train_data.csv")
x_train, y_train = train_df['x_train'], train_df['y_train']

class Linear_Regression:
    def __init__(self, X, Y):
        self.__X = X
        self.__Y = Y
        self.__w =  # rand

    def train(self, lr=1e-6):
        self.__lr = lr

        prev_loss = 1e6
        while 1:

            self.__w[0] = self.__w[0] - (learning_rate * ((1 / m) *
                                                      np.sum(Y_pred - Y)))

            self.__w[1] = self.__w[1] - (learning_rate * ((1 / m) *
                                                      np.sum((Y_pred - Y) * self.X)))
            cur_loss = self.__Loss_func()

            if abs(prev_loss -cur_loss) < 1e-10:    # until convergence (Loss)
                break

            prev_loss = cur_loss

    def predict(self, x_test):

    def __Loss_func(self, ):
        pass

    def __grad():
        pass


# Main program
model = Linear_Regression(x_train, y_train)



