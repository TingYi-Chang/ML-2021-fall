import numpy as np
import math
import pandas as pd
import sys

dim = 106

def load_data():
    x_train = pd.read_csv('X_train')
    x_test = pd.read_csv('X_test')

    x_train = x_train.values
    x_test = x_test.values

    y_train = pd.read_csv('Y_train', header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)


def normalize(x_train, x_test):
    
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)
    x_all_nor = (x_all - mean) / std

    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]

    return x_train_nor, x_test_nor


def train(x_train, y_train):
    class1 = x_train[y_train == 1].T
    class2 = x_train[y_train == 0].T
    num1 = class1.shape[1]
    num2 = class2.shape[1]
    N1 = num1/(num1+num2)
    N2 = num2/(num1+num2)
    sigma1 = np.cov(class1)
    sigma2 = np.cov(class2)
    shared_sigma = sigma1 * N1 + sigma2 * N2
    mu1 = np.mean(class1, axis = 1).reshape((-1, 1))
    mu2 = np.mean(class2, axis = 1).reshape((-1, 1))
    #print (mu1.shape, mu2.shape) 
    return mu1, mu2, shared_sigma, N1, N2


def predict(x_test, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2).T, sigma_inverse)
    b = (-0.5) * np.dot(np.dot(mu1.T, sigma_inverse), mu1) + (0.5) * np.dot(np.dot(mu2.T, sigma_inverse), mu2) + np.log(float(N1)/N2)

    z = np.dot(w, x_test.T) + b
    pred = sigmoid(z)
    
    return pred

def dump(y_test):
    import csv
    with open(sys.argv[1], 'w', newline='') as csvf:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvf)
        writer.writerow(['id','label'])
        for i in range(len(y_test)):
            writer.writerow( [i + 1, int(y_test[i])] )

def gaussian(x , mean , inv_covariance):
	return np.exp(-0.5 * (((x - mean).T @ inv_covariance) @ (x - mean)))

def accuracy(x , y , prior_probability_1 , prior_probability_2 , mean_1 , mean_2 , covariance):
	inv_covariance = np.linalg.inv(covariance)

	count = 0
	number_of_data = x.shape[0]
	for i in range(number_of_data):
		probability_1 = prior_probability_1 * gaussian(x[i].reshape((-1 , 1)) , mean_1 , inv_covariance)
		probability_2 = prior_probability_2 * gaussian(x[i].reshape((-1 , 1)) , mean_2 , inv_covariance)
		if ((probability_1 > probability_2 and y[i] == 1) or (probability_1 < probability_2 and y[i] == 0)):
			count += 1

	return count / number_of_data

def main():
    x_train, y_train, x_test = load_data()
    x_train, x_test = normalize(x_train, x_test)
    mu1, mu2, shared_sigma, N1, N2 = train(x_train, y_train)
    print (mu1.shape, mu2.shape, shared_sigma.shape)

    y = predict(x_train, mu1, mu2, shared_sigma, N1, N2)
    y = np.around(y)
    result = (y_train == y)
    y_test = predict(x_test, mu1, mu2, shared_sigma, N1, N2)
    
    y_test = np.around(y_test)
    dump(y_test.T)

    return


if (__name__ == '__main__'):
	main()

x_train,y_train,x_test = load_data()





# TODO: predict x_test


