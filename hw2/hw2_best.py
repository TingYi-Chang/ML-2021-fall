import numpy as np
import pandas as pd
import sys

def load_data():
    x_train = pd.read_csv('X_train')
    x_test = pd.read_csv('X_test')
    
    x_train.drop('fnlwgt', axis=1, inplace=True)
    x_test.drop('fnlwgt', axis=1, inplace=True)
    x_train = x_train.values
    x_test = x_test.values
    
    for i in [0, 2, 3, 4]:
        print (x_train[0,i])
        for j in range(2, 5):
            x_train = np.hstack((x_train, (x_train[:,i]**j).reshape((-1 , 1))))
    for i in [0, 2, 3, 4]:
        for j in range(2, 5):
            x_test = np.hstack((x_test , (x_test[:,i]**j).reshape((-1 , 1))))
    
    y_train = pd.read_csv('Y_train', header = None)
    y_train = y_train.values
    y_train = y_train.reshape(-1)

    return x_train, y_train, x_test

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def normalize(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis = 0)
    x_max = np.max(x_all, axis = 0)
    x_min = np.min(x_all, axis = 0)
    x_all_nor = (x_all-x_min)/(x_max-x_min)
    x_train_nor = x_all_nor[0:x_train.shape[0]]
    x_test_nor = x_all_nor[x_train.shape[0]:]
    return x_train_nor, x_test_nor

def standardization(x_train, x_test):
    x_all = np.concatenate((x_train, x_test), axis = 0)
    mean = np.mean(x_all, axis = 0)
    std = np.std(x_all, axis = 0)
    x_all_std = (x_all - mean) / std

    x_train_std = x_all_std[0:x_train.shape[0]]
    x_test_std = x_all_std[x_train.shape[0]:]
    return (x_train_std, x_test_std)

def train(x_train, y_train):
    w = np.zeros(x_train.shape[1])
    b = 0.0

    epoch = 1000
    lr = 0.1
    b_lr = 0
    w_lr = np.ones(x_train.shape[1])
    
    for e in range(epoch):
        z = np.dot(x_train , w) + b
        pred = sigmoid(z)
        loss = y_train - pred

		# Calculate gradient.
        w_grad = -1 * np.dot(loss , x_train)
        b_grad = -1 * np.sum(loss)

        # Update w and b.
        w_lr += w_grad**2
        b_lr += b_grad**2
        b -= lr / np.sqrt(b_lr) * b_grad
        w -= lr / np.sqrt(w_lr) * w_grad
        
        # Calculate loss and accuracy.
        loss = -1 * np.mean(y_train * np.log(pred + 1e-100) + (1 - y_train) * np.log(1 - pred + 1e-100))
        train_accuracy = accuracy(x_train , y_train , w , b)
        # validation_accuracy = accuracy(validation_x , validation_y , w , b)
        #print ("train accuracy: ", train_accuracy, "validation accuracy:" ,validation_accuracy)
        print ("num: ", e, ", train accuracy: ", train_accuracy)
    return w, b

def accuracy(x , y , weight , bias):
	count = 0
	number_of_data = x.shape[0]
	for i in range(number_of_data):
		probability = sigmoid(weight @ x[i] + bias)
		if ((probability > 0.5 and y[i] == 1) or (probability < 0.5 and y[i] == 0)):
			count += 1
	return count / number_of_data

def main():
    x_train, y_train, x_test = load_data()
    x_train, x_test = standardization(x_train, x_test)
    w, b = train(x_train, y_train)
    predict(x_test, w, b)

# TODO: predict x_test
def predict(x_test, w, b):
    y_test = list()
    number_of_data = x_test.shape[0]
    for i in range(number_of_data):
        y_test.append(1 if (sigmoid(w @ x_test[i] + b) > 0.5) else 0)
    import csv
    with open(sys.argv[1], 'w', newline='') as csvf:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvf)
        writer.writerow(['id','label'])
        for i in range(len(y_test)):
            writer.writerow( [i + 1, int(y_test[i])] )

if (__name__ == '__main__'):
	main()