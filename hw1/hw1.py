import numpy as np
import csv
import math
from numpy.lib.twodim_base import tril_indices
import pandas as pd
import sys

def read_data(feats, data_name):
  data = pd.read_csv(data_name) #讀入讀入training file
  for col in list(data.columns[2:]):
      data[col] = data[col].astype(str).map(lambda x: x.rstrip('x*#A'))
  data = data.values
  train_data = np.transpose(np.array(np.float64(data)))
  train_x, train_y = parse2train(train_data, feats)

  """
  # Find out the valid data
  unique, counts = np.unique(train_data[17,:], return_counts=True)
  print(dict(zip(unique, counts))); 
  #
  """
  train_x = np.array(train_x)
  train_y = np.array(train_y)
  
  valid_x = train_x[ : 1000]
  valid_y = train_y[ : 1000]
  train_x = train_x[1000 : ]
  train_y = train_y[1000 : ]
  return (train_x , train_y , valid_x , valid_y)

def valid(x, y,feats_num):
  # TODO: try to filter out extreme values ex:PM2.5 > 100
  for i in range(9):
    for j in feats_num:
      if (feats_num == 0):
        if (x[j,i] < 0):
          return False
      if (feats_num == 1):
        if (x[j,i] < 0 or x[j,i] > 3):
          return False
      if (feats_num == 2):
        if (x[j,i] < 0 or x[j,i] > 8):
          return False
      if (feats_num == 3):
        if (x[j,i] < 0 or x[j,i] > 1.5):
          return False
      if (feats_num == 4):
        if (x[j,i] < 0 or x[j,i] > 50):
          return False
      if (feats_num == 5):
        if (x[j,i] < 0 or x[j,i] > 100):
          return False
      if (feats_num == 6):
        if (x[j,i] < 0 or x[j,i] > 150):
          return False
      if (feats_num == 7):
        if (x[j,i] < 0 or x[j,i] > 200):
          return False
      if (feats_num == 8):
        if (x[j,i] < 0 or x[j,i] > 500):
          return False
      if (feats_num == 9):
        if (x[j,i] < 0):
          return False
      if (feats_num == 10):
        if (x[j,i] < 0):
          return False
      if (feats_num == 11):
        if (x[j,i] < 0 or x[j,i] > 50):
          return False
      if (feats_num == 12):
        if (x[j,i] < 0 or x[j,i] > 30):
          return False
      if (feats_num == 13):
        if (x[j,i] < 0):
          return False
      if (feats_num == 14):
        if (x[j,i] < 0):
          return False
      if (feats_num == 15):
        if (x[j,i] < 0):
          return False
      if (feats_num == 16):
        if (x[j,i] < 0):
          return False
      if (feats_num == 17):
        if (x[j,i] < 0 or x[j,i] > 150):
          return False
  if (y < 0 or y > 150):
    return False
  return True

def parse2train(data, feats):
  x = []
  y = []

  # 用前面9筆資料預測下一筆PM2.5 所以需要-9
  total_length = data.shape[1] - 9
  for i in range(total_length):
    x_tmp = data[feats,i:i+9]
    y_tmp = data[17,i+9] #第第18個feature是是PM2.5
    # TODO: try to filter out extreme values
    
    if valid(x_tmp, y_tmp,feats):
      x.append(x_tmp.reshape(-1,))
      y.append(y_tmp)
    """
    x.append(x_tmp.reshape(-1,))
    y.append(y_tmp)
    """
  # x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
  x = np.array(x)
  y = np.array(y)
  return x,y

def minibatch(x, y):
    # 打亂data順序
    index = np.arange(x.shape[0])
    np.random.shuffle(index)
    x = x[index]
    y = y[index]
    
    # 訓練參數以及初始化
    batch_size = 32
    lr = 1e-8
    lam = 0.001
    beta_1 = np.full(x[0].shape, 0.9).reshape(-1, 1)
    beta_2 = np.full(x[0].shape, 0.99).reshape(-1, 1)
    w = np.full(x[0].shape, 0.1).reshape(-1, 1)
    bias = 0.1
    m_t = np.full(x[0].shape, 0).reshape(-1, 1)
    v_t = np.full(x[0].shape, 0).reshape(-1, 1)
    m_t_b = 0.0
    v_t_b = 0.0
    t = 0
    epsilon = 1e-8
    
    for num in range(4500):
        for b in range(int(x.shape[0]/batch_size)):
            t+=1
            x_batch = x[b*batch_size:(b+1)*batch_size]
            y_batch = y[b*batch_size:(b+1)*batch_size].reshape(-1,1)
            loss = y_batch - np.dot(x_batch,w) - bias
            
            # 計算gradient
            g_t = np.dot(x_batch.transpose(),loss) * (-2) +  2 * lam * np.sum(w)
            g_t_b = loss.sum(axis=0) * (2)
            m_t = beta_1*m_t + (1-beta_1)*g_t 
            v_t = beta_2*v_t + (1-beta_2)*np.multiply(g_t, g_t)
            m_cap = m_t/(1-(beta_1**t))
            v_cap = v_t/(1-(beta_2**t))
            m_t_b = 0.9*m_t_b + (1-0.9)*g_t_b
            v_t_b = 0.99*v_t_b + (1-0.99)*(g_t_b*g_t_b) 
            m_cap_b = m_t_b/(1-(0.9**t))
            v_cap_b = v_t_b/(1-(0.99**t))
            w_0 = np.copy(w)
            
            # 更新weight, bias
            w -= ((lr*m_cap)/(np.sqrt(v_cap)+epsilon)).reshape(-1, 1)
            bias -= (lr*m_cap_b)/(math.sqrt(v_cap_b)+epsilon)
            

    return w, bias

def RMSE(x, y, weight, bias, data_type):
  RMSE = np.sqrt(np.mean(((x @ weight + bias) - y.reshape((-1 , 1)))**2))
  print(data_type,' RMSE :', RMSE)
  return

def parse2test(data, feats):
  x = []
  y = []

  # 用前面9筆資料預測下一筆PM2.5 所以需要-9
  total_length = data.shape[1] - 9
  for i in range(857):
    x_tmp = data[feats,9 * i: 9 * i + 9]
    x.append(x_tmp.reshape(-1,))
  # x 會是一個(n, 18, 9)的陣列， y 則是(n, 1) 
  x = np.array(x)
  return x

def predict(feats, w, bias, test_name, sol_name):
  print (test_name)
  data = pd.read_csv(test_name)
  data = data.values
  test_data = np.transpose(np.array(np.float64(data)))
  print (test_data.shape)
  test_x = parse2test(test_data, feats)
  with open(sol_name, 'w', newline='') as csvf:
    writer = csv.writer(csvf)
    writer.writerow(['Id','Predicted'])
    for i in range(int(test_x.shape[0])):
        writer.writerow([i + 1,(np.dot(np.reshape(w,-1),test_x[i]) + bias)[0]])
  return

def save_model(w, bias, model_name):
	model = np.hstack((w.reshape(-1) , bias))
	np.save(model_name , model)
	return

def main():
  feats = [1,2,3,17]
  (train_x, train_y, valid_x, valid_y) = read_data(feats, sys.argv[1])
  (w , bias) = minibatch(train_x, train_y)
  RMSE(train_x, train_y, w, bias, 'train')
  RMSE(valid_x, valid_y, w , bias, 'valid')
  #save_model(w, bias, sys.argv[4])
  predict(feats, w, bias, sys.argv[2], sys.argv[3])

if (__name__ == '__main__'):
  main()