import numpy as np

ary = np.mat([[1,2,3],
              [4,8,5],
              [3,12,9],
              [1,8,5],
              [5,14,2],
              [7,4,1],
              [9,8,9],
              [3,8,1],
              [11,5,6],
              [10,11,7]
              ]).T
# (a)
print ("(a)")
mean = np.mean(ary,axis=1)
sigma = np.zeros((3,3))
for i in range(10):
    sigma += np.dot((ary[:,i]-mean),((ary[:,i]-mean)).T)
sigma /= 10

eigenValues,eigenVectors=np.linalg.eig(sigma)
idx = eigenValues.argsort()[::-1]
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]
print("V1: ",eigenVectors[:,0], "\nV2:", eigenVectors[:,1],\
     "\nV3:", eigenVectors[:,2])

#(b)
print ("\n\n(b)")
W = eigenVectors.T
for i in range(10):
    Wx = np.dot(W,ary[:,i])
    print ("sample",i+1,":\n",Wx)

# (c)
print("\n\n(c)")
W1 = W[0:2]
Err = 0
for i in range(10):
    W1x = np.dot(W,ary[:,i])
    Err += (np.linalg.norm(ary[:,i]- np.dot(W1.T,np.dot(W1,ary[:,i]))))**2
print ("Err:",Err/10)


