import os
import sys
import numpy as np 
from skimage.io import imread, imsave

IMAGE_PATH = 'Aberdeen'

# Images for compression & reconstruction
test_image = ['7.jpg','119.jpg','138.jpg','157.jpg','333.jpg'] 

# Number of principal components used

k = 5

# Image Processing
def process(M): 
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

#Load Data
filelist = os.listdir(IMAGE_PATH) 

# Record the shape of images
img_shape = imread(os.path.join(IMAGE_PATH,filelist[0])).shape 

img_data = []
for filename in filelist:
    tmp = imread(os.path.join(IMAGE_PATH,filename))  
    img_data.append(tmp.flatten())
training_data = np.array(img_data).astype('float32')
training_data = training_data.astype(np.float64)

mean = np.mean(training_data, axis = 0).astype(np.float64)
training_data -= mean 

# Report Problem 1.a
mean = process(mean)
imsave('mean.jpg', mean.reshape(img_shape))  

# Use SVD to find the eigenvectors 
U, S, V = np.linalg.svd(training_data.T, full_matrices = False)
print ("SVD calculated")
print ("U shape:", U.shape)
print ("S shape:", S.shape)
print ("V shape:", V.shape)



#Report Problem 3

for i in range(5):
    eigenface = process(-U[:,i]).reshape(img_shape)
    imsave(str(i) + '_eigenface.jpg', eigenface, quality=100)

for i in range(5):
    number = S[i] * 100 / sum(S)
    print(i, ' Eigenfaces: ',number, round(number, 1))

#Report Problem 4
for idx in range(len(test_image)): 
    # Load image & Normalize
    print("Picture:",idx)
    picked_img = imread(os.path.join(IMAGE_PATH,test_image[idx]))
    X = picked_img.flatten().astype('float32')
    X -= mean   

    # Compression
    weight = np.dot(X, U)
    
    # Reconstruction
    recon = np.zeros(img_shape[0]*img_shape[1]*img_shape[2])
    for i in range(5):
        layer = weight[i] * U[:,i]
        recon += layer
    reconstruct = process(recon+mean)
    imsave(test_image[idx][:-4] + '_reconstruction.jpg', reconstruct.reshape(img_shape))