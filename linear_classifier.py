import os
import sys
import numpy as np
from PIL import Image

train_path = sys.argv[1]
test_path = sys.argv[2]

class_count={}
class_ind ={}
images = []
labels = []
image_label=[]
fp = open(train_path)
train = fp.readlines()
cnt = 0
for each in train:
	tt = each.split(' ')
	tt[1] = tt[1].split('\n')
	clas = tt[1][0]
	if clas not in class_count.keys():
		class_count[clas] = 0
		class_ind[clas]=[]
		labels.append(clas)
	image_label.append(clas)	
	class_ind[clas].append(cnt)	
	class_count[clas]=class_count[clas]+1
	image = Image.open(tt[0]).convert('L')
	image = image.resize((32,32), Image.ANTIALIAS)
	array = np.array(image)
	images.append(array.flatten())
	cnt = cnt + 1

dim = (len(labels),cnt)
t = np.zeros(dim)

for i in range(len(image_label)):
	ind = labels.index(image_label[i])
	t[ind][i] = 1




images = np.array(images)

mean = np.mean(images,axis=0)
image_mean =images - mean



cov_image = np.matmul(image_mean.T,image_mean)
image_lam, image_vec = np.linalg.eigh(cov_image)



indi = np.argsort(image_lam)


image_vec = image_vec[:,np.flip(indi,axis=0)]



N = 32

n_vec = image_vec[:,0:N]
n_data = np.matmul(n_vec.T,images.T)



weit_mat = np.zeros((len(labels),32))

for i in range(10000):

	prob = np.matmul(weit_mat,n_data)
	prob_exp = np.exp(prob-np.max(prob,axis=0))
	exp_sum = np.sum(prob_exp,axis=0)
	w_x = prob_exp/exp_sum

	w_x = t - w_x
	
	grad = np.matmul(w_x,n_data.T)
	weit_mat += (0.1) * grad 











fp1 = open(test_path)

test = fp1.readlines()
for each in test:
	each = each.split('\n')
	test_image = Image.open(each[0]).convert('L')
	test_image = test_image.resize((32,32), Image.ANTIALIAS)
	array = np.array(test_image)
	array = array.flatten()
	

	test_data = np.matmul(n_vec.T,array.T)
	test_data = test_data.T


	prob = np.matmul(weit_mat,test_data.T)


	ind = np.argmax(prob)

	print(labels[ind])

