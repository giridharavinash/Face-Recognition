import os
import sys
import numpy as np
from PIL import Image

train_path = sys.argv[1]
test_path = sys.argv[2]

class_count={}
class_mean = {}
class_var = {}
class_ind ={}
images = []
labels = []
fp = open(train_path)
train = fp.readlines()
cnt = 0
for each in train:
	tt = each.split(' ')
	tt[1] = tt[1].split('\n')
	clas = tt[1][0]
	if clas not in class_count.keys():
		class_count[clas] = 0
		class_mean[clas]=[]
		class_var[clas]=[]
		class_ind[clas]=[]
		labels.append(clas)
	class_ind[clas].append(cnt)	
	class_count[clas]=class_count[clas]+1
	image = Image.open(tt[0]).convert('L')
	image = image.resize((32,32), Image.ANTIALIAS)
	array = np.array(image)
	images.append(array.flatten())
	cnt = cnt + 1

images = np.array(images)

image_mean =images - np.mean(images,axis=0)
# print(len(images))



cov_image = np.dot(image_mean.T,image_mean)
image_lam, image_vec = np.linalg.eig(cov_image)

# print(image_vec)

indi = np.argsort(-image_lam)


image_vec = image_vec[:,indi]

# sorted_vec = image_vec[:,np.flip(indi,axis=0)]


N = 32

n_vec = image_vec[:,0:N]
n_data = np.matmul(n_vec.T,images.T)
n_data = n_data.T

for i in labels:
	dattt = np.array(n_data[class_ind[i],:])
	class_mean[i].append(np.mean(dattt,axis=0))
	class_var[i].append(np.var(dattt,axis=0))
	
	
	
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
	
	max_p = 0
	ans = ''
	for clas in labels: 
		prob  = 0
		mean = class_mean[clas][0]	
		var  = class_var[clas][0]
		

		for j in range(len(test_data)):
			val = 1/np.sqrt(2*np.pi*var[j])
			prob = prob+(np.exp(-abs(np.square(test_data[j]-mean[j])/var[j])))*val	
		# prob = prob * class_count[clas]
		if prob > max_p:
			max_p = prob 
			ans = clas
	print(ans)		



		