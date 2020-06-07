import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()



# dataset=pd.read_csv("test_data.csv",delimiter=',')
# X = dataset.iloc[:, [0, 1]].values
# Y = dataset.iloc[:, 2].values
# # classifier.fit(X, Y)

# Mengimpor dataset
dataset=pd.read_csv("SeranganNormal.csv",delimiter=',')
X = dataset.iloc[:, [1, 2]].values
Y = dataset.iloc[:, 5].values

dataset.drop(['Time'],axis=1,inplace=True)
dataset.drop(['Length'],axis=1,inplace=True)

# Menjadi dataset ke dalam Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

n_dataset = int(len(Y))
print("jumlah Dataset : ", n_dataset)

data_0x = []
data_0y = []
data_1x = []
data_1y = []
for i in range(int(len(X))):
	# print(i)
	if Y[i] == 0:
		data_0x.append(X[i,0])
		data_0y.append(X[i,1])
	elif Y[i] == 1:
		data_1x.append(X[i,0])
		data_1y.append(X[i,1])


counter_fiture_x = collections.Counter(X[:,0])
counter_fiture_y = collections.Counter(X[:,1])
counter_class = collections.Counter(Y)
print("data fiture x : ", counter_fiture_x)
print("data fiture y : ", counter_fiture_y)
print("data kelas : ", counter_class)

# Probabilitas Kelas
print("\nProbabilitas kelas")
P_0 = counter_class[0]/n_dataset
P_1 = counter_class[1]/n_dataset
print("P_0 : ", P_0)
print("P_1 : ", P_1)

# Probabilitas Fiture kelas 0
print("\nProbabilitas Fiture kelas 0")
P_0x = 0.0
counter_data_0x=collections.Counter(data_0x)
# print("data fiture 0x : ", data_0x)
print("data fiture 0x : ", counter_data_0x)
for fit in counter_data_0x:
	print("class_fiture : ", fit)
	temp_x = (counter_data_0x[fit]/n_dataset)
	print(temp_x)
	P_0x += (temp_x)

print("P_0x : ", P_0x)

P_0y = 0.0
counter_data_0y=collections.Counter(data_0y)
# print("data fiture 0y : ", data_0y)
print("data fiture 0y : ", counter_data_0y)
for fit in counter_data_0y:
	print("class_fiture : ", fit)
	temp_y = (counter_data_0y[fit]/n_dataset)
	print(temp_y)
	P_0y += (temp_y)

print("P_0y : ", P_0y)



# Probabilitas Fiture kelas 1
print("\nProbabilitas Fiture kelas 1")
P_1x = 0.0
counter_data_1x=collections.Counter(data_1x)
# print("data fiture 1x : ", data_1x)
print("data fiture 1x : ", counter_data_1x)
for fit in counter_data_1x:
	print("class_fiture : ", fit)
	temp_x = (counter_data_0x[fit]/n_dataset)
	print(temp_x)
	P_1x += (temp_x)

print("P_1x : ", P_1x)

P_1y = 0.0
counter_data_1y=collections.Counter(data_1y)
# print("data fiture 1y : ", data_1y)
print("data fiture 1y : ", counter_data_1y)
for fit in counter_data_1y:
	print("class_fiture : ", fit)
	temp_y = (counter_data_1y[fit]/n_dataset)
	print(temp_y)
	P_1y += (temp_y)

print("P_1y : ", P_1y)

# Probabilitas Total
P_0_final = P_0x * P_0y/2 
P_1_final = P_1x * P_1y/2 

print("Probabilitas class 0 : ", P_0_final)
print("Probabilitas class 1 : ", P_1_final)

label = ['Class 0', 'Class 1']
index = np.arange(len(label))
print(index)
plt.bar(index, [P_0_final, P_1_final])
plt.xlabel('Class', fontsize=12)
plt.ylabel('Probabilitas', fontsize=12)
plt.xticks(index, label, fontsize=12)
plt.title('Probabilitas Dataset')
plt.show()