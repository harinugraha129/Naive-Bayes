import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import collections

# load dataset
dataset=pd.read_csv("SeranganNormal.csv",delimiter=',')
X = dataset.iloc[:, [1, 2, 3, 4]].values
Y = dataset.iloc[:, 5].values
dataset.drop(['Time'],axis=1,inplace=True)
dataset.drop(['Length'],axis=1,inplace=True)

# membagi dataset ke dalam Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


# sampel data testing probability
class_0 = [192168106,192168102,6,58] 
print("data test 0 : ", class_0)

# menghitung jumlah data training
n_dataTraining = len(X_train)
print("n_dataTraining : ", n_dataTraining)

# menghitung jumlah data fiture
n_fiture = len(X_train[0,:])
print("n_fiture : ", n_fiture)

# Menghituna Probabilitas hanya kelas
counter_class = collections.Counter(Y_train)
print("data kelas : ", counter_class)
P_0 = counter_class[0]/n_dataTraining
P_1 = counter_class[1]/n_dataTraining

# Probabilitas pada Fiture

# inisialisasi list data kelas 0 dan kelas 1
data_0 = []
data_1 = []

# memisahkan semua fiture kelas 0 dan kelas 1
for i in range(n_dataTraining):
	if Y_train[i] == 0:
		data_0.append(X_train[i])
	elif Y_train[i] == 1:
		data_1.append(X_train[i])

# konversi fiture list ke numpy array
data_0 = np.array(data_0)
data_1 = np.array(data_1)

# Menghitung Probabilitas Kelas 0
# perulangan untuk mengakses tiap fiture
for i in range(n_fiture):
	# menampung nilai pada fiture ke i
	x = data_0[:,i]
	# menghitung jumlah data unik pada fiture ke i
	counter_x = collections.Counter(x)
	# mendapatkan probabilitas pada fiture ke i
	P_x = float(counter_x[class_0[i]]/counter_class[0])
	# mengalikan dengan probabilitas awal
	P_0 = P_0 * P_x

# dikali 100 untuk menjadi persen
P_0 = P_0 * 100
print("P_0 final : ", P_0, " %")

# Probabilitas Kelas 1
# langkahnya sama dengan probabilitas kelas 0
for i in range(n_fiture):
	x = data_1[:,i]
	counter_x = collections.Counter(x)
	P_x = float(counter_x[class_0[i]]/counter_class[1])
	P_1 = P_1 * P_x
P_1 = P_1 * 100
print("P_1 final : ", P_1, " %")

# visualisasi probabilitas data
fig = plt.figure()
ax = plt.axes()
plt.xlabel('Class', fontsize=12)
plt.ylabel('Probabilitas', fontsize=12)
plt.title("Probabilitas : "+str(class_0))
plt.plot(0, P_0, "o")
plt.plot(1, P_1, "o")
plt.show()