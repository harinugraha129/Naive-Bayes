import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Mengimpor dataset
dataset=pd.read_csv("SeranganNormal.csv",delimiter=',')
X = dataset.iloc[:, [1, 2]].values
Y = dataset.iloc[:, 5].values

dataset.drop(['Time'],axis=1,inplace=True)
dataset.drop(['Length'],axis=1,inplace=True)

# Menjadi dataset ke dalam Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

# Probabilitas Data
prob_train_0 = []
prob_train_1 = []
range_train = int(len(X_train))
deret_0 = []
deret_1 = []
fiture_1 = list(X_train[:,0])
fiture_2 = list(X_train[:,1])

for i in range(range_train):
	# for j in range(int(len(X_train))):
	temp1 = fiture_1.count(X_train[i][0])
	temp2 = fiture_2.count(X_train[i][1])
	temp_prob = (temp1/range_train)*(temp2/range_train)
	
	if Y_train[i]==0:
		prob_train_0.append(temp_prob)
		deret_0.append(i)
	else:
		prob_train_1.append(temp_prob)
		deret_1.append(i)

plt.plot(deret_0, prob_train_0, color="chocolate", label="0")
plt.plot(deret_1, prob_train_1, color="green", label="1")
plt.legend(loc=(1,0))
plt.title('Probabilitas data (Training set)')
plt.xlabel('Data List')
plt.ylabel('Probabilitas')
plt.show()

prob_test_0 = []
prob_test_1 = []
range_test = int(len(X_test))
deret_0 = []
deret_1 = []
fiture_1 = list(X_test[:,0])
fiture_2 = list(X_test[:,1])

for i in range(range_test):
	# for j in range(int(len(X_train))):
	temp1 = fiture_1.count(X_test[i][0])
	temp2 = fiture_2.count(X_test[i][1])
	temp_prob = (temp1/range_test)*(temp2/range_test)
	
	if Y_test[i]==0:
		prob_test_0.append(temp_prob)
		deret_0.append(i)
	else:
		prob_test_1.append(temp_prob)
		deret_1.append(i)

plt.plot(deret_0, prob_test_0, color="chocolate", label="0")
plt.plot(deret_1, prob_test_1, color="green", label="1")
plt.legend(loc=(1,0))
plt.title('Probabilitas data (Testing set)')
plt.xlabel('Data List')
plt.ylabel('Probabilitas')
plt.show()

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Membuat model Naive Bayes terhadap Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

GaussianNB(priors=None, var_smoothing=1e-09)

# Memprediksi hasil test set
y_pred = classifier.predict(X_test)

# Membuat confusion matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
acc = accuracy_score(Y_test, y_pred)
pre = precision_score(Y_test, y_pred)
rec = recall_score(Y_test, y_pred)
f1s = f1_score(Y_test, y_pred)


print("Accuracy : ", acc)
print("Presisi : ", pre)
print("Recall : ", rec)
print("F1 Score : ", f1s)


# Visualisasi hasil model Naive Bayes dari Training set
from matplotlib.colors import ListedColormap
X_set, Y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('Source/Destination')
plt.ylabel('Label')
plt.legend()
plt.show()

from matplotlib.colors import ListedColormap
X_set, Y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01), np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(Y_set)):
    plt.scatter(X_set[Y_set == j, 0], X_set[Y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('Source/Destination')
plt.ylabel('Label')
plt.legend()
plt.show()

# Drawing Class Chart
from collections import Counter

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct
# Data to plot data train
labels = Counter(Y_train).keys() # equals to list(set(words))
sizes = Counter(Y_train).values() # counts the elements
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# print(str(labels))
# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct=make_autopct(sizes), shadow=True, startangle=140)

plt.axis('equal')
plt.title("Perbandingan Kelas Data Training")
plt.show()

# Data to plot data testing
labels = Counter(Y_test).keys() # equals to list(set(words))
sizes = Counter(Y_test).values() # counts the elements
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# print(str(labels))
# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct=make_autopct(sizes), shadow=True, startangle=140)

plt.axis('equal')
plt.title("Perbandingan Kelas Data Testing")
plt.show()


# Data to plot kelas Prediksi
labels = Counter(y_pred).keys() # equals to list(set(words))
sizes = Counter(y_pred).values() # counts the elements
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# print(str(labels))
# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct=make_autopct(sizes), shadow=True, startangle=140)

plt.axis('equal')
plt.title("Perbandingan Kelas Hasil Prediksi")
plt.show()


