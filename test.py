import matplotlib.pyplot as plt

# Data to plot
labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [250, 250, 350, 150]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
# explode = (0.1, 0, 0, 0)  # explode 1st slice

def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct

# Plot
plt.pie(sizes, labels=labels, colors=colors,
autopct=make_autopct(sizes), shadow=True, startangle=140)


plt.axis('equal')
plt.show()
from collections import Counter
a = Counter(sizes).keys() # equals to list(set(words))
b = Counter(sizes).values() # counts the elements' frequency  # length of the list stored at `'key'
print(str(a))
print(str(b))

# from sklearn import svm, datasets
# from sklearn.model_selection import train_test_split
# import numpy as np

# iris = datasets.load_iris()
# X = iris.data
# y = iris.target

# # Add noisy features
# random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

# # Limit to the two first classes, and split into training and test
# X_train, X_test, y_train, y_test = train_test_split(X[y < 2], y[y < 2],
#                                                     test_size=.5,
#                                                     random_state=random_state)

# # Create a simple classifier
# classifier = svm.LinearSVC(random_state=random_state)
# classifier.fit(X_train, y_train)
# y_score = classifier.decision_function(X_test)

# from sklearn.metrics import average_precision_score
# average_precision = average_precision_score(y_test, y_score)

# print('Average precision-recall score: {0:0.2f}'.format(
#       average_precision))



# # Plot the Precision-Recall curve
# from sklearn.metrics import precision_recall_curve
# from sklearn.metrics import plot_precision_recall_curve
# import matplotlib.pyplot as plt

# disp = plot_precision_recall_curve(classifier, X_test, y_test)
# disp.ax_.set_title('2-class Precision-Recall curve: '
#                    'AP={0:0.2f}'.format(average_precision))
# plt.show()