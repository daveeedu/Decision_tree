import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from sklearn import tree
import matplotlib.pyplot as plt

col_names = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']
Iris = pd.read_csv("Iris.csv",)
Iris['Species'].unique()
label_encoder = preprocessing.LabelEncoder()
Iris['Species']=label_encoder.fit_transform(Iris['Species'])
Iris['Species'].unique()

print(Iris)
feature_cols = ['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x = Iris[feature_cols]
y = Iris.Species

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=1)
clf = DecisionTreeClassifier().fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
tree.plot_tree(clf)
plt.show()












