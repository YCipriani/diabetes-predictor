import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

pd.set_option('display.expand_frame_repr', False)

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
diabetes_data = pd.read_csv('diabetes.csv', header=None, names=col_names)
diabetes_data = diabetes_data.iloc[1:]


feature_cols = ['pregnant', 'insulin', 'bmi', 'age', 'glucose', 'bp', 'pedigree']
X = diabetes_data[feature_cols]
y = diabetes_data.label

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.33, random_state=20)

logreg = LogisticRegression(max_iter=3000)
# fit the model with data
logreg.fit(X_train, y_train)

predictions=logreg.predict(X_test)

confusion_matrix = metrics.confusion_matrix(y_test, predictions)
print(confusion_matrix)

print("Accuracy:",metrics.accuracy_score(y_test, predictions))