import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
###
df = pd.read_csv("C:/Users/M/Desktop/abbas/learning me/MakTab Khoone/Data Sets/heart.csv")
data = df[['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh',
            'exng', 'oldpeak', 'slp', 'caa', 'thall']]
###
from sklearn.preprocessing import StandardScaler
sclaer = StandardScaler().fit(data)
data = sclaer.transform(data)
###
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(data,df['output'])
###
# loss func
def loss(y_pre,y_test):
    correct_pre  = 0
    for i in range(len(y_pre)):
        if y_pre[i] == y_test[i]:
            correct_pre += 1
    print('model Accuracy ===>' , 100*correct_pre/len(y_pre) , "%")
###
# Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators =400,               # Number of trees
                                max_depth=5,                     # The maximum depth of the tree
                                min_weight_fraction_leaf=0.01,   # The minimum weighted fraction of the sum total of weights
                                max_features=5                   # The maximum of features that will be used in each tree
                        
)
forest.fit(X_train,Y_train)

# find loss
Y_pre5 = forest.predict(X_test)
loss(Y_pre5,Y_test.values)