import numpy as np
import pandas as pd

from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white")  # white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)

import warnings

warnings.simplefilter(action='ignore')

train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

train_data = train_df.copy()

train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)  # substitui NaN Ages por a média das idades
train_data["Embarked"].fillna(train_df["Embarked"].value_counts().idxmax(), inplace=True)  # substitui NaN Embarked
# por valor com mais frequencia
train_data.drop('Cabin', axis=1, inplace=True)  # remove rows que tem NaN Cabin

train_data["TravelAlone"] = np.where((train_data["SibSp"] + train_data["Parch"]) > 0, 0, 1)  # a pessoa estava sozinha
# sibsp Number of Siblings/Spouses Aboard
# parch Number of Parents/Children Aboard
train_data.drop("SibSp", axis=1, inplace=True)  # remove SibSp
train_data.drop("Parch", axis=1, inplace=True)  # remove Parch

training = pd.get_dummies(train_data, columns=["Pclass", "Embarked", "Sex"])
training.drop("Sex_female", axis=1, inplace=True)  # remove Sex_female
training.drop('PassengerId', axis=1, inplace=True)  # remove PassengerId
training.drop('Name', axis=1, inplace=True)  # remove Name
training.drop('Ticket', axis=1, inplace=True)  # remove Ticket

training["IsMinor"] = np.where(train_df["Age"] <= 16, 1, 0)  # é menor

# data test
test_data = test_df.copy()

test_data["Age"].fillna(test_df["Age"].median(skipna=True), inplace=True)
test_data["Embarked"].fillna(test_df["Embarked"].value_counts().idxmax(), inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)
test_data["TravelAlone"] = np.where((test_data["SibSp"] + test_data["Parch"]) > 0, 0, 1)
test_data.drop("SibSp", axis=1, inplace=True)
test_data.drop("Parch", axis=1, inplace=True)

testing = pd.get_dummies(test_data, columns=["Pclass", "Embarked", "Sex"])
testing.drop("Sex_female", axis=1, inplace=True)  # remove Sex_female
testing.drop('PassengerId', axis=1, inplace=True)  # remove PassengerId
testing.drop('Name', axis=1, inplace=True)  # remove Name
testing.drop('Ticket', axis=1, inplace=True)  # remove Ticket

testing["IsMinor"] = np.where(test_df["Age"] <= 16, 1, 0) # é menor
#

# logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score

cols = ["Age",
        "Fare",
        "TravelAlone",
        "Pclass_1",
        "Pclass_2",
        "Embarked_C",
        "Embarked_S",
        "Sex_male",
        "IsMinor"
        ]

X = training[cols]
y = training["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

print(f'Size X_train {len(X_train)}')
print(f'Size X_test {len(X_test)}')
print("-"*10, "Test", "-"*10)
print(f'Size y_train {len(y_train)}')
print(f'Size y_test {len(y_test)}')

model = LogisticRegression()  # modelo de regressao logistico
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f'Accuracy Score: {accuracy_score(y_test, y_pred)*100}')  # Pontuação do acerto
print(f'Precision Score: {precision_score(y_test, y_pred)*100}')  # Pontuação da precisão
print(f'Log Loss: {log_loss(y_test, y_pred_proba)*100}')  # Pontuação da perca
