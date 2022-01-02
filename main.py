import remi.gui as gui
from remi import start, App
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn import svm
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import recall_score, accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.combine import SMOTEENN
from sklearn.feature_selection import SelectKBest
from collections import Counter
import warnings

from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits

warnings.filterwarnings('ignore')
import matplotlib.ticker as mtick

data = pd.read_csv("data.csv")
tdata = data.copy()
data.drop(columns=['customerID'])
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
categorical_feature = {
    feature
    for feature in data.columns if data[feature].dtypes == 'O'
}
encoder = LabelEncoder()
for feature in categorical_feature:
    data[feature] = encoder.fit_transform(data[feature])
print(encoder.classes_)
data.TotalCharges = data.TotalCharges.fillna(data.TotalCharges.mean())
X_train = data.drop(['Churn'], axis=1)
y_train = data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    random_state=42)

# knn
knn = KNeighborsClassifier(n_neighbors=3, p=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accu_knn = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print("the size of the knn is:")
print(knn.__sizeof__())

for i in data.columns:
    print(i + str(data[i][0]))

# Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accu_gnb = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print("the size of the gnb is:")
print(gnb.__sizeof__())



# Support Vector Machine
svc = svm.SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
accu_svc = accuracy_score(y_test, y_pred)
print(accuracy_score(y_test, y_pred))
print("the size of the svc is:")
print(svc.__sizeof__())

# decisionTree Classifier
Dtc = DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_leaf=15)
Dtc.fit(X_train, y_train)
dtc_pred = Dtc.predict(X_test)

print(f'Accuracy score : {accuracy_score(dtc_pred, y_test)}')
print(f'Confusion matrix :\n {confusion_matrix(dtc_pred, y_test)}')
print(f'Classification report :\n {classification_report(dtc_pred, y_test)}')

# Random forest classifier
Rfc = RandomForestClassifier(n_estimators=120,criterion='gini', max_depth=15, min_samples_leaf=10, min_samples_split=5)
Rfc.fit(X_train, y_train)
rfc_pred = Rfc.predict(X_test)

print(f'Accuracy score : {accuracy_score(rfc_pred, y_test)}')
print(f'Confusion matrix :\n {confusion_matrix(rfc_pred, y_test)}')
print(f'Classification report :\n {classification_report(rfc_pred, y_test)}')

# logistic regression
Log_reg = LogisticRegression(C=150, max_iter=150)
Log_reg.fit(X_train, y_train)
log_pred = Log_reg.predict(X_test)

print(f'Accuracy score : {accuracy_score(log_pred, y_test)}')
print(f'Confusion matrix :\n {confusion_matrix(log_pred, y_test)}')
print(f'Classification report :\n {classification_report(log_pred, y_test)}')

rfc=RandomForestClassifier(n_estimators=120,criterion='gini', max_depth=15, min_samples_leaf=10, min_samples_split=5)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

print(f'Accuracy score : {accuracy_score(rfc_pred, y_test)}')
print(f'Confusion matrix :\n {confusion_matrix(rfc_pred, y_test)}')
print(f'Classification report :\n {classification_report(rfc_pred, y_test)}')

Dtc = DecisionTreeClassifier(criterion='gini', splitter='random', min_samples_leaf=15)
Dtc.fit(X_train, y_train)
dtc_pred = Dtc.predict(X_test)

print(f'Accuracy score : {accuracy_score(dtc_pred, y_test)}')
print(f'Confusion matrix :\n {confusion_matrix(dtc_pred, y_test)}')
print(f'Classification report :\n {classification_report(dtc_pred, y_test)}')
# print("the accuracy score of knn is", accu_knn)
# print("the accuracy score of gnb is", accu_gnb)
# print("the accuracy score of svc is", accu_svc)


class main_container(App):
    qustlist = [["Female", "Male"], ["1", "0"], ["Yes", "No"], ["Yes", "No"],
                ["0", "1"], ["Yes", "No"], ["Yes", "No", "No phone service"],
                ["DSL", "Fiber optic", "No"],
                ["Yes", "No", "No internet service"],
                ["Yes", "No", "No internet service"],
                ["Yes", "No", "No internet service"],
                ["Yes", "No", "No internet service"],
                ["Yes", "No", "No internet service"],
                ["Yes", "No", "No internet service"],
                ["Month-to-month", "One year", "Two year"], ["Yes", "No"],
                [
                    "Electronic check", "Credit card (automatic)",
                    "Bank transfer (automatic)", "Mailed check"
                ], ["0", "1"], ["0", "1"]]

    labellist = [
        "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges"
    ]

    def __init__(self, *args):
        super(main_container, self).__init__(*args)
        self.title = "test"

    def main(self):
        mainContainer = gui.VBox(width=300,
                                 height=2000,
                                 style={
                                     'margin': '0px auto',
                                     'padding': '0px'
                                 })
        self.btn = gui.Button('Submit', width=200, height=30)
        self.tmp = []
        self.tmplabel = []
        for i in range(len(self.qustlist)):
            if (self.labellist[i] == "tenure"
                    or self.labellist[i] == "MonthlyCharges"
                    or self.labellist[i] == "TotalCharges"):
                self.tmp.append(gui.TextInput(width=200, height=30))
            else:
                self.tmp.append(
                    gui.DropDown(self.qustlist[i], width=200, height=30))
            self.tmplabel.append(
                gui.Label(self.labellist[i], width=200, height=30))
        for i in range(len(self.qustlist)):
            mainContainer.append(self.tmplabel[i])
            mainContainer.append(self.tmp[i])
        self.ansknn = gui.Label("", width=200, height=30)
        self.ansbayes= gui.Label("", width=200, height=30)
        self.anssvm = gui.Label("", width=200, height=30)
        self.ansdtc = gui.Label("", width=200, height=30)
        self.ansrfc = gui.Label("", width=200, height=30)
        mainContainer.append(self.btn)
        # mainContainer.append(self.ans)
        mainContainer.append(self.ansknn)
        mainContainer.append(self.ansbayes)
        mainContainer.append(self.anssvm)
        mainContainer.append(self.ansdtc)
        mainContainer.append(self.ansrfc)
        self.btn.onclick.do(self.on_button_pressed)
        return mainContainer

    def on_button_pressed(self, widget):
        global encoder
        test_dd = ["5375"]
        for i in range(len(self.qustlist)):
            if (self.tmp[i].get_value() == None
                    or self.tmp[i].get_value() == ""):
                test_dd.append(self.qustlist[i][0])
            else:
                test_dd.append(self.tmp[i].get_value())
        test_dd = np.array(test_dd)
        test_dd = test_dd.reshape(1, -1)
        test_dd = pd.DataFrame(test_dd)
        test_dd.columns = X_train.columns
        encoder = LabelEncoder()
        for feature in categorical_feature:
            if (feature == "Churn" or feature == "customerID"
                    or feature == "tenure" or feature == "MonthlyCharges"
                    or feature == "TotalCharges"):
                continue
            else:
                encoder.fit(tdata[feature])
                test_dd[feature] = encoder.transform(test_dd[feature])
        # for i in test_dd.columns:
        #     print(i + str(test_dd[i][0]))
        # print(i + " : " + str(test_dd[i]))
        Y_pred_knn = pd.DataFrame(knn.predict(test_dd))
        Y_pred_bayes = pd.DataFrame(knn.predict(test_dd))
        Y_pred_svm = pd.DataFrame(knn.predict(test_dd))
        Y_pred_rfc = pd.DataFrame(knn.predict(test_dd))
        Y_pred_dtc = pd.DataFrame(knn.predict(test_dd))
        if Y_pred_knn[0][0] == 1:
            self.ansknn.set_text("knn:Yes")
        else:
            self.ansknn.set_text("knn:No")
        if Y_pred_bayes[0][0] == 1:
            self.ansbayes.set_text("bayes:Yes")
        else:
            self.ansbayes.set_text("bayes:No")
        if Y_pred_svm[0][0] == 1:
            self.anssvm.set_text("svm:Yes")
        else:
            self.anssvm.set_text("svm:No")
        if Y_pred_rfc[0][0] == 1:
            self.ansrfc.set_text("rfc:Yes")
        else:
            self.ansrfc.set_text("rfc:No")
        if Y_pred_dtc[0][0] == 1:
            self.ansdtc.set_text("dtc:Yes")
        else:
            self.ansdtc.set_text("dtc:No")
        print(gnb.__sizeof__())
        print(svc.__sizeof__())
        print(gnb.__sizeof__())
        # print(Y_pred_knn)
        # print(Y_pred_bayes)
        # print(Y_pred_svm)
start(main_container)
