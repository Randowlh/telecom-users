import remi.gui as gui
from remi import start, App
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import scipy.stats as stats
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
warnings.filterwarnings('ignore')
import matplotlib.ticker as mtick # for showing percentage in it
data = pd.read_csv("data.csv")
tdata=data.copy()
data.drop(columns=['customerID'])
# transform data
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
# print(f'Count of Numerical feature: {len(numerical_feature)}')
# print(f'Numerical feature are:\n {numerical_feature}')
categorical_feature = {feature for feature in data.columns if data[feature].dtypes == 'O'}
# print(f'Count of Categorical feature: {len(categorical_feature)}')
# print(f'Categorical feature are:\n {categorical_feature}')

# print(data);
encoder = LabelEncoder()
# fit encoder with data
for feature in categorical_feature:
    data[feature] = encoder.fit_transform(data[feature])
print(encoder.classes_)
# print(data);
# start to train with knn
data.TotalCharges = data.TotalCharges.fillna(data.TotalCharges.mean())
X_train=data.drop(['Churn'],axis=1)
y_train=data['Churn']
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
knn=KNeighborsClassifier(n_neighbors=3,p=2) 
# print(y_train)
# print(X_train)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test,y_pred))
print("the size of the knn is:")
print(knn.__sizeof__())
for i in data.columns:
    print(i+str(data[i][0]));
class main_container(App):
    qustlist =[["Female","Male"],["1","0"],["Yes","No"],["Yes","No"],["0","1"],["Yes","No"],["Yes","No","No phone service"],["DSL","Fiber optic","No"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Month-to-month","One year","Two year"],["Yes","No"],["Electronic check","Credit card (automatic)","Bank transfer (automatic)","Mailed check"],["0","1"],["0","1"]]

    labellist = ["gender","SeniorCitizen","Partner","Dependents","tenure",	"PhoneService",	"MultipleLines",    "InternetService",	"OnlineSecurity","OnlineBackup","DeviceProtection",	"TechSupport",	"StreamingTV",	"StreamingMovies",	"Contract",	"PaperlessBilling",	"PaymentMethod","MonthlyCharges",	"TotalCharges"]

    def __init__(self, *args):
        super(main_container, self).__init__(*args)
        self.title = "test"
    def main(self):
        # create a container VBox type, with vertical orientation
        mainContainer = gui.VBox(width=300, height=2000, style={'margin':'0px auto', 'padding':'0px'})
        self.btn = gui.Button('Submit', width=200, height=30)
        self.tmp=[]
        self.tmplabel=[]
        for i in range(len(self.qustlist)):
            if(self.labellist[i]=="tenure" or self.labellist[i]=="MonthlyCharges" or self.labellist[i]=="TotalCharges"):
                self.tmp.append(gui.TextInput(width=200, height=30))
            else:
                self.tmp.append(gui.DropDown(self.qustlist[i],width=200, height=30))
            self.tmplabel.append(gui.Label(self.labellist[i], width=200, height=30))
        # tmp.reverse()
        # tmplabel.reverse()
        for i in range(len(self.qustlist)):
            mainContainer.append(self.tmplabel[i])
            mainContainer.append(self.tmp[i])
        # add a button
        self.ans=gui.Label("", width=200, height=30)
        mainContainer.append(self.btn)
        mainContainer.append(self.ans)
        self.btn.onclick.do(self.on_button_pressed)
        return mainContainer
    def on_button_pressed(self, widget):
        global encoder;
        test_dd=["5375"];
        for i in range(len(self.qustlist)):
            if(self.tmp[i].get_value()==None or self.tmp[i].get_value() == ""):
                test_dd.append(self.qustlist[i][0])
            else :
                test_dd.append(self.tmp[i].get_value())
        test_dd=np.array(test_dd)
        test_dd=test_dd.reshape(1,-1)
        test_dd=pd.DataFrame(test_dd)
        test_dd.columns=X_train.columns
        # for i in test_dd.columns:
        #     print(i+" : "+str(test_dd[i].shape))
        #     print("")
        encoder = LabelEncoder()
        # print(categorical_feature);
        for feature in categorical_feature:
            if(feature=="Churn" or feature=="customerID" or feature=="tenure" or feature=="MonthlyCharges" or feature=="TotalCharges"):
                continue;
            else:# print(tdata[feature]);
                encoder.fit(tdata[feature])
                test_dd[feature] = encoder.transform(test_dd[feature])
        # # print(test_dd);
        for i in test_dd.columns:
            print(i+str(test_dd[i][0]));
        print(i+" : "+str(test_dd[i]))
        Y_pred=pd.DataFrame(knn.predict(test_dd))
        # Y_pred=knn.predict(test_dd)
        if Y_pred[0][0]==1:
            self.ans.set_text("Yes")
        else: 
            self.ans.set_text("No")
        # print(data);
start(main_container);