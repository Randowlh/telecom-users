import remi.gui as gui
from remi import start, App
class main_container(App):
    qustlist =[["Female","Male"],["Yes","No"],["Yes","No"],["Yes","No"],["Yes","No"],["Yes","No"],["Yes","No","No phone service"],["DSL","Fiber optic","No"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Yes","No","No internet service"],["Month-to-month","One year","Two year"],["Yes","No"],["Electronic check","Credit card (automatic)","Bank transfer (automatic)","Mailed check"],["Yes","No"],["Yes","No"]]

    labellist = ["gender","SeniorCitizen","Partner","Dependents","tenure",	"PhoneService",	"MultipleLines",    "InternetService",	"OnlineSecurity","OnlineBackup","DeviceProtection",	"TechSupport",	"StreamingTV",	"StreamingMovies",	"Contract",	"PaperlessBilling",	"PaymentMethod","MonthlyCharges",	"TotalCharges"]

    def __init__(self, *args):
        super(main_container, self).__init__(*args)
        self.title = "test"
    def main(self):
        # create a container VBox type, with vertical orientation
        mainContainer = gui.VBox(width=300, height=2000, style={'margin':'0px auto', 'padding':'0px'})
        self.btn = gui.Button('Submit', width=200, height=30)
        tmp=[]
        tmplabel=[]
        for i in range(len(self.qustlist)):
            if(self.labellist[i]=="tenure" or self.labellist[i]=="MonthlyCharges" or self.labellist[i]=="TotalCharges"):
                tmp.append(gui.TextInput(width=200, height=30))
            else:
                tmp.append(gui.DropDown(self.qustlist[i],width=200, height=30))
            tmplabel.append(gui.Label(self.labellist[i], width=200, height=30))
        # tmp.reverse()
        # tmplabel.reverse()
        for i in range(len(self.qustlist)):
            mainContainer.append(tmplabel[i])
            mainContainer.append(tmp[i])
        # add a button
        mainContainer.append(self.btn)
        # self.btn.onclick.do(self.on_button_pressed)
        return mainContainer
    # def on_button_pressed(self, widget):
        # print(self.dd.get_value()) 
start(main_container);