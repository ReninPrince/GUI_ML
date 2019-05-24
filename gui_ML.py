import pymongo
from pymongo import MongoClient
from tkinter import *
import time;
import datetime
import random
from tkinter import messagebox
import numpy as np
import pandas as pd
from tkinter import simpledialog

#GLOBAL VALUES
d_c = []
x = pd.DataFrame()
y = pd.DataFrame()
X_train = pd.DataFrame()
X_test = pd.DataFrame()
y_train = pd.DataFrame()
y_test = pd.DataFrame()
X_poly = pd.DataFrame()
y_pred = pd.DataFrame()

alldata = pd.DataFrame()
radio = []
radio1 = []
Values2 = []
Values1 = []
Values = []
ScaleV = 0
SplitV = 0
size = 0
algs = str(0)
answer = str(0)
def fourth():
    root4 = Tk()
    root4.overrideredirect(True)
    root4.geometry("{0}x{1}+0+0".format(root4.winfo_screenwidth(), root4.winfo_screenheight()))
    root4.title("Store Name")
    #-------------------------------------------------------------------------------------------------------------------------------------------
    global y_pred,x,y,X_train, X_test, y_train, y_test,X_poly,ScaleV,SplitV,Yscale,algs,answer,size
    predictor = StringVar()
    predicted = StringVar()
    k = []
    tp = []
    try:
        col = list(y.columns)
        col1 = list(y)
        for i in range(0,10):
            for j in col1:
                k.append(y[j][i])
            
            t = y_pred[i][0]
            tp.append(round(t,2))
    except:
        print("went wrong")
        pass
    
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    Titlecard = Frame(root4, width = 1280, height = 100, bd = 7, bg = 'dodgerblue', relief = GROOVE)
    Titlecard.pack(side = 'top', anchor = CENTER, fill = X)
    rt = time.strftime("%d/%m/%y")
    body  = Frame(root4, width = 1280, height = 600, bd = 9, bg = 'dodgerblue3', relief = FLAT)
    body.pack(side = 'top',expand = 1 ,fill = BOTH)
    login = Frame(body, width = 600, height = 400, bd = 7, bg = 'Steelblue2', relief = RAISED)
    login.pack(side = TOP, anchor = CENTER ,expand = 1, fill = BOTH, ipady = 20,ipadx = 10)
    loginbtns = Frame(body, width = 700, height = 30, bd = 7, bg = 'Steelblue2', relief = RAISED)
    loginbtns.pack(side = BOTTOM,anchor = CENTER, fill = X)
    #-------------------------------------------------------------------------------------------------------------------------------------------
    def predictor1():
        global y_pred,x,y,X_train, X_test, y_train, y_test,X_poly,ScaleV,SplitV,Yscale,algs,answer,size
        pro = round(float(predictor.get()),2)
        pru = str(str(pro) + ',')
        lsp = pru.split(',')
        prel = lsp[:-1]
        pre = pd.DataFrame(prel)
        if len(x) != 0 and len(y) != 0:
            if SplitV == 1 and  ScaleV == 1 :
                size1 = size
                yscale = Yscale
                from sklearn.cross_validation import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = size1, random_state = 0)
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                if yscale > 0:
                    y_train = sc_X.fit_transform(y_train)
                    y_test = sc_X.transform(y_test)
                if str(algs) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                    
                elif  str(algs) == "Multiple Linear Regression":
                    pass

                elif str(algs) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(X_train,y_train)
                    from sklearn.preprocessing import PolynomialFeatures
    ##                answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(X_train)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Support Vector Regression":
    ##                answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Decision Tree Regression":
    ##                answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Random Forest Regression":
    ##                answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(X_train,np.ravel( y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)
                predicted.set(predicted1)

                    
            elif SplitV == 1 and ScaleV == 0:
                size1 = size
                from sklearn.cross_validation import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = size1, random_state = 0)
                if str(algs) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)
                    
                elif  str(algs) == "Multiple Linear Regression":
                    pass

                elif str(algs) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(X_train,np.ravel(y_train))
                    from sklearn.preprocessing import PolynomialFeatures
    ##                answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(X_train)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly,np.ravel( y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Support Vector Regression":
    ##                answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Decision Tree Regression":
    ##                answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Random Forest Regression":
    ##                answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(X_train,np.ravel( y_train))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)
                predicted.set(predicted1)

                    
            elif SplitV == 0 and ScaleV == 1:
                yscale1 = Yscale
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                x = sc_X.fit_transform(x)
                if yscale1 > 0:
                    y = sc_X.fit_transform(y)
                if str(algs) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(x,np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)
                    
                elif  str(algs) == "Multiple Linear Regression":
                    pass

                elif str(algs) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(x,np.ravel(y))
                    from sklearn.preprocessing import PolynomialFeatures
    ##                answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(x)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Support Vector Regression":
    ##                answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(x,np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Decision Tree Regression":
    ##                answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(x,np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Random Forest Regression":
    ##                answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(x, np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)
                predicted.set(predicted1)

                    
            else:
                if str(algs) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(x,np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif  str(algs) == "Multiple Linear Regression":
                    pass

                elif str(algs) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(x,np.ravel(y))
                    from sklearn.preprocessing import PolynomialFeatures
    ##                answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(x)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Support Vector Regression":
    ##                answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(x,np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Decision Tree Regression":
    ##                answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(x,np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)

                elif str(algs) == "Random Forest Regression":
    ##                answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(x, np.ravel(y))
                    y_pred1 = regressor.predict(pre)
                    predicted1 = str(y_pred1)
                predicted.set(predicted1)
    
    def backk():
        global y_pred,x,y,X_train, X_test, y_train, y_test,X_poly,ScaleV,SplitV,Yscale,algs,answer,size,tp,k
        y_pred = pd.DataFrame()
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.DataFrame()
        y_test = pd.DataFrame()
        ScaleV = 0
        SplitV = 0
        size = 0
        algs = str(0)
        answer = str(0)
        tp = []
        k = []
        root4.destroy()
        third()
    def exiit():
        qexit = messagebox.askyesno("GUI","DO YOU WISH TO EXIT")
        if qexit > 0:
              root4.destroy()
   #-------------------------------------------------------------------------------------------------------------------------------------------
    date1 = Label(Titlecard, text = "DATE:" + rt,relief = GROOVE, width = 17, bd  = 7,bg = 'white', fg = 'black',font = ('arial', 15, 'italic'))
    date1.pack(side = RIGHT, anchor = NW, pady = 15)

    Title = Label(Titlecard, text = "SHOP NAME", relief = GROOVE, width = 15 , bd = 7, bg = 'dodgerblue4',
                  fg = 'lightSkyblue2', font = ('arial', 20, 'italic'))
    Title.pack(side = LEFT,pady = 15, ipadx = 35, padx =45)

    logintitle = Label(login, text = "Predicted values :", relief = FLAT, width = 10 , bd = 6, bg = 'black',
                       fg = 'Steelblue', font = ('arial', 20, 'italic'))
    logintitle.grid(row = 0, column = 0, columnspan = 3)
    #-------------------------------------------------------------------------------------------------------------------------------------------

 
    
    Label(login, text = "Predicted values :", relief = FLAT, width = 15 , bd = 6, bg = 'Steelblue2',
                       fg = 'black', font = ('arial', 20, 'italic')).grid(row = 0, column = 1)
    Label(login, text = "Dependent values :", relief = FLAT, width = 15 , bd = 6, bg = 'Steelblue2',
           fg = 'black', font = ('arial', 20, 'italic')).grid(row = 0, column = 2)
    Label(login, text = "Enter the value \nto predict :", relief = FLAT, width = 15 , bd = 6, bg = 'Steelblue2',
           fg = 'Steelblue2', font = ('arial', 20, 'italic')).grid(row = 0, column = 3)
    Label(login, text = "Enter the value \nto predict :", relief = FLAT, width = 15 , bd = 6, bg = 'Steelblue2',
           fg = 'black', font = ('arial', 20, 'italic')).grid(row = 0, column = 4)
    Entry(login, relief=SUNKEN,font = ('arial', 15, 'italic'), textvariable = predictor,
               bd = 9, insertwidth = 3).grid(row=1,column=4,pady = 20)
    Label(login, text = "Predicted value :", relief = FLAT, width = 15 , bd = 6, bg = 'Steelblue2',
           fg = 'black', font = ('arial', 20, 'italic')).grid(row = 2, column = 4)
    Label(login, textvariable = predicted, relief=FLAT,font = ('arial', 15, 'italic'),width = 15 , bd = 6, bg = 'white',
           fg = 'black').grid(row=3,column=4,pady = 20)
    
    btn1 = Button(login, text = "PREDICT",command=predictor1, relief = GROOVE, width = 10 , bd = 5, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).grid(row = 1, column = 5)

    btn1 = Button(loginbtns, text = "BACK" ,command = backk, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X,ipady = 6)
    btn2 = Button(loginbtns, text = "EXIT",command = exiit, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X,ipady = 6)

    try:
        r = 1
        for i in range(6):
            Label(login, text =  str(tp[i]), relief = GROOVE, width = 15 , bd = 4, bg = 'Steelblue2',
                           fg = 'black', font = ('arial', 20, 'italic')).grid(row = r, column = 1)
            r = r + 1
            
        r = 1
        for i in range(6):
            Label(login, text =  str(round(k[i],2)), relief = GROOVE, width = 15 , bd = 4, bg = 'Steelblue2',
                           fg = 'black', font = ('arial', 20, 'italic')).grid(row = r, column = 2)
            r = r + 1
    except:
        print("something here went wrong")
        Label(login, text =  "Couldn't\n import \ndata", relief = GROOVE, width = 15 , bd = 4, bg = 'Steelblue2',
                           fg = 'black', font = ('arial', 20, 'italic')).grid(row = 1, column = 1)
        Label(login, text =  "Couldn't\n import \ndata", relief = GROOVE, width = 15 , bd = 4, bg = 'Steelblue2',
                           fg = 'black', font = ('arial', 20, 'italic')).grid(row = 1, column = 2)

        pass
    root4.mainloop()
#-------------------------------------------------------------------------------------------------------------------------------------------

def third():
    root2 = Tk()
    root2.overrideredirect(True)
    root2.geometry("{0}x{1}+0+0".format(root2.winfo_screenwidth(), root2.winfo_screenheight()))
    root2.title("GUI for ML algorithims")
    #-------------------------------------------------------------------------------------------------------------------------------------------
    Titlecard = Frame(root2, width = 1280, height = 100, bd = 7, bg = 'blue', relief = GROOVE)
    Titlecard.pack(side = 'top', anchor = CENTER, fill = X)
    rt = time.strftime("%d/%m/%y")
    body  = Frame(root2, width = 1280, height = 600, bd = 9, bg = 'dodgerblue3', relief = FLAT)
    body.pack(side = 'top',expand=1,fill = BOTH)
    login = Frame(body, width = 1000, height = 600, bd = 7, bg = 'dodgerblue3', relief = RAISED)
    login.pack(side = TOP,expand=1, anchor = CENTER, fill = BOTH, ipady = 40,ipadx = 10)
    loginbtns = Frame(body, width = 700, height = 50, bd = 7, bg = 'Steelblue2', relief = RAISED)
    loginbtns.pack(side = BOTTOM,anchor = CENTER, fill = X)
    #-------------------------------------------------------------------------------------------------------------------------------------------

    Scale = IntVar()
    Split = IntVar()
    Spsize = StringVar()
    tkvar = StringVar()
    #-------------------------------------------------------------------------------------------------------------------------------------------
    def back():
        global d_c,alldata,x,y,radio,radio1,Values,Values1,Values2,X_train, X_test, y_train, y_test,y_pred
        radio = []
        radio1 = []
        Values2 = []
        Values1 = []
        Values = []
        X_train = pd.DataFrame()
        X_test = pd.DataFrame()
        y_train = pd.DataFrame()
        y_test = pd.DataFrame()
        y_pred = pd.DataFrame()
        root2.destroy()
        second()

    def okay():
        global x,y,X_train, X_test, y_train, y_test,X_poly,ScaleV,SplitV,Yscale,algs,answer,size,y_pred
        if len(x) != 0 and len(y) != 0:
            ScaleV = Scale.get()
            SplitV = Split.get()
            algs = str(tkvar.get())
            if Split.get() == 1 and  Scale.get() == 1 :
                size = float(Spsize.get())
                yscale = messagebox.askyesno("GUI","Do you want to scale dependent variable?")
                Yscale = yscale
                from sklearn.cross_validation import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = size, random_state = 0)
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                X_train = sc_X.fit_transform(X_train)
                X_test = sc_X.transform(X_test)
                if yscale > 0:
                    y_train = sc_X.fit_transform(y_train)
                    y_test = sc_X.transform(y_test)
                algs = str(tkvar().get)
                if str(tkvar.get()) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred = regressor.predict(X_test)
                    
                elif  str(tkvar.get()) == "Multiple Linear Regression":
                    pass

                elif str(tkvar.get()) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(X_train,np.ravel(y_train))
                    from sklearn.preprocessing import PolynomialFeatures
                    answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(X_train)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, np.ravel(y_train))
                    y_pred = regressor.predict(X_test)

                elif str(tkvar.get()) == "Support Vector Regression":
                    answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred = regressor.predict(X_test)

                elif str(tkvar.get()) == "Decision Tree Regression":
                    answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred = regressor.predict(X_test)

                elif str(tkvar.get()) == "Random Forest Regression":
                    answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(X_train, np.ravel(y_train))
                    y_pred = regressor.predict(X_test)
                root2.destroy()
                fourth()

                    
            elif Split.get() == 1 and Scale.get() == 0:
                size = float(Spsize.get())
                from sklearn.cross_validation import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = size, random_state = 0)
                if str(tkvar.get()) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred = regressor.predict(X_test)
                    
                elif  str(tkvar.get()) == "Multiple Linear Regression":
                    pass

                elif str(tkvar.get()) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(X_train,y_train)
                    from sklearn.preprocessing import PolynomialFeatures
                    answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(X_train)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, np.ravel(y_train))
                    y_pred = regressor.predict(X_test)

                elif str(tkvar.get()) == "Support Vector Regression":
                    answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred = regressor.predict(X_test)

                elif str(tkvar.get()) == "Decision Tree Regression":
                    answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(X_train,np.ravel(y_train))
                    y_pred = regressor.predict(X_test)

                elif str(tkvar.get()) == "Random Forest Regression":
                    answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(X_train, np.ravel(y_train))
                    y_pred = regressor.predict(X_test)
                root2.destroy()
                fourth()

                    
            elif Split.get() == 0 and Scale.get() == 1:
                yscale1 = messagebox.askyesno("GUI","Do you want to scale dependent variable?")
                Yscale = yscale1
                from sklearn.preprocessing import StandardScaler
                sc_X = StandardScaler()
                x = sc_X.fit_transform(x)
                if yscale1 > 0:
                    y = sc_X.fit_transform(y)
                if str(tkvar.get()) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(x,y)
                    y_pred = regressor.predict(x)
                    
                elif  str(tkvar.get()) == "Multiple Linear Regression":
                    pass

                elif str(tkvar.get()) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(x,y)
                    from sklearn.preprocessing import PolynomialFeatures
                    answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(x)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, y)
                    y_pred = regressor.predict(x)

                elif str(tkvar.get()) == "Support Vector Regression":
                    answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(x,y)
                    y_pred = regressor.predict(x)

                elif str(tkvar.get()) == "Decision Tree Regression":
                    answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(x,y)
                    y_pred = regressor.predict(x)

                elif str(tkvar.get()) == "Random Forest Regression":
                    answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(x, y)
                    y_pred = regressor.predict(x)
                root2.destroy()
                fourth()

                    
            else:
                if str(tkvar.get()) == "Simple Linear Regression":
                    from sklearn.linear_model import LinearRegression
                    regressor = LinearRegression()
                    regressor.fit(x,y)
                    y_pred = regressor.predict(x)

                elif  str(tkvar.get()) == "Multiple Linear Regression":
                    pass

                elif str(tkvar.get()) == "Polynomial  Regression":
                    from sklearn.linear_model import LinearRegression
                    lin_reg = LinearRegression()
                    lin_reg.fit(x,y)
                    from sklearn.preprocessing import PolynomialFeatures
                    answer = simpledialog.askstring("GUI", ["Degree:"])
                    poly_reg = PolynomialFeatures(degree = int(answer))
                    X_poly = poly_reg.fit_transform(x)
                    reg2 = LinearRegression()
                    reg2.fit(X_poly, y)
                    y_pred = regressor.predict(x)

                elif str(tkvar.get()) == "Support Vector Regression":
                    answer = simpledialog.askstring("GUI", ["Kernel:"])
                    from sklearn.svm import SVR
                    regressor = SVR(kernel = answer)
                    regressor.fit(x,y)
                    y_pred = regressor.predict(x)

                elif str(tkvar.get()) == "Decision Tree Regression":
                    answer = simpledialog.askstring("GUI", ["Random state:"])
                    from sklearn.tree import DecisionTreeRegressor
                    regressor = DecisionTreeRegressor(random_state = int(answer))
                    regressor.fit(x,y)
                    y_pred = regressor.predict(x)

                elif str(tkvar.get()) == "Random Forest Regression":
                    answer = simpledialog.askstring("GUI", ["n_estimators:"])
                    from sklearn.ensemble import RandomForestRegressor
                    regressor = RandomForestRegressor(n_estimators = int(answer), random_state = 0)
                    regressor.fit(x, y)
                    y_pred = regressor.predict(x)
                root2.destroy()
                fourth()
                
                    
        



         




         
    #-------------------------------------------------------------------------------------------------------------------------------------------
    date1 = Label(Titlecard, text = "DATE:" + rt,relief = GROOVE, width = 17, bd  = 7,bg = 'white', fg = 'black',font = ('arial', 15, 'italic'))
    date1.pack(side = RIGHT, anchor = NW, pady = 15)

    Title = Label(Titlecard, text = "GUI for ML algorithims", relief = GROOVE, width = 15 , bd = 7, bg = 'dodgerblue4',
                  fg = 'lightSkyblue2', font = ('arial', 20, 'italic'))
    Title.pack(side = LEFT,pady = 15, ipadx = 35, padx =45)
    #dummy
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=20, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=0,padx = 25, pady = 15,ipady = 2)


    
    #dummy
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=20, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=1,column=0,padx = 25, pady = 15,ipady = 2)
    #heading
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=25, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=1,column=0,padx = 25, pady = 15,ipady = 2)
    #dummy
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=20, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=1,column=2,padx = 25, pady = 15,ipady = 2)
    #dummy
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=20, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=3,column=2,padx = 25, pady = 15,ipady = 2)

    Radiobutton(login, text = "YES",value = 1,variable=Scale,indicatoron=0
                     ,bg = 'steelblue',font = ('arial', 15, 'bold')).grid(row = 2,column = 0,padx =5, ipadx =15)

    Radiobutton(login, text = "NO",value = 2,variable=Scale,indicatoron=0
                     ,bg = 'steelblue',font = ('arial', 15, 'bold')).grid(row = 4,column = 0,padx =5, ipadx =15)

    #heading
    Label(login, text="Do you wish to split the data? ", relief=FLAT,width=25, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=1,column=2,padx = 25, pady = 15,ipady = 2)

    Radiobutton(login, text = "YES",value = 1,variable=Split,indicatoron=0
                     ,bg = 'steelblue',font = ('arial', 15, 'bold')).grid(row = 2,column = 2,padx =5, ipadx =15)

    Radiobutton(login, text = "NO",value = 2,variable=Split,indicatoron=0
                     ,bg = 'steelblue',font = ('arial', 15, 'bold')).grid(row = 4,column = 2,padx =5, ipadx =15)

    #heading
    Label(login, text="Enter split size : ", relief=FLAT,width=25, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=1,column=4,padx = 25, pady = 15,ipady = 2)
    Entry(login,relief=SUNKEN,font = ('arial', 15, 'italic'), textvariable = Spsize,
               bd = 9, insertwidth = 3).grid(row=2,column=4)
    #dummy
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=20, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=5,column=0,padx = 25, pady = 7,ipady = 2)

    #dummy
    Label(login, text="Do you wish to scale the datas? ", relief=FLAT,width=20, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=7,column=0,padx = 25, pady = 7,ipady = 2)
    
    #heading
    Label(login, text="Select your algorithim : ", relief=FLAT,width=30, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=6,column=0,padx = 25, pady = 7,ipady = 2)

    #heading_under construction
    Label(login, text="Select your error correction : ", relief=FLAT,width=30, bd = 4, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=8,column=0,padx = 25, pady = 7,ipady = 2)

    
    choices = { 'Simple Linear Regression','Multiple Linear Regression','Polynomial  Regression',
                'Support Vector Regression','Decision Tree Regression','Random Forest Regression'}
    tkvar.set('Simple Linear Regression') # set the default option

    popupMenu = OptionMenu(login, tkvar, *choices)
    popupMenu.config(fg = 'black',bg = 'dodgerblue3', relief=GROOVE, bd = 7)
    popupMenu["menu"].config(fg = 'black',bg = 'dodgerblue3', relief=FLAT ,bd = 10)
    popupMenu.grid(row=6,column=3,columnspan=4,padx = 30, pady = 7,ipadx = 25)
    

    
    btn1 = Button(loginbtns, text = "OKAY",command=okay, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X)

    btn3 = Button(loginbtns, text = "BACK",command=back, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X)


    root2.mainloop()


    
def second():
    root1 = Tk()
    root1.overrideredirect(True)
    root1.geometry("{0}x{1}+0+0".format(root1.winfo_screenwidth(), root1.winfo_screenheight()))
    root1.title("GUI for ML algorithims")
    #-------------------------------------------------------------------------------------------------------------------------------------------
    Titlecard = Frame(root1, width = 1280, height = 100, bd = 7, bg = 'blue', relief = GROOVE)
    Titlecard.pack(side = 'top', anchor = CENTER, fill = X)
    rt = time.strftime("%d/%m/%y")
    body  = Frame(root1, width = 1280, height = 600, bd = 9, bg = 'dodgerblue3', relief = FLAT)
    body.pack(side = 'top',expand = 1,fill = BOTH)
    login = Frame(body, width = 1000, height = 600, bd = 7, bg = 'dodgerblue3', relief = RAISED)
    login.pack(side = TOP, anchor = CENTER,expand=1, fill = X, ipady = 40,ipadx = 10) 
    #-------------------------------------------------------------------------------------------------------------------------------------------

    var = IntVar()
    var1 = IntVar()
    global d_c,radio,radio1,Values,Values1,Values2
    for i in range(len(d_c)):
        text = str(d_c[i])
        Values.append(text)
        length = len(alldata[text])
        Values1.append(length)
        if length != len(alldata):
            g = len(alldata) - length
            Values2.append(g)
        else:
            Values2.append('NULL')
        text1 = str(str(text) + "1")
        text1 = IntVar()
        radio.append(text1)
        text2 = str(str(text) + "2")
        text2 = IntVar()
        radio1.append(text2)
    rn = len(d_c)
    #-------------------------------------------------------------------------------------------------------------------------------------------
    def back():
        global d_c,alldata,x,y,radio,radio1,Values,Values1,Values2
        root1.destroy()
        d_c = []
        x = pd.DataFrame()
        y = pd.DataFrame() 
        alldata = pd.DataFrame()
        radio = []
        radio1 = []
        Values2 = []
        Values1 = []
        Values = []    
        main()

    def clear():
         for y in Values:
              y.set("")
    def exiit():
         qexit = messagebox.askyesno("GUI","DO YOU WISH TO EXIT")
         if qexit > 0:
              root.destroy()
    def assign():
        global x,y,radio,radio1,Values
        for i in range(len(radio)):
            if radio[i].get() == 1:
                x[str(Values[i])] = alldata[str(Values[i])]
        for j in range(len(radio1)):
            if radio1[j].get() == 1:
                y[str(Values[j])] = alldata[str(Values[j])]
        root1.destroy()
        third()
         
         
    #-------------------------------------------------------------------------------------------------------------------------------------------
    date1 = Label(Titlecard, text = "DATE:" + rt,relief = GROOVE, width = 17, bd  = 7,bg = 'white', fg = 'black',font = ('arial', 15, 'italic'))
    date1.pack(side = RIGHT, anchor = NW, pady = 15)

    Title = Label(Titlecard, text = "GUI for ML algorithims", relief = GROOVE, width = 15 , bd = 7, bg = 'dodgerblue4',
                  fg = 'lightSkyblue2', font = ('arial', 20, 'italic'))
    Title.pack(side = LEFT,pady = 15, ipadx = 35, padx =45)

    Label(login, text="Column name: ", relief=FLAT,width=20, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=0,padx = 15, pady = 15,ipady = 2)
    Label(login, text="Number of datas :", relief=FLAT,width=15, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=1,padx = 25, pady = 15,ipady = 2)
    Label(login, text="Number of \n missing values : ", relief=FLAT,width=15, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=2,padx = 15, pady = 15,ipady = 2)
    Label(login, text="Select \n independent values : ", relief=FLAT,width=16, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=3,padx = 15, pady = 15,ipady = 2)
    Label(login, text="Select \n dependent values : ", relief=FLAT,width=16, bd = 4, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=4,padx = 15, pady = 15,ipady = 2)
      
    r = 1
    for t in Values:
         Label(login, text=t, relief=FLAT,width=20, bd = 6, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=r,column=0,padx = 15, pady = 15,ipady = 2)
         r = r + 1

    r = 1
    for t in Values1:
         Label(login, text=t, relief=FLAT,width=20, bd = 6, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=r,column=1,padx = 40, pady = 15,ipady = 2)
         r = r + 1
         
    r = 1
    for t in Values2:
         Label(login, text=t, relief=FLAT,width=15, bd = 6, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=r,column=2,padx = 25, pady = 15,ipady = 2)
         r = r + 1
         
    r = 1   
    for t in radio:
        Checkbutton(login,variable=t, fg = 'black',bg = 'dodgerblue3'
                    ).grid(row=r,column=3,padx = 25, pady = 15,ipady = 2)

        r = r + 1
        
    r = 1    
    for t in radio1:
        Checkbutton(login,  variable=t, fg = 'black',bg = 'dodgerblue3'
                    ).grid(row=r,column=4,padx = 25, pady = 15,ipady = 2)
        r = r + 1

    btn1 = Button(body, text = "OKAY" ,command = assign, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X,ipady = 10)
    btn2 = Button(body, text = "CLEAR", relief = FLAT, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'Steelblue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X,ipady = 10)
    btn3 = Button(body, text = "BACK",command = back, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).pack(side =LEFT, anchor = CENTER,expand = 2, fill = X,ipady = 10)

 #-------------------------------------------------------------------------------------------------------------------------------------------
    root1.mainloop()














   
def main():
    root = Tk()
    root.overrideredirect(True)
    root.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    root.title("GUI for ML algorithims")
    #-------------------------------------------------------------------------------------------------------------------------------------------
    Titlecard = Frame(root, width = 1280, height = 100, bd = 7, bg = 'blue', relief = GROOVE)
    Titlecard.pack(side = 'top', anchor = CENTER, fill = X)
    rt = time.strftime("%d/%m/%y")
    body  = Frame(root, width = 1280, height = 600, bd = 9, bg = 'dodgerblue3', relief = FLAT)
    body.pack(side = 'top',fill = BOTH)
    login = Frame(body, width = 600, height = 600, bd = 7, bg = 'dodgerblue3', relief = RAISED)
    login.pack(side = TOP, anchor = CENTER, fill = Y, ipady = 100,ipadx = 10) 
    #-------------------------------------------------------------------------------------------------------------------------------------------
    Username = StringVar()
    Password = StringVar()
    Values1 = ['File name :']
    Values = [Username]
    #-------------------------------------------------------------------------------------------------------------------------------------------
    def clear():
         for y in Values:
              y.set("")
    def exiit():
         qexit = messagebox.askyesno("GUI","DO YOU WISH TO EXIT")
         if qexit > 0:
              root.destroy()

    def logn():
        global d_c,alldata
        Username2 = str(str(Username.get()) + ".csv")
        try:
            dataset = pd.read_csv(Username2)
            
            for i in range(len(dataset.columns)):
                d_c.append(dataset.columns[i])
            if len(dataset) > 0:
                alldata = alldata.append(dataset, ignore_index = True)
                root.destroy()
                second()
            else:
                print('nw')
                
        except:
            print("no file")
            messagebox.showerror("GUI", "Incorrect file name")
         
         
    
    #-------------------------------------------------------------------------------------------------------------------------------------------
    date1 = Label(Titlecard, text = "DATE:" + rt,relief = GROOVE, width = 17, bd  = 7,bg = 'white', fg = 'black',font = ('arial', 15, 'italic'))
    date1.pack(side = RIGHT, anchor = NW, pady = 15)

    Title = Label(Titlecard, text = "GUI for ML algorithims", relief = GROOVE, width = 15 , bd = 7, bg = 'dodgerblue4',
                  fg = 'lightSkyblue2', font = ('arial', 20, 'italic'))
    Title.pack(side = LEFT,pady = 15, ipadx = 35, padx =45)

    Label(login, text='File name', relief=FLAT,width=10,padx = 10, pady = 10, bd = 6, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=0,column=0)
    
    logintitle = Label(login, text = "Enter the excel file", relief = FLAT, width = 20 , bd = 6, bg = 'dodgerblue3',
                       fg = 'black', font = ('arial', 20, 'italic'))
    logintitle.grid(row = 1, column = 0, columnspan = 3)

##    Label(login, text='File name', relief=FLAT,width=10,padx = 10, pady = 10, bd = 6, fg = 'dodgerblue3',bg = 'dodgerblue3',
##               font = ('arial', 15, 'bold')).grid(row=2,column=0)
    Label(login, text='File name', relief=FLAT,width=10,padx = 10, pady = 10, bd = 6, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=3,column=0)
    Label(login, text='File name', relief=FLAT,width=10,padx = 10, pady = 10, bd = 6, fg = 'black',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=4,column=0)

    
    Label(login, text='File name', relief=FLAT,width=10,padx = 10, pady = 10, bd = 6, fg = 'dodgerblue3',bg = 'dodgerblue3',
               font = ('arial', 15, 'bold')).grid(row=5,column=0)

    Entry(login, relief=SUNKEN,font = ('arial', 15, 'italic'), textvariable = Username,
               bd = 9, insertwidth = 3).grid(row=4,column=1,pady = (20,20))

   #-------------------------------------------------------------------------------------------------------------------------------------------
    btn1 = Button(login, text = "OK",command = logn, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).grid(row = 6, column = 0,columnspan = 3,pady = (4,20))
    btn2 = Button(login, text = "CLEAR",command = clear, relief = RAISED, width = 10 , bd = 6, bg = 'Steelblue2',
                       fg = 'blue2', font = ('arial', 20, 'italic')).grid(row = 7, column = 0, columnspan = 3,pady = (4,20))
    btn4 = Button(login, text = "EXIT",command = exiit, relief = RAISED, width = 10 , bd = 6, bg = 'red',
                       fg = 'black', font = ('arial', 20, 'italic')).grid(row = 8, column = 0, columnspan = 3,pady = (4,20))
    #-------------------------------------------------------------------------------------------------------------------------------------------
    root.mainloop()








































main()

