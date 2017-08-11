import pandas as pd
import numpy as np
from sklearn import *
import matplotlib.pyplot as pyt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
dtr=pd.read_csv('train2.csv')
dtr.fillna(value=0,inplace=True)
df=np.array(dtr)
for i in range(0,len(df)):
    if df[i][5]=='S':
        df[i][6]=1
    elif df[i][5]=='C':
        df[i][7]=1
    elif df[i][5]=='Q':
        df[i][8]=1
for h in range(0,len(df)):
    if df[h][9]=='male':
        df[h][10]=1
    elif df[h][9]=='female':
        df[h][11]=1
#ptrw=np.delete(df,np.s_[5::5],0)
#ptrw=np.delete(df,np.s_[9::9],0)
X=np.delete(df,[5,9,12],axis=1)
y=np.delete(df,[0,1,2,3,4,5,6,7,8,9,10,11],axis=1)
#y=np.array(y,dtype=float)#print X[0]
#print y[1]
X_train=X[:847]
X_test=X[847:]
y_train=y[:847]
y_test=y[847:]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=1)
#print X_train
lreg = linear_model.LinearRegression()
lreg.fit(X_train,y_train)
print "Regression Cofficient /n",lreg.coef_
print "Predrction /n ",lreg.predict(X_test)
print "Score ",lreg.score(X_test,y_test)
X_train=X_train.tolist()
y_train=y_train.tolist()
X_test=X_test.tolist()
y_test=y_test.tolist()
clfsv = KNeighborsClassifier()
clfsv.fit(X_train,y_train)
print "KNN prediction ",(clfsv.predict(X_test))
#print "Regression Cofficient /n",svmm.coef_
#print "Predrction /n ",svmm.predict(X_test)
#print "Score ",svmm.score(X_test,y_test)

'''
for i in range(0,len(df)):
    for j in range(0,len(df[i])):
        if np.isnan(df[i][j])==True:
            df[i][j]=np.nan_to_num
'''
#print np.isnan(df)#==True:

 #   print "Yes"
#np.delete(df,)
#for i in range(0,len(df)):
#    print df[i][12]

#print ptrw[0]#np.array(ptrw,dtype=int)#.isnull().values.sum()

#rt=np.nan_to_num(er)
#uy=np.isnan(rt)

#print uy
#artr=[[1,2,3,4],[3,4,5,2],[6,4,2,5],[4,5,3,2]]

#print artr
#print ptrw
#if df[]
#print df
#print und
