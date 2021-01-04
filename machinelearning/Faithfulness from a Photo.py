#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# read data as data
data = pd.read_csv('FaithfulFaces.csv', index_col=[0])

data.describe()


# In[2]:


data = pd.read_csv('FaithfulFaces.csv')


# In[3]:


data.describe()


# In[4]:


data.head()


# In[5]:


data = pd.read_csv('FaithfulFaces.csv', index_col=[0])


# In[6]:


data.describe()


# In[7]:


corr = data.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.Reds)
plt.show()


# In[8]:


sns.relplot(x="Attract", y="SexDimorph", hue="Cheater", data=data);


# In[9]:


x_cheater = data.replace({'Cheater' : {1: 'yes', 0: 'no'}})
y_cheater = data.replace({'Cheater' : {1: 'yes', 0: 'no'}})
sns.relplot(x="Attract", y="SexDimorph", hue='x_cheater' 'y_cheater', data=data);


# In[10]:


data.replace({'Cheater' : {1: 'yes', 0: 'no'}})

sns.relplot(x="Attract", y="SexDimorph", hue="Cheater", data=data);


# In[11]:


data.replace({'Cheater' : {1: 'yes', 0: 'no'}}).groupby('Cheater').size().reset_index(name='Counts')['Cheater']

sns.relplot(x="Attract", y="SexDimorph", hue="Cheater", data=data);


# In[12]:


data['Cheater'] = data['Cheater'].map({0 : 'No', 1 : 'Yes'})


# In[13]:


sns.relplot(x="Attract", y="SexDimorph", hue="Cheater", data=data);


# In[14]:


data['Cheater'] = data['Cheater'].map({'No' : 'Yes', 'Yes' : 'No'})
sns.relplot(x="Attract", y="SexDimorph", hue="Cheater", data=data);


# In[15]:


data_ml = data.copy()


# In[16]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[17]:


data_ml = pd.get_dummies(data_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=true)
data_ml.head()


# In[18]:


data_ml = pd.get_dummies(data_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=True)


# In[19]:


data_ml.head()


# In[20]:


data_ml.head(10)


# In[21]:


data_ml.info()


# In[22]:


corr = data_ml.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[23]:


data_ml=data_ml.rename(columns={"Cheater_Yes":"Cheater"})


# In[24]:


data_ml.head()


# In[25]:



data_ml.info()


# In[26]:


corr = data_ml.corr()
f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[27]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


scaler.fit(data_ml.drop('Attract',axis=1))
scaled_features = scaler.transform(data_ml.drop('Attract',axis=1))
data_ml_sc = pd.DataFrame(scaled_features, columns=data_ml.columns[:-1])


# In[28]:


print(data.Attract.sum()/data.Attract.count())


# In[29]:


print(data.Cheater.sum()/data.Cheater.count())


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[31]:


data=pd.read_csv('FaithfulFaces.csv')
data.head()


# In[32]:


data=pd.read_csv('FaithfulFaces.csv', index_col=[0])
data.head()


# In[33]:


y=data.Attract
x=data.drop('Attract',axis=1)


# In[34]:


data_train,data_test,data_train,data_test=train_test_split(x,y,test_size=0.2)
data_train.head()


# In[35]:


data_test.head()


# In[36]:


data_train.shape


# In[37]:


data_test_ml = data_test.copy()
data_train_ml = data_train.copy()


# In[38]:


data_train_ml.head()


# In[39]:


from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.2, random_state = 0)


# In[40]:


xTrain.head()


# In[41]:


data_train_ml = x,yTrain.copy()


# In[42]:


data_train_ml.head()


# In[43]:


data_train_ml = xTrain.copy()


# In[44]:


data_train_ml.head()


# In[45]:


data_test_ml = xTest.copy()
data_test_ml.head()


# In[46]:


data_train_ml.describe()


# In[47]:


data_test_ml.describe()


# In[48]:


data_train_ml.info()


# In[49]:


data_train_ml = pd.get_dummies(data_train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
data_train_ml.drop(['SexDimorph','Cheater ','Trust ', 'Faithful'],axis=1,inplace=True)
data_train_ml.dropna(inplace=True)


# In[50]:


data_train_ml = pd.get_dummies(data_train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
data_train_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)
data_train_ml.dropna(inplace=True)


# In[51]:


data_train_ml.head()


# In[52]:



data_train_ml = pd.get_dummies(data_train_ml, columns=['FaceSex_M', 'RaterSex_M', 'FaceSex_F', 'RaterSex_F'], drop_first=True)
data_train_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)
data_train_ml.dropna(inplace=True)


# In[53]:



data_train_ml = pd.get_dummies(data_train_ml, columns=['FaceSex_M', 'RaterSex_M'], drop_first=True)
data_train_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)
data_train_ml.dropna(inplace=True)


# In[54]:


sex_dimorph = df_test_ml['SexDimorph']
data_test_ml = pd.get_dummies(data_test_ml, columns=['FaceSex_M', 'RaterSex_M'], drop_first=True)
data_test_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)


# In[55]:


sex_dimorph = data_test_ml['SexDimorph']
data_test_ml = pd.get_dummies(data_test_ml, columns=['FaceSex_M', 'RaterSex_M'], drop_first=True)
data_test_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)


# In[56]:


sex_dimorph = data_test_ml['SexDimorph']
data_test_ml = pd.get_dummies(data_test_ml, columns=['FaceSex_F', 'RaterSex_F'], drop_first=True)
data_test_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)


# In[57]:


data_test_ml.head()


# In[58]:


data_test_ml.info


# In[59]:


sex_dimorph = data_test_ml['SexDimorph']
data_test_ml = pd.get_dummies(data_test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
data_test_ml.drop(['SexDimorph','Cheater', 'Trust', 'Faithful'],axis=1,inplace=True)


# In[60]:


data_train_ml.head(10)


# In[61]:


data_train_ml.head


# In[62]:


data_train_ml.info()


# In[63]:


data_test_ml.info()


# In[64]:


corr = df_train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[65]:


corr = data_train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[66]:


data.head()


# In[67]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0]))
data['split'] = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[68]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0])
data['split'] = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[69]:


train.head()


# In[70]:


test.head()


# In[71]:


train_ml = train.copy()
test_ml = test.copy()


# In[72]:


train.info()


# In[73]:


train_ml = pd.get_dummies(train_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=True)
train_ml.drop(['SexDimorph','Attract','Trust', 'Faithful'],axis=1,inplace=True)
train_ml.dropna(inplace=True)


# In[74]:


test_ml = pd.get_dummies(test_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=True)
test_ml.drop(['SexDimorph','Attract','Trust', 'Faithful'],axis=1,inplace=True)
test_ml.dropna(inplace=True)


# In[75]:


train_ml.head()


# In[76]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index=False)
data['split'] = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[77]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=False)
data['split'] = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[78]:


train.info()


# In[79]:


train.head()


# In[80]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index=False)
data['split'] = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[81]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0]))
data = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[82]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0])
data = np.random.randn(data.shape[0], 1)

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[83]:


train.head()


# In[84]:


import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0])
rng = RandomState()

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[85]:


from numpy.random import RandomState
import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0])
rng = RandomState()

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[86]:


train.head()


# In[87]:


test.head()


# In[88]:


train_ml = train.copy()
test_ml = test.copy()


# In[89]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[90]:


train_ml.head()


# In[91]:


test_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[92]:


test_ml.head()


# In[93]:


test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[94]:


test_ml.head()


# In[95]:


corr = df_train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[96]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[97]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaler.fit(train_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(train_ml.drop('Cheater',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[98]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('Cheater',axis=1), train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[99]:


# unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[100]:


test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[101]:


scaler.fit(train_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(train_ml.drop('Cheater',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])


# In[102]:


scaler.fit(test_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(test_ml.drop('Cheater',axis=1))
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns[:-1])


# In[103]:


# unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[104]:


X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")


# In[105]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[106]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[107]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[108]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)
sub_logreg = pd.DataFrame()
sub_logreg['SexDimorph'] = df_test['SexDimorph']
sub_logreg['Cheater'] = pred_all_logreg
#sub_logmodel.to_csv('logmodel.csv',index=False)


# In[109]:


From sklearn import *
logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[110]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[111]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[112]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[113]:


knn.fit(X_train_all, y_train_all)
pred_all_knn = knn.predict(X_test_all)


# In[114]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[115]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[116]:


pred_dtree = dtree.predict(X_train)
print(classification_report(y_train,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[117]:


pred_dtree = dtree.predict(X_train)
print(classification_report(y_train,pred_dtree))
print(accuracy_score(y_train, pred_dtree))


# In[118]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[119]:


pred_dtree = dtree.predict(X_train)
print(classification_report(y_train,pred_dtree))
print(accuracy_score(y_train, pred_dtree))


# In[120]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[121]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[122]:


dtree_2 = DecisionTreeClassifier(max_features=7, max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[123]:


dtree_2.fit(X_train_all, y_train_all)
pred_all_dtree2 = dtree_2.predict(X_test_all)


# In[124]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6, max_features=7)
rfc.fit(X_train, y_train)


# In[125]:


pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test, pred_rfc))


# In[126]:


rfc.fit(X_train_all, y_train_all)
pred_all_rfc = rfc.predict(X_test_all)


# In[127]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[128]:


dtree_2 = DecisionTreeClassifier(max_features="auto", max_depth="auto",  min_samples_split="auto")
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[129]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[130]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[131]:


dtree_2 = DecisionTreeClassifier(max_features=0 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[132]:


dtree_2 = DecisionTreeClassifier(max_features=n_features, max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[133]:


from sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100)#, probability=True)
svc.fit(X_train_sc, y_train_sc)


# In[134]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6, max_features=7)
rfc.fit(X_train, y_train)


# In[135]:


pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test, pred_rfc))


# In[136]:


rfc.fit(X_train_all, y_train_all)
pred_all_rfc = rfc.predict(X_test_all)


# In[137]:


pred_rfc = rfc.predict(X_train)
print(confusion_matrix(y_train, pred_rfc))
print(classification_report(y_train, pred_rfc))
print(accuracy_score(y_train, pred_rfc))


# In[138]:


from sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100)#, probability=True)
svc.fit(X_train_sc, y_train_sc)


# In[139]:


pred_svc = svc.predict(X_test_sc)
print(confusion_matrix(y_test_sc, pred_svc))
print(classification_report(y_test_sc, pred_svc))
print(accuracy_score(y_test_sc, pred_svc))


# In[140]:


from sklearn.model_selection import cross_val_score


# In[141]:


scores_svc = cross_val_score(svc, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_svc)
print(scores_svc.mean())


# In[142]:


scores_rfc = cross_val_score(rfc, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_rfc)
print(scores_rfc.mean())


# In[143]:


scores_dtree_2 = cross_val_score(dtree_2, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_dtree_2)
print(scores_dtree_2.mean())


# In[1]:


from numpy.random import RandomState
import pandas as pd
import numpy as np

data = pd.read_csv('FaithfulFaces.csv', index_col=[0])
rng = RandomState()

msk = np.random.rand(len(data)) <= 0.7

train = data[msk]
test = data[~msk]


# In[2]:


train.head()


# In[3]:


test.head()


# In[4]:


train_ml = train.copy()
test_ml = test.copy()


# In[5]:


train.info()


# In[6]:


train.describe()


# In[7]:


train_ml.describe()


# In[8]:


train_ml.info()


# In[9]:


train_ml = pd.get_dummies(train_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=True)


# In[10]:


train_ml.describe()


# In[11]:


train_ml = train.copy()


# In[12]:


train_ml = pd.get_dummies(train_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=True)
test_ml = pd.get_dummies(test_ml, columns=['Cheater', 'FaceSex', 'RaterSex'], drop_first=True)


# In[13]:


train_ml.head()


# In[14]:


train_ml.head(10)


# In[15]:


corr = df_train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[16]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[17]:


import matplotlib.pyplot as plt
corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[18]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[19]:


train_ml = train.copy()
test_ml = test.copy()


# In[20]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[22]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[23]:


corr = test_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaled_features = scaler.transform(train_ml)
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns)

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[25]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaler.fit(train_ml)
scaled_features = scaler.transform(train_ml)
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns)

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[26]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml), train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml, train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[28]:


# unscaled
X_train_all = train_ml
y_train_all = train_ml
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml
X_test_all_sc = test_ml_sc


# In[29]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[30]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[31]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[32]:


train_ml = train.copy()
test_ml = test.copy()


# In[33]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)

test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[34]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[35]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('Cheater',axis=1), train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[36]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[37]:


# for train_ml
scaler.fit(train_ml)
scaled_features = scaler.transform(train_ml)
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns)

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml, train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[39]:


#unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[40]:


X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")

*


# In[41]:


X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")


# In[42]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[43]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[44]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_train)
print(confusion_matrix(y_train, pred_logreg))
print(classification_report(y_train, pred_logreg))
print(accuracy_score(y_train, pred_logreg))


# In[45]:


train_ml = train.copy()
test_ml = test.copy()

train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[46]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[47]:



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('Cheater',axis=1), train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)

scaler.fit(train_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(train_ml.drop('Cheater',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

scaler.fit(test_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(test_ml.drop('Cheater',axis=1))
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns[:-1])


# In[48]:


# unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[49]:



X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")


# In[50]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[51]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[52]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all.drop('Cheater',axis=1))


# In[53]:


sub_logreg = pd.DataFrame()
sub_logreg = test
sub_logreg['Cheater'] = pred_all_logreg
#sub_logmodel.to_csv('logmodel.csv',index=False)


# In[54]:


sub_logreg = pd.DataFrame()
sub_logreg = test
sub_logreg.loc = pred_all_logreg


# In[55]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[56]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[57]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[58]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all.drop('Cheater',axis=1))


# In[59]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[60]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[61]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[62]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[63]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[64]:


train_ml = train.copy()
test_ml = test.copy()


# In[65]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)

test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[66]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[67]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml, test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml, test_size=0.30, random_state=101)


# In[68]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('Cheater',axis=1), train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[69]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[70]:


scaler.fit(train_ml)
scaled_features = scaler.transform(train_ml)
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns)


# In[71]:


test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[72]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml, train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[73]:


train_ml = train.copy()
test_ml = test.copy()


# In[74]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)

test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[75]:


corr = train_ml.corr()

f,ax = plt.subplots(figsize=(9,6))
sns.heatmap(corr, annot = True, linewidths=1.5 , fmt = '.2f',ax=ax)
plt.show()


# In[76]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# for train_ml
scaler.fit(train_ml)
scaled_features = scaler.transform(train_ml)
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns)

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[77]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml, train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[78]:


# unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[79]:


X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")


# In[80]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[81]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[82]:


train_ml = train.copy()
test_ml = test.copy()


# In[83]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)

test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[84]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml, train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)


# In[85]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[86]:


# for train_ml
scaler.fit(train_ml)
scaled_features = scaler.transform(train_ml)
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns)

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[87]:


# unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[88]:


X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")


# In[89]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[90]:


train_ml = train.copy()
test_ml = test.copy()

train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)


# In[91]:




from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_ml.drop('Cheater',axis=1), train_ml['Cheater'], test_size=0.30, random_state=101)
X_train_sc, X_test_sc, y_train_sc, y_test_sc = train_test_split(train_ml_sc, train_ml['Cheater'], test_size=0.30, random_state=101)

scaler.fit(train_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(train_ml.drop('Cheater',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

scaler.fit(test_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(test_ml.drop('Cheater',axis=1))
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns[:-1])



# unscaled
X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc




X_test_all.fillna(X_test_all.mean(), inplace=True)
print("*")



from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all.drop('Cheater',axis=1))


# In[92]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaler.fit(train_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(train_ml.drop('Cheater',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaler.fit(test_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(test_ml.drop('Cheater',axis=1))
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns[:-1])


# In[93]:


X_train_all = train_ml.drop('Cheater',axis=1)
y_train_all = train_ml['Cheater']
X_test_all = test_ml

# scaled
X_train_all_sc = train_ml_sc
y_train_all_sc = train_ml['Cheater']
X_test_all_sc = test_ml_sc


# In[94]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[95]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all.drop('Cheater',axis=1))


# In[96]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[97]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[98]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[99]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
pred_logreg = logreg.predict(X_test)
print(confusion_matrix(y_test, pred_logreg))
print(classification_report(y_test, pred_logreg))
print(accuracy_score(y_test, pred_logreg))


# In[100]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[101]:


logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all.drop('Cheater',axis=1))


# In[102]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[103]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[104]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[105]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[106]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)
pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[107]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[108]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[109]:



pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[110]:



pred.fit(X_train_sc,y_train_sc)
pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[111]:


knn.fit(X_train_sc,y_train_sc)
pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[112]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[113]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[114]:


from sklearn import *
logreg.fit(X_train_all, y_train_all)
pred_all_logreg = logreg.predict(X_test_all)


# In[115]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(X_train,y_train)
pred_gnb = gnb.predict(X_test)
print(confusion_matrix(y_test, pred_gnb))
print(classification_report(y_test, pred_gnb))
print(accuracy_score(y_test, pred_gnb))


# In[116]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train_sc,y_train_sc)


# In[117]:


pred_knn = knn.predict(X_test)
print(confusion_matrix(y_test, pred_knn))
print(classification_report(y_test, pred_knn))
print(accuracy_score(y_test, pred_knn))


# In[118]:


knn.fit(X_train_all, y_train_all)
pred_all_knn = knn.predict(X_test_all)


# In[119]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[120]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[121]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[122]:


dtree_2.fit(X_train_all, y_train_all)
pred_all_dtree2 = dtree_2.predict(X_test_all)


# In[123]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(max_depth=6, max_features=7)
rfc.fit(X_train, y_train)


# In[124]:


pred_rfc = rfc.predict(X_test)
print(confusion_matrix(y_test, pred_rfc))
print(classification_report(y_test, pred_rfc))
print(accuracy_score(y_test, pred_rfc))


# In[125]:


rom sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100)#, probability=True)
svc.fit(X_train_sc, y_train_sc)


# In[126]:


from sklearn.svm import SVC
svc = SVC(gamma = 0.01, C = 100)#, probability=True)
svc.fit(X_train_sc, y_train_sc)


# In[127]:


pred_svc = svc.predict(X_test_sc)
print(confusion_matrix(y_test_sc, pred_svc))
print(classification_report(y_test_sc, pred_svc))
print(accuracy_score(y_test_sc, pred_svc))


# In[128]:


from sklearn.model_selection import cross_val_score


# In[129]:


scores_svc = cross_val_score(svc, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_svc)
print(scores_svc.mean())


# In[130]:


scores_rfc = cross_val_score(rfc, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_rfc)
print(scores_rfc.mean())


# In[131]:


ll_sc, cv=10, scoring='accuracy')
print(scores_dtree_2)
print(scores_dtree_2.mean())


# In[132]:


scores_dtree_2 = cross_val_score(dtree_2, X_train_all_sc, y_train_all_sc, cv=10, scoring='accuracy')
print(scores_dtree_2)
print(scores_dtree_2.mean())


# In[133]:


print("dtree_2 : " , scores_dtree_2.mean())
print("rfc     : " , scores_rfc.mean())
print("svc     : " , scores_svc.mean())


# In[134]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[135]:


from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[136]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[137]:


pred_dtree = dtree.predict(X_train)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[138]:


pred_dtree = dtree.predict(X_train)
print(classification_report(y_train,pred_dtree))
print(accuracy_score(y_train, pred_dtree))


# In[139]:


pred_dtree = dtree.predict(X_test)
print(classification_report(y_test,pred_dtree))
print(accuracy_score(y_test, pred_dtree))


# In[140]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[141]:


dtree_2 = DecisionTreeClassifier(max_features=8 , max_depth=7,  min_samples_split=9)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[142]:


dtree_2 = DecisionTreeClassifier(max_features=7, max_depth=6,  min_samples_split=8)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[143]:


dtree_2 = DecisionTreeClassifier(max_features=None, max_depth=None,  min_samples_split=None)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[144]:


dtree_2 = DecisionTreeClassifier(max_features=max_features * n_features, max_depth=None,  min_samples_split=None)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[145]:


dtree_2 = DecisionTreeClassifier(max_features=sqrt(n_features), max_depth=None,  min_samples_split=None)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[146]:


dtree_2 = DecisionTreeClassifier(max_features=Auto, max_depth=None,  min_samples_split=None)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[147]:


dtree_2 = DecisionTreeClassifier(max_features=auto, max_depth=None,  min_samples_split=None)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[148]:


dtree_2 = DecisionTreeClassifier(max_features=float, max_depth=None,  min_samples_split=None)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[149]:


dtree_2 = DecisionTreeClassifier(max_features=float, max_depth=float,  min_samples_split=float)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[150]:


dtree_2 = DecisionTreeClassifier(max_features=10, max_depth=8,  min_samples_split=9)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[151]:


dtree_2 = DecisionTreeClassifier(max_features=10, n_features, max_depth=8,  min_samples_split=9)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[152]:


dtree_2 = DecisionTreeClassifier(max_features=7 , n_features, max_depth=5,  min_samples_split=6)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[153]:


dtree_2 = DecisionTreeClassifier(max_features=7 , max_depth=5,  min_samples_split=6)
dtree_2.fit(X_train,y_train)
pred_dtree_2 = dtree_2.predict(X_test)
print(classification_report(y_test, pred_dtree_2))
print(accuracy_score(y_test, pred_dtree_2))


# In[155]:


df_train_ml = df_train.copy()
df_test_ml = df_test.copy()


# In[156]:


train_ml = train.copy()
test_ml = test.copy()


# In[157]:


train_ml.info()


# In[158]:


train.head()


# In[159]:


test.head()


# In[160]:


train.info()


# In[161]:


sns.countplot(x='Cheater', data=train);


# In[162]:


sns.countplot(x='Cheater', data=train['Faithful']);


# In[163]:


train.groupby(['Cheater','Sex'])['Cheater'].count()


# In[164]:


train.groupby(['Cheater','FaceSex'])['Cheater'].count()


# In[165]:


sns.catplot(x='FaceSex', col='Cheater', kind='count', data=train);


# In[166]:


print("% of women cheaters: " , train[train.FaceSex == 'female'].Cheater.sum()/train[train.FaceSex == 'female'].Cheater.count())
print("% of men cheaters:   " , train[train.FaceSex == 'male'].Cheater.sum()/train[train.FaceSex == 'male'].Cheater.count())


# In[167]:


print("% of women cheaters: " , train[train.FaceSex == 'F'].Cheater.sum()/train[train.FaceSex == 'F'].Cheater.count())
print("% of men cheaters:   " , train[train.FaceSex == 'M'].Cheater.sum()/train[train.FaceSex == 'M'].Cheater.count())


# In[168]:


f,ax=plt.subplots(1,2,figsize=(16,7))
train['Cheater'][train['FaceSex']=='M'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[0],shadow=True)
train['Cheater'][train['FaceSex']=='F'].value_counts().plot.pie(explode=[0,0.2],autopct='%1.1f%%',ax=ax[1],shadow=True)
ax[0].set_title('Cheater (male)')
ax[1].set_title('Cheater (female)')

plt.show()


# In[169]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# for train_ml
scaler.fit(train_ml.drop('Cheater',axis=1))
scaled_features = scaler.transform(train_ml.drop('Cheater',axis=1))
train_ml_sc = pd.DataFrame(scaled_features, columns=train_ml.columns[:-1])

# for test_ml
test_ml.fillna(test_ml.mean(), inplace=True)
# scaler.fit(test_ml)
scaled_features = scaler.transform(test_ml)
test_ml_sc = pd.DataFrame(scaled_features, columns=test_ml.columns)


# In[170]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
train_ml.dropna(inplace=True)


# In[171]:



test_ml = pd.get_dummies(df_test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
test_ml.dropna(inplace=True)


# In[172]:


train_ml = pd.get_dummies(train_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
train_ml.dropna(inplace=True)


test_ml = pd.get_dummies(test_ml, columns=['FaceSex', 'RaterSex'], drop_first=True)
test_ml.dropna(inplace=True)


# In[ ]:




