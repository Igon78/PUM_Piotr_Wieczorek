#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split

pd.set_option('display.max_columns', None)


# Pobranie danych i sprawdzenie podstawowych informacji

# In[2]:


train_data = pd.read_csv("przygotowanie_train.csv")
train_data.head()


# In[3]:


train_data.info() #sprawdzenie typu kolumn i gdzie występują nulle


# Czyszczenie danych

# In[4]:


#wydzielenie list kolumn numerycznych i kategorialnych
numeric_cols = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts',
                'Num_Credit_Card','Interest_Rate','Monthly_Balance',
                'Num_of_Loan','Delay_from_due_date','Num_of_Delayed_Payment',
                'Changed_Credit_Limit','Num_Credit_Inquiries','Outstanding_Debt',
               'Credit_Utilization_Ratio','Total_EMI_per_month','Amount_invested_monthly']
categorical_cols =['ID','Customer_ID','Month','Name','SSN','Occupation',
                   'Type_of_Loan','Credit_Mix','Credit_History_Age',
                   'Payment_of_Min_Amount','Payment_Behaviour','Credit_Score']


# In[5]:


#wydzielenie kolumn do zmiany typu
numeric_convert = ['Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit',
                    'Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']
#sprawdzenie czy w danych numerycznych, które mają typ object, nie ma błędów
for feature in numeric_convert:
    uniques = train_data[feature].unique()
    print('Feature:','\n',feature, '\n', uniques,'\n','--'*40,'\n')


# In[6]:


#usunięcie zbędnych znaków '-', '_'
for feature in numeric_convert:
    train_data[feature] = train_data[feature].str.strip('-_')
#zastąpienie pustych kolumn NaN
for feature in numeric_convert:
    train_data[feature] = train_data[feature].replace({'':np.nan})

#zmiana typu zmiennych ilościowych
for feature in numeric_convert:
    train_data[feature] = train_data[feature].astype('float64')


# In[7]:


for feature in numeric_convert:
    uniques = train_data[feature].unique()
    print('Feature:','\n',feature, '\n', uniques,'\n','--'*40,'\n')
#wszystkie kolumny ilościowe są teraz typu float i nie posiadają błędnych znaków


# Usunięcie lub uzupełnienie pustych wartości

# In[8]:


#sprawdzenie w których kolumnach są puste wartości i ile ich jest
train_data.isnull().sum()


# In[9]:


data_clean = train_data.copy()
#usunięcie duplikatów według kolumny ID
data_clean.drop_duplicates(subset='ID',inplace=True)
#lista z kolumnami w których są puste wartości
cols_nulls = ['Name', 'Monthly_Inhand_Salary','Type_of_Loan',
              'Num_of_Delayed_Payment','Num_Credit_Inquiries','Credit_History_Age',
             'Amount_invested_monthly','Monthly_Balance']

#uzupełnienie pustych wartości średnią
data_clean['Monthly_Inhand_Salary'].fillna(method='pad',inplace=True)
data_clean['Monthly_Balance'].fillna(method='pad',inplace=True)
data_clean['Changed_Credit_Limit'].fillna(method='pad',inplace=True)
data_clean['Amount_invested_monthly'].fillna(method='pad',inplace=True)

#uzupełnienie pustych wartości gdy klient nie wziął żadnej pożyczki
data_clean.loc[(data_clean['Num_of_Loan']==0) & (data_clean['Type_of_Loan'].isnull()),'Type_of_Loan']='No Loan'

#uzupełnienie pustych wartości gdy klient nie prosił o kartę kredytową
data_clean.loc[(data_clean['Num_Credit_Inquiries']==0)&(data_clean['Credit_History_Age'].isnull()),'Credit_History_Age']='No credit'

#usunięcie kolumn nieistotnych dla tworzenia modelu
data_clean.drop(columns=['ID','Customer_ID','SSN','Name'],inplace=True)

#usunięcie wierszy z pustymi wartościami
data_clean.dropna(subset=['Num_of_Delayed_Payment','Num_Credit_Inquiries','Type_of_Loan','Credit_History_Age'],inplace=True)

data_clean.isnull().sum()


# In[10]:


#korelacje cech numerycznych
plt.figure(figsize = (20,18))
sns.heatmap(data_clean[numeric_cols].corr(),annot=True,linewidths=0.1,cmap='Blues')
plt.title('Numerical Features Correlation')
plt.show() 


# In[11]:


#Usunięcie znacząco odstających danych, 
#zakresy dla których były usuwane dane zostały sprawdzone oddzielnie dla każdego atrybutu numerycznego
#z wykorzystaniem value_counts.sort_index() lub value_counts.sort_values()
data_clean.drop(data_clean.loc[data_clean['Age']>95].index,inplace=True)
data_clean.drop(data_clean.loc[data_clean['Num_Bank_Accounts']>11].index,inplace=True)
data_clean.drop(data_clean.loc[data_clean['Num_Bank_Accounts']<0].index,inplace=True)
data_clean.drop(data_clean.loc[data_clean['Num_Credit_Card']>11].index,inplace=True)
data_clean.drop(data_clean.loc[data_clean['Interest_Rate']>34].index,inplace=True)
data_clean.loc[data_clean['Num_of_Loan']==100].Num_of_loan = 10 #przy zamianie zmiennych na float 10 zamieniono na 100.00
data_clean.drop(data_clean.loc[data_clean['Num_of_Loan']>10].index,inplace=True)
data_clean.drop(data_clean.loc[data_clean['Num_of_Delayed_Payment']>28].index,inplace=True)
data_clean.drop(data_clean.loc[data_clean['Num_Credit_Inquiries']>17].index,inplace=True)
data_clean.info()


# In[12]:


data_clean['Credit_History_Age'].value_counts()


# In[13]:


#zamiana atrybutu Credit History Age na liczbę miesięcy
def transform_history_age(v):
    if(v=='No credit'): return 0
    wynik =int(v[0:v.find('Years')-1])*12+int(v[v.find('d')+1:v.find('Months')-1])
    return wynik

data_clean['Credit_History_Age'] = data_clean['Credit_History_Age'].apply(transform_history_age)
numeric_cols.append('Credit_History_Age')
data_clean['Credit_History_Age'].value_counts()


# In[14]:


for column in numeric_cols:
    Q1 = data_clean[column].quantile(0.25)
    Q3 = data_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    data_clean.drop(data_clean.loc[data_clean[column]>(Q3+1.5*IQR)].index,inplace=True)
    data_clean.drop(data_clean.loc[data_clean[column]<(Q1-1.5*IQR)].index,inplace=True)


# In[15]:


scaler = MinMaxScaler()
for i in data_clean[numeric_cols]:
    data_clean[i]=scaler.fit_transform(data_clean[[i]])
data_clean.head()


# Transformacja danych kategorialnych

# In[16]:


categorical_cols =['Month','Occupation','Type_of_Loan','Credit_Mix',
        'Payment_of_Min_Amount','Payment_Behaviour','Credit_Score']

#data_clean['Payment_of_Min_Amount'].value_counts()


data_clean[categorical_cols] = data_clean[categorical_cols].apply(LabelEncoder().fit_transform)
    
data_clean.head()


# In[17]:


#korelacja cech ze zmienną Credit Score
pd.DataFrame(abs(data_clean.corr()['Credit_Score'].drop('Credit_Score')*100).sort_values(ascending=False)).plot.bar(figsize = (10,8))
#najmniej skorelowane cechy z atrybutem Credit Score to Amount_invested_monthly, Monthly_Inhand_salary, ich usunięcie nie wpłynie znacząco na efektywność modelu

data_clean.drop(columns=['Amount_invested_monthly','Monthly_Inhand_Salary','Type_of_Loan'],inplace=True)


# Podzielenie zbioru na zbiór treningowy i testowy

# In[39]:


x = data_clean.drop(['Credit_Score'],axis=1)
y = data_clean['Credit_Score']
x.head()


# Redukacja wymiarowości

# In[40]:


from sklearn.decomposition import PCA #redukcja wymiarowości z analizą głównych składowych
pca = PCA(n_components=10) #redukcja 20 atrybutów na 10 atrybutów 
pca_features = pca.fit_transform(x)

pca_df = pd.DataFrame(data=pca_features, columns=['PC1', 'PC2', 'PC3','PC4', 'PC5', 'PC6','PC7', 'PC8', 'PC9','PC10'])


# In[41]:


#wykres
plt.bar(range(1,len(pca.explained_variance_)+1),pca.explained_variance_) 


# In[42]:


x_train,x_test,y_train,y_test = train_test_split(pca_df,y,test_size=0.25,random_state=42)
x_train.head()


# In[ ]:




