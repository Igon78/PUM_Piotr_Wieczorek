import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split


pd.set_option('display.max_columns', None)
#pobranie danych
train_data = pd.read_csv("przygotowanie_train.csv")

#sprawdzenie wymiarów
print(train_data.shape) #wymiary
print(train_data.info()) #podstawowe informacje o kolumnach
print(train_data.head(5)) #część tabeli

#usuwanie duplikatów
train_data.drop_duplicates(subset='ID',inplace=True)
#Zmiana typów danych – usunięcie/uzupełnienie braków
features_convert = ['Age','Annual_Income','Num_of_Loan','Num_of_Delayed_Payment','Changed_Credit_Limit',
                    'Outstanding_Debt','Amount_invested_monthly','Monthly_Balance']
#sprawdzanie czy nie ma błędów w danych
for feature in features_convert:
    uniques = train_data[feature].unique()
    print('Feature:','\n',feature, '\n', uniques,'\n','--'*40,'\n')
#sprawdzanie liczby danych kategorycznych
cat_attributes = ['ID','Customer_ID','Month','Name','SSN','Occupation','Type_of_Loan',
                          'Credit_Mix','Credit_History_Age','Payment_of_Min_Amount','Payment_Behaviour']

'''for feature in cat_attributes:
    plt.figure(figsize=(20, 18))
    sns.countplot(x=train_data[feature])
    plt.show()'''

#usunięcie zbędnych znaków '-', '_'
for feature in features_convert:
    train_data[feature] = train_data[feature].str.strip('-_')
#zastąpienie pustych kolumn NaN
for feature in features_convert:
    train_data[feature] = train_data[feature].replace({'':np.nan})

#zmiana typu zmiennych ilościowych
for feature in features_convert:
    train_data[feature] = train_data[feature].astype('float64')

#sprawdzenie w których kolumnach są puste wartości
print(train_data.isnull().sum())
#uzupelnienie braków srednią lub zerem
train_data['Monthly_Inhand_Salary'] = train_data['Monthly_Inhand_Salary'].fillna(method='pad')
train_data['Monthly_Balance'] = train_data['Monthly_Balance'].fillna(method='pad')
#train_data.drop(['Name'],inplace=True,axis=1)
train_data['Type_of_Loan'] = train_data['Type_of_Loan'].fillna('No_loan')
'''
train_data['Num_of_Delayed_Payment'] = train_data['Num_of_Delayed_Payment'].fillna(0)
train_data['Changed_Credit_Limit'] = train_data['Changed_Credit_Limit'].fillna(0)
train_data['Num_Credit_Inquiries'] = train_data['Num_Credit_Inquiries'].fillna(0)
train_data['Credit_History_Age'] = train_data['Credit_History_Age'].fillna(0)
train_data['Amount_invested_monthly'] = train_data['Amount_invested_monthly'].fillna(0)'''
train_data.dropna(inplace=True)

print(train_data.isnull().sum())

#kodowanie zmiennych kategorialnych
le = LabelEncoder()
train_data[cat_attributes].apply(le.fit_transform)
#sprawdzenie transformacji
print(train_data.head())

print(train_data['Credit_History_Age'].head())

#korelacje cech numerycznych
col_float = ['Age','Annual_Income','Delay_from_due_date','Num_of_Delayed_Payment',
             'Outstanding_Debt','Total_EMI_per_month','Monthly_Balance']

plt.figure(figsize = (20,18))
sns.heatmap(train_data[col_float].corr(),annot=True,linewidths=0.1,cmap='Blues')
plt.title('Numerical Features Correlation')
plt.show() # Outstanding_Debt jest silnie skorelowany z Delay from due date

#standaryzacja danych
scaler = MinMaxScaler()
for i in train_data[col_float]:
    train_data[i] = scaler.fit_transform(train_data[[i]])

train_data.head()

#podzielenie zestawu danych
x = train_data.drop(['Credit_Score'],axis=1)
y = train_data['Credit_Score']
x_train,y_train,x_test,y_test = train_test_split(x,y,test_size=0.25,random_state=42)
print(y.head())




