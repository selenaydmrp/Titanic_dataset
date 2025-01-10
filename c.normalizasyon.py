import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

data3 = pd.read_csv('titanic3.csv')
print(data3.head())
print(data3.info())

print(data3.describe())
print(data3.head)


print(data3.isnull().sum())

data3['Age'] = data3['Age'].fillna(data3['Age'].mean()) #ortalama ile doldurdum
print(data3['Age'].isnull().sum())

data3['Family']=data3['SibSp']+data3['Parch']+1 #aile üyelerinin sayısını yeni sütunda gösterdim
def myFunc(num):
  if num == 1:
    return 0
  elif num>1 and num<=4:
    return 1
  else:
    return 2
data3['Family'] = data3['Family'].apply(myFunc)  #aile üyelerini 0 1 2 ile kategoriye ayırdım
data3.drop(columns=['SibSp','Parch'],axis=1,inplace=True)


print('Yolcuların seyahat ettiği sınıflar: ')
print(data3['Pclass'].unique()) # seyahat sınıfı zaten kategorik işlem yapmama gerek kalmadı
print('Yolcuların cinsiyet tanımları: ')


def cinsiyet(num):
    if num == 'female':
        return 0
    elif num == 'male':
        return 1
data3['Sex'] = data3['Sex'].apply(cinsiyet) # cinsiyeti düzenledim


print('güncel veriseti: ')
print(data3.head())

def embarked(num):
    if num == 'S':
        return 0
    elif num == 'C':
        return 1
    elif num == 'Q':
        return 2
data3['Embarked'] = data3['Embarked'].apply(embarked)  
data3['Embarked'] = data3['Embarked'].fillna(data3['Embarked'].median()) #yolcuların bindiği limanları ortalama ile doldurdum   
print(data3['Embarked'].isnull().sum())  


print('güncel veriseti: ')
print(data3.head())
print(data3.isnull().sum()) #nan değer kalmadı


print(data3['Age'].isnull().sum())

data3['Age'] = data3['Age'].apply(lambda x: x * 100 if x < 1 else x).astype(int)
print(data3['Age'])

print("Min Age:", data3['Age'].min())
print("Max Age:", data3['Age'].max())


df = pd.DataFrame(data3) #normalizasyon uygulayıp yeni bir dataframe aktardım (MİN-MAX)
df['Age_Normalized'] = (df['Age'] - df['Age'].min()) / (df['Age'].max() - df['Age'].min())
df.drop(columns=['Age'],axis=1,inplace=True)
print("Normalized DataFrame:")
df.to_csv("normalized_data.csv", index=False)
print(df)

print(df)

print("Fare Sütununun Benzersiz Değerleri:")
print(df['Fare'].unique())
print("\nFare Sütununun Min ve Max Değerleri:")
print("Min Değer:", df['Fare'].min())
print("Max Değer:", df['Fare'].max())

#FARE KOLONU ÇOK GENİŞ ARALIKTAYDI Logaritmik dönüşüm UYGULADIM 
df['Fare_yeni'] = np.log1p(df['Fare'])
print(df[['Fare', 'Fare_yeni']].head())

print("Min Değer:", df['Fare_yeni'].min())
print("Max Değer:", df['Fare_yeni'].max())

print('Normalizasyon sonucu fiyatladırmanın (Fare_yeni sütunu) son hali: ')
df['Fare_yeni'] = zscore(df['Fare_yeni']) #Z SKOR NORMALİZASYONU UYGULADIM
print(df[['Fare_yeni']].head())

print("Min Değer:", df['Fare_yeni'].min())
print("Max Değer:", df['Fare_yeni'].max())

X = df.drop(columns=['Survived'])  
y = df['Survived'] 

kf = KFold(n_splits=5, shuffle=True, random_state=42)  #k fold çalışması
accuracies = []
sensitivities = []
specificities = []
f1_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() 

    sensitivity = recall_score(y_test, y_pred)
    sensitivities.append(sensitivity)

    specificity = tn / (tn + fp)
    specificities.append(specificity)

    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)

print(f"Doğruluk Skorları (Accuracy): {accuracies}")
print(f"Ortalama Doğruluk Skoru: {np.mean(accuracies)}")
print(f"Duyarlılık Skorları (Sensitivity): {sensitivities}")
print(f"Ortalama Duyarlılık: {np.mean(sensitivities)}")
print(f"Özgüllük Skorları (Specificity): {specificities}")
print(f"Ortalama Özgüllük: {np.mean(specificities)}")
print(f"F1-Skorları: {f1_scores}")
print(f"Ortalama F1-Skoru: {np.mean(f1_scores)}")

matris = confusion_matrix(y_test, y_pred)
print("Son Fold için Karışıklık Matrisi:")
print(matris)


accuracy_scores = [0.8212290502793296, 0.8033707865168539, 0.8258426966292135, 0.8033707865168539, 0.8202247191011236]
sensitivity_scores = [0.7837837837837838, 0.6268656716417911, 0.8285714285714286, 0.6515151515151515, 0.7384615384615385]
specificity_scores = [0.8476190476190476, 0.9099099099099099, 0.8240740740740741, 0.8928571428571429, 0.8672566371681416]
f1_scores = [0.7837837837837838, 0.7058823529411765, 0.7891156462585034, 0.7107438016528925, 0.75]

avg_accuracy = np.mean(accuracy_scores)
avg_sensitivity = np.mean(sensitivity_scores)
avg_specificity = np.mean(specificity_scores)
avg_f1 = np.mean(f1_scores)

metrics = ['Accuracy', 'Sensitivity', 'Specificity', 'F1-Score']
values = [accuracy_scores, sensitivity_scores, specificity_scores, f1_scores]
avg_values = [avg_accuracy, avg_sensitivity, avg_specificity, avg_f1]


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracy_scores = []
sensitivity_scores = []
specificity_scores = []
f1_scores = []

for train_index, test_index in kfold.split(X, y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                           
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
   
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)
   
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn)
    sensitivity_scores.append(sensitivity)
    
    specificity = tn / (tn + fp)
    specificity_scores.append(specificity)
    
    f1 = f1_score(y_test, y_pred)
    f1_scores.append(f1)
                
    print(f"Fold Karışıklık Matrisi:\n{cm}")
    print(f"Doğruluk: {accuracy:.4f}, Duyarlılık: {sensitivity:.4f}, Özgüllük: {specificity:.4f}, F1-Skoru: {f1:.4f}\n")

print(f"Ortalama Doğruluk Skoru: {np.mean(accuracy_scores):.4f}")
print(f"Ortalama Duyarlılık Skoru: {np.mean(sensitivity_scores):.4f}")
print(f"Ortalama Özgüllük Skoru: {np.mean(specificity_scores):.4f}")
print(f"Ortalama F1 Skoru: {np.mean(f1_scores):.4f}")


knn = KNeighborsClassifier(n_neighbors=5) #knn çalışması
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"KNN Doğruluk: {accuracy:.4f}")
print("KNN Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_knn))
print("KNN Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_knn))


karar_agaci = DecisionTreeClassifier(random_state=42)  #karar ağacı çalışması
karar_agaci.fit(X_train, y_train)  
y_pred_dt = karar_agaci.predict(X_test)  
print(f"Decision Tree Doğruluk: {accuracy_score(y_test, y_pred_dt)}")
print("Decision Tree Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred_dt))
print("Decision Tree Karışıklık Matrisi:")
print(confusion_matrix(y_test, y_pred_dt))


