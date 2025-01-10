import numpy as np 
import sys
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

data = pd.read_csv('Titanic-Dataset.csv')
print(data.info())

print(data.describe())
print(data.head)
print(data.isnull().sum())
data2 = data.drop(columns=['Cabin'])
data2.drop(columns=['PassengerId','Name', 'Ticket'],axis=1,inplace=True) #gereksiz sütunları sildim
data2.to_csv('titanic2.csv', index=False)

print(data2.sample(5))
data2['Family']=data2['SibSp']+data2['Parch']+1 #aile üyelerinin sayısını yeni sütunda gösterdim
print(data2.head())

def myFunc(num):
  if num == 1:
    return 0
  elif num>1 and num<=4:
    return 1
  else:
    return 2
data2['Family'] = data2['Family'].apply(myFunc)  #aile üyelerini 0 1 2 ile kategoriye ayırdım
data2.drop(columns=['SibSp','Parch'],axis=1,inplace=True)
print(data2.sample(5))   

print('Yolcuların seyahat ettiği sınıflar: ')
print(data2['Pclass'].unique()) # seyahat sınıfı zaten kategorik işlem yapmama gerek kalmadı
print('Yolcuların cinsiyet tanımları: ')
print(data2['Sex'].unique()) 

def cinsiyet(num):
    if num == 'female':
        return 0
    elif num == 'male':
        return 1
data2['Sex'] = data2['Sex'].apply(cinsiyet) # cinsiyeti düzenledim
print(data2.head()) 

print("NaN Değerlerin Sayısı: ")
print(data2['Age'].isna().sum())
print("Unique Değerler: ")
print(data2['Age'].unique())

data2['Age'] = data2['Age'].fillna(0)  # age sütunundaki nan değerleri 0 ile doldurdum
print(data2['Age'].head())
print(data2['Age'].isna().sum())
print(data2['Age'].unique())

print('güncel veriseti: ')
print(data2.head())

print(data2['Embarked'].unique())
data2['Embarked'] = data2['Embarked'].fillna(data2['Embarked'].mode()[0])
print(data2['Embarked'].head())
print(data2['Embarked'].unique())

def embarked(num):
    if num == 'S':
        return 0
    elif num == 'C':
        return 1
    elif num == 'Q':
        return 2
data2['Embarked'] = data2['Embarked'].apply(embarked)       
print(data2['Embarked'].head())

print('güncel veriseti: ')
print(data2.head())

print(data2.isnull().sum()) #nan değer kalmadı 

'''
plt.figure(figsize=(8, 6)) # Yaş ve Seyahat Sınıfı arasındaki ilişkiyi görselleştirdim
sns.boxplot(x='Pclass', y='Age', data=data2, palette='Set2')
plt.title('Yaş ve Seyahat Sınıfı İlişkisi')
plt.xlabel('Seyahat Sınıfı')
plt.ylabel('Yaş')
plt.show()
'''
'''
corr = data2.corr() # Korelasyon matrisini görselleştir
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Özellikler Arası Korelasyon')
plt.show()
'''

X = data2.drop(columns=['Survived'])
y = data2['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_basari(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    # Tahminleri al
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılıkları
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    FP = cm[0, 1]
    sensitivity = TP / (TP + FN)  # Duyarlılık
    specificity = TN / (TN + FP)  # Özgüllük
    f1_score_value = 2 * (sensitivity * accuracy) / (sensitivity + accuracy)
    
    # Sonuçları yazdır
    print(f"Model: {model.__class__.__name__}")
    print("Doğruluk:", accuracy)
    print("Duyarlılık:", sensitivity)
    print("Özgüllük:", specificity)
    print("F1-Skor:", f1_score_value)
    print("Karışıklık Matrisi:")
    print(cm)
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    print("\nİlk 10 Tahmin:")
    print("Tahmin Edilen Sınıflar:", y_pred[:10])
    print("Tahmin Olasılıkları:", y_pred_proba[:10])
    
    # Dönüş olarak metrikleri bir sözlükte döndür
    return {
        'model': model.__class__.__name__,
        'f1_score': f1_score_value,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }

# Modellerin listesi
models = [
    LogisticRegression(max_iter=500),
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(probability=True)
]

# Modellerin sonuçlarını karşılaştırma
best_model_info = None
best_f1_score = 0

# En iyi modeli bulmak için döngü
for model in models:
    results = model_basari(model, X_train, X_test, y_train, y_test)
    
    if results['f1_score'] > best_f1_score:
        best_f1_score = results['f1_score']
        best_model_info = results

# En iyi modelin sonuçlarını yazdırma
print("\nEn İyi Model:", best_model_info['model'])
print(f"Doğruluk: {best_model_info['accuracy']}")
print(f"Duyarlılık: {best_model_info['sensitivity']}")
print(f"Özgüllük: {best_model_info['specificity']}")
print(f"F1-Skor: {best_model_info['f1_score']}")
print("Karışıklık Matrisi:")
print(best_model_info['confusion_matrix'])

def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Negatif', 'Pozitif'], yticklabels=['Negatif', 'Pozitif'])
    plt.title(f'{model_name} Karışıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.show()

# Model başarısı fonksiyonunun sonunda karışıklık matrisini görselleştir
def model_basari(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    
    # Tahminleri al
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    TP = cm[1, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    FP = cm[0, 1]
    sensitivity = TP / (TP + FN)  # Duyarlılık
    specificity = TN / (TN + FP)  # Özgüllük
    f1_score_value = 2 * (sensitivity * accuracy) / (sensitivity + accuracy)
    
    # Sonuçları yazdır
    print(f"Model: {model.__class__.__name__}")
    print("Doğruluk:", accuracy)
    print("Duyarlılık:", sensitivity)
    print("Özgüllük:", specificity)
    print("F1-Skor:", f1_score_value)
    print("Karışıklık Matrisi:")
    print(cm)
    print("Sınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

    # Karışıklık matrisini görselleştir
    plot_confusion_matrix(cm, model.__class__.__name__)

    # Dönüş olarak metrikleri bir sözlükte döndür
    return {
        'model': model.__class__.__name__,
        'f1_score': f1_score_value,
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }

# Modellerin listesi
models = [
    LogisticRegression(max_iter=500),
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(probability=True)
]

# Modellerin sonuçlarını karşılaştırma
best_model_info = None
best_f1_score = 0

# En iyi modeli bulmak için döngü
for model in models:
    results = model_basari(model, X_train, X_test, y_train, y_test)
    
    if results['f1_score'] > best_f1_score:
        best_f1_score = results['f1_score']
        best_model_info = results

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Modeli kaydet
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)


# En iyi modelin sonuçlarını yazdırma
print("\nEn İyi Model:", best_model_info['model'])
print(f"Doğruluk: {best_model_info['accuracy']}")
print(f"Duyarlılık: {best_model_info['sensitivity']}")
print(f"Özgüllük: {best_model_info['specificity']}")
print(f"F1-Skor: {best_model_info['f1_score']}")
print("Karışıklık Matrisi:")
print(best_model_info['confusion_matrix'])