import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Titanic-Dataset.csv')
print(data.head())
print(data.info())

print(data.describe())
print(data.head)

data3 = data.drop(columns=['Cabin'])
data3.drop(columns=['PassengerId','Name', 'Ticket'],axis=1,inplace=True) #gereksiz sütunları sildim
data3.to_csv('titanic3.csv', index=False)

print(data3.isnull().sum())

data3['Age'] = data3['Age'].fillna(data3['Age'].median()) #ortalama ile doldurdum
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


survived_sayilari = data3['Survived'].value_counts()
print(survived_sayilari)

hayatta_kalanlar = data3[data3['Survived'] == 1] #DENGESİZLİK  OLUŞTURDUM
hayatta_kalmayanlar = data3[data3['Survived'] == 0]
hayatta_kalanlar_orneklenmis = hayatta_kalanlar.sample(frac=0.2, random_state=42)
dengesiz_veri = pd.concat([hayatta_kalanlar_orneklenmis, hayatta_kalmayanlar])
dengesiz_veri = dengesiz_veri.sample(frac=1, random_state=42).reset_index(drop=True)
print(dengesiz_veri['Survived'].value_counts())

dengesiz_veri.to_csv('imbalanced_titanic.csv', index=False) # DENGESİZ VERİ SETİNİ KAYDETTİM
print(dengesiz_veri.head())               

X = dengesiz_veri.drop(columns=['Survived']) #dengesizlik ile eğitim test uyguladım
y = dengesiz_veri['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Doğruluk:", accuracy_score(y_test, y_pred))
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))
print("Karışıklık Matrisi:\n", confusion_matrix(y_test, y_pred))


plt.figure(figsize=(12, 6)) #dengesizlik için öncesi ve sonrası görseli
plt.subplot(1, 2, 1)
data['Survived'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Orijinal Veri Seti')
plt.xlabel('Survived')
plt.ylabel('Frekans')
plt.subplot(1, 2, 2)
dengesiz_veri['Survived'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Dengesiz Veri Seti')
plt.xlabel('Survived')
plt.ylabel('Frekans')
plt.tight_layout()
plt.show()


cm = confusion_matrix(y_test, y_pred)  # Karışıklık matrisi görseli
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Hayatta Kalmayan', 'Hayatta Kalan'], yticklabels=['Hayatta Kalmayan', 'Hayatta Kalan'])
plt.title('Karışıklık Matrisi')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()


X = dengesiz_veri.drop('Survived', axis=1) #ros uygulaması
y = dengesiz_veri['Survived']
ros = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_res, y_res = ros.fit_resample(X, y)


plt.figure(figsize=(6, 6)) #ros görselleştirme
plt.bar(y_res.value_counts().index, y_res.value_counts().values, color=['skyblue', 'salmon'])
plt.title('Dengeye Getirilmiş Veri Seti (Random OverSampling)')
plt.xlabel('Survived')
plt.ylabel('Frekans')
plt.show()
print(f'Dengeye Getirilmiş Veri Seti: \n{y_res.value_counts()}')


smote = SMOTE(sampling_strategy='minority', random_state=42) #smothe ile dengesizlikle başa çıkma
X_res_smote, y_res_smote = smote.fit_resample(X, y)


plt.figure(figsize=(6, 6)) #smothe görselleştirme
plt.bar(y_res_smote.value_counts().index, y_res_smote.value_counts().values, color=['skyblue', 'salmon'])
plt.title('Dengeye Getirilmiş Veri Seti (SMOTE)')
plt.xlabel('Survived')
plt.ylabel('Frekans')
plt.show()
print(f'Dengeye Getirilmiş Veri Seti (SMOTE): \n{y_res_smote.value_counts()}')


X_res_smote = pd.DataFrame(X_res_smote, columns=X.columns) #smothe ile eğitim ve test uyguladım
y_res_smote = pd.Series(y_res_smote)
X_train, X_test, y_train, y_test = train_test_split(X_res_smote, y_res_smote, test_size=0.2, random_state=42)


def model_basari(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)  # Modeli eğitim verisiyle eğit
    y_pred = model.predict(X_test)  # Sınıflandırma tahminleri
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılıkları
    
    cm = confusion_matrix(y_test, y_pred)  # Karışıklık matrisi
    accuracy = accuracy_score(y_test, y_pred)  # Doğruluk
    TP = cm[1, 1]
    FN = cm[1, 0]
    TN = cm[0, 0]
    FP = cm[0, 1]
    sensitivity = TP / (TP + FN)  # Duyarlılık
    specificity = TN / (TN + FP)  # Özgüllük
    f1_score_value = f1_score(y_test, y_pred)  # F1 skoru
    
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
    RandomForestClassifier(random_state=42),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(random_state=42)
]

best_model_info = None
best_f1_score = 0

# En iyi modeli bulmak için döngü
for model in models:
    results = model_basari(model, X_train, X_test, y_train, y_test)
    
    # En yüksek F1 skoru ile en iyi modelin bilgisini sakla
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

# MODELLER İÇİN TOPLU GÖRSELLEŞTİRME
cm_rf = confusion_matrix(y_test, model.predict(X_test))  # Son kullanılan modelin CM
cm_knn = confusion_matrix(y_test, KNeighborsClassifier(n_neighbors=5).fit(X_res_smote, y_res_smote).predict(X_test))
cm_dt = confusion_matrix(y_test, DecisionTreeClassifier(random_state=42).fit(X_res_smote, y_res_smote).predict(X_test))

# Model doğruluklarını hesaplama
accuracy_rf = accuracy_score(y_test, model.predict(X_test))
accuracy_knn = accuracy_score(y_test, KNeighborsClassifier(n_neighbors=5).fit(X_res_smote, y_res_smote).predict(X_test))
accuracy_dt = accuracy_score(y_test, DecisionTreeClassifier(random_state=42).fit(X_res_smote, y_res_smote).predict(X_test))

# Her model için doğruluk oranlarını yazdırma
print(f"Random Forest Doğruluk: {accuracy_rf}")
print(f"KNN Doğruluk: {accuracy_knn}")
print(f"Decision Tree Doğruluk: {accuracy_dt}")
