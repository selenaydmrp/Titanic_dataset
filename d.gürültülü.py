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

data3 = pd.read_csv('verisetim_guncel.csv')
print(data3.head())
print(data3.info())
print(data3.describe())

print(data3['Pclass'].unique())
print(data3['Sex'].unique())
print(data3['Embarked'].unique())
print(data3.isnull().sum())

X = data3.drop('Survived', axis=1)  # Özellikler
y = data3['Survived']               # Hedef değişken

def add_noise_to_labels(y, noise_level=0.05):
    y = y.copy()  # Orijinal veriyi bozmamak için kopyalıyoruz
    n_samples = len(y)
    n_noise = int(noise_level * n_samples)
    
    noise_indices = np.random.choice(n_samples, n_noise, replace=False)
    y.iloc[noise_indices] = np.random.choice(np.unique(y), n_noise)  
    return y


y_noisy = add_noise_to_labels(y, noise_level=0.05)
print(y_noisy.head())

def add_noise_to_features(X, noise_level=0.05):
    noise = np.random.normal(0, noise_level, X.shape)  
    X_noisy = X + noise
    return X_noisy

X_noisy = add_noise_to_features(X.select_dtypes(include=[np.number]), noise_level=0.05)
print(X_noisy.head())
print(y_noisy.head())

print("Orijinal y:", y.values[:10])  #HEDEF DEĞİŞKEN İÇİN NE KADARLIKK VERİ GÜRÜLTÜLÜ HALE GETİRİLDİ İLK HALİ VE SON HALİ KIYASLAMASI
print("Gürültülü y_noisy:", y_noisy.values[:10])  
print("Değiştirilen örnek sayısı:", np.sum(y != y_noisy))

print("Orijinal x:", X.values[:10])  # İlk 10 değer
print("Gürültülü x_noisy:", X_noisy.values[:10])  # İlk 10 değer
print("Değiştirilen örnek sayısı:", np.sum(X != X_noisy))

X_train, X_test, y_train, y_test = train_test_split(X_noisy, y_noisy, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5) # KNN algoritması
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Doğruluk :", accuracy_score(y_test, y_pred_knn))
print("KNN Sınıflandırma raporu:\n", classification_report(y_test, y_pred_knn))

rf = RandomForestClassifier(n_estimators=100, random_state=42) #Random Forest algoritması
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Doğruluk Oranı:", accuracy_score(y_test, y_pred_rf))
print("Random Forest Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_rf))

svm = SVC(kernel='rbf', random_state=42)  #SVM algoritması
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

print("SVM Doğruluk Oranı:", accuracy_score(y_test, y_pred_svm))
print("SVM Sınıflandırma Raporu:\n", classification_report(y_test, y_pred_svm))

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42)
}


accuracies = {}
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies[model_name] = acc

best_model_name = max(accuracies, key=accuracies.get)
best_model = models[best_model_name]
print(f"En iyi model: {best_model_name} (Doğruluk: {accuracies[best_model_name]:.2f})")

def model_basarisi(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)  
    specificity = cm[1, 1] / (cm[1, 1] + cm[0, 1])  
    f1 = f1_score(y_true, y_pred)
    
    print("Karışıklık Matrisi:")
    print(cm)
    print("\nDoğruluk:", accuracy)
    print("Duyarlılık (Recall):", sensitivity)
    print("Özgüllük:", specificity)
    print("F1 Skoru:", f1)


y_pred_test = best_model.predict(X_test)
y_pred_proba_test = best_model.predict_proba(X_test)

print("\n--- Test Verisi Tahmin Sonuçları ---")
print("Tahmin Edilen Sınıflar (İlk 10):", y_pred_test[:10])
print("Tahmin Olasılıkları (İlk 10):")
for i, probs in enumerate(y_pred_proba_test[:10]):
    print(f"Örnek {i+1}: {probs}")

print("\n--- Sınıflandırma Raporu ---")
print(classification_report(y_test, y_pred_test))

model_basarisi(y_test, y_pred_test)