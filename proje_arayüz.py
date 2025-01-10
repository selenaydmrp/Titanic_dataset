import sys
import pandas as pd
import numpy as np
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtWidgets import QTableWidgetItem
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score

class Ui_Form(QtWidgets.QWidget):
    def __init__(self):  # __init__ yanlış yazılmıştı, düzeltildi
        super().__init__()
        self.setupUi()

    def setupUi(self):
        self.setWindowTitle("Data Processing App")
        self.resize(800, 600)

        self.pushButton = QtWidgets.QPushButton("Load Dataset", self)
        self.pushButton.setGeometry(50, 40, 121, 51)
        self.pushButton.clicked.connect(self.load_dataset)

        self.spinBox = QtWidgets.QSpinBox(self)
        self.spinBox.setGeometry(30, 120, 171, 22)

        self.pushButton_2 = QtWidgets.QPushButton("Add NaNs", self)
        self.pushButton_2.setGeometry(220, 120, 93, 28)
        self.pushButton_2.clicked.connect(self.add_nans)

        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(32, 160, 161, 21)
        self.comboBox.addItems([
            "Forward Fill",
            "Backward Fill",
            "Mean Fill",
            "Remove Rows",
            "Fill with 0"
        ])

        self.pushButton_3 = QtWidgets.QPushButton("Impute NaNs", self)
        self.pushButton_3.setGeometry(220, 160, 93, 28)
        self.pushButton_3.clicked.connect(self.impute_nans)

        self.comboBox_2 = QtWidgets.QComboBox(self)
        self.comboBox_2.setGeometry(30, 200, 161, 21)
        self.comboBox_2.addItems(["Min-Max Scaling", "Z-Score Scaling"])

        self.pushButton_4 = QtWidgets.QPushButton("Normalize", self)
        self.pushButton_4.setGeometry(220, 200, 93, 28)
        self.pushButton_4.clicked.connect(self.normalize_data)

        self.comboBox_3 = QtWidgets.QComboBox(self)
        self.comboBox_3.setGeometry(30, 250, 161, 22)
        self.comboBox_3.addItems(["KNN", "Random Forest", "Decision Tree"])

        self.pushButton_5 = QtWidgets.QPushButton("Classify", self)
        self.pushButton_5.setGeometry(220, 250, 93, 28)
        self.pushButton_5.clicked.connect(self.classify_data)

        self.label = QtWidgets.QLabel("Metrics:", self)
        self.label.setGeometry(350, 170, 300, 28)
        self.label_2 = QtWidgets.QLabel("Normalization Metrics:", self)  # Yeni widget
        self.label_2.setGeometry(350, 210, 300, 28)

        self.label_3 = QtWidgets.QLabel("Classification Metrics:", self)  # Yeni widget
        self.label_3.setGeometry(350, 250, 300, 28)
        self.tableWidget = QtWidgets.QTableWidget(self)
        self.tableWidget.setGeometry(50, 300, 700, 250)
        # Yeni LineEdit ve Label Tanımlamaları
        self.label_7 = QtWidgets.QLabel("Pclass", self)
        self.label_7.setGeometry(800, 60, 55, 16)
        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(870, 60, 137, 22)

        self.label_8 = QtWidgets.QLabel("Sex", self)
        self.label_8.setGeometry(800, 90, 55, 16)
        self.lineEdit_2 = QtWidgets.QLineEdit(self)
        self.lineEdit_2.setGeometry(870, 90, 137, 22)

        self.label_9 = QtWidgets.QLabel("Age", self)
        self.label_9.setGeometry(800, 120, 55, 16)
        self.lineEdit_3 = QtWidgets.QLineEdit(self)
        self.lineEdit_3.setGeometry(870, 120, 137, 22)

        self.label_6 = QtWidgets.QLabel("Fare", self)
        self.label_6.setGeometry(800, 150, 55, 16)
        self.lineEdit_4 = QtWidgets.QLineEdit(self)
        self.lineEdit_4.setGeometry(870, 150, 137, 22)

        self.label_5 = QtWidgets.QLabel("Embarked", self)
        self.label_5.setGeometry(800, 180, 55, 16)
        self.lineEdit_5 = QtWidgets.QLineEdit(self)
        self.lineEdit_5.setGeometry(870, 180, 137, 22)

        self.label_4 = QtWidgets.QLabel("Family", self)
        self.label_4.setGeometry(800, 210, 55, 16)
        self.lineEdit_6 = QtWidgets.QLineEdit(self)
        self.lineEdit_6.setGeometry(870, 210, 137, 22)

        self.pushButton_6 = QtWidgets.QPushButton("Predict", self)
        self.pushButton_6.setGeometry(850, 260, 93, 28)
        self.pushButton_6.clicked.connect(self.predict_survival)

        self.pushButton_8 = QtWidgets.QPushButton("Apply SMOTE", self)
        self.pushButton_8.setGeometry(850, 600, 150, 28)
        self.pushButton_8.clicked.connect(self.apply_smote)

        self.label_10 = QtWidgets.QLabel("SMOTE Status", self)
        self.label_10.setGeometry(800, 650, 300, 50)
        self.label_10.setWordWrap(True)  # Çok satırlı metin desteği

        self.label_11 = QtWidgets.QLabel(self)
        self.label_11.setGeometry(800, 650, 300, 300)
        self.label_11.setScaledContents(True)
        self.textEdit = QtWidgets.QTextEdit(self)
        self.textEdit.setGeometry(800, 400, 300, 200)

        self.pushButton_7 = QtWidgets.QPushButton("Perform K-Fold", self)
        self.pushButton_7.setGeometry(850, 350, 150, 28)
        self.pushButton_7.clicked.connect(self.perform_kfold)


    def perform_kfold(self):
        if self.dataset is not None:
            try:
                # Özellikler ve hedef değişkeni ayırma
                X = self.dataset.drop(columns=[self.target_column])
                y = self.dataset[self.target_column]

                # Seçilen modeli comboBox_3'ten al
                selected_model = self.comboBox_3.currentText()

                if selected_model == "KNN":
                    model = KNeighborsClassifier()
                elif selected_model == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif selected_model == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                else:
                    self.textEdit.append("Error: No valid model selected.")
                    return

                # K-Fold çapraz doğrulama
                kfold = KFold(n_splits=5, shuffle=True, random_state=42)
                scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

                # Sonuçları metin kutusuna yazdırma
                self.textEdit.clear()
                self.textEdit.append(f"K-Fold Cross Validation Results for {selected_model}:\n")
                for fold_idx, score in enumerate(scores, start=1):
                    self.textEdit.append(f"Fold {fold_idx}: {score:.4f}")
                self.textEdit.append(f"\nAverage Accuracy: {scores.mean():.4f}")
            except Exception as e:
                self.textEdit.append(f"Error during K-Fold validation: {e}")

    def apply_smote(self):
        if self.dataset is not None:
            try:
                # Sayısal sütunları ve hedef değişkeni ayırın
                X = self.dataset.drop(columns=[self.target_column]).select_dtypes(include=[np.number])
                y = self.dataset[self.target_column]

                # SMOTE uygulaması
                smote = SMOTE(random_state=42)
                X_resampled, y_resampled = smote.fit_resample(X, y)

                # Yeni veri setini güncelleyin
                self.dataset = pd.DataFrame(X_resampled, columns=X.columns)
                self.dataset[self.target_column] = y_resampled

                # Tabloyu güncelleyin
                self.show_data_in_table()

                # Sınıf dağılımını görselleştirin
                class_counts = pd.Series(y_resampled).value_counts()
                plt.figure(figsize=(6, 4))
                plt.bar(class_counts.index, class_counts.values, color=['blue', 'red'])
                plt.xticks([0, 1], ['Class 0', 'Class 1'])
                plt.ylabel("Frequency")
                plt.title("Class Distribution After SMOTE")
                plt.show()

                # Sonuç mesajını güncelleyin
                self.label_10.setText("SMOTE applied successfully. Class distribution balanced.")
            except Exception as e:
                # Hata durumunda mesaj gösterin
                self.label_10.setText(f"Error applying SMOTE: {e}")




    def predict_survival(self):
        # Kullanıcı girdilerini al
        pclass = int(self.lineEdit.text())
        sex = 1 if self.lineEdit_2.text().lower() == "male" else 0  # "male" için 1, diğerleri için 0
        age = float(self.lineEdit_3.text())
        fare = float(self.lineEdit_4.text())
       # embarked = ord(self.lineEdit_5.text().upper()) - ord('A')  # Harfleri sayıya dönüştür
        embarked_input = self.lineEdit_5.text().upper()
        embarked_mapping = {"S": 0, "C": 1, "Q" : 2}  # S için 1, C için 2
        if embarked_input not in embarked_mapping:
            raise ValueError("Embarked must be 'S', 'C', or 'Q'")
        embarked = embarked_mapping[embarked_input]
        family = int(self.lineEdit_6.text())

        # Girdi özelliklerini düzenle
        input_features = np.array([[pclass, sex, age, fare, embarked, family]])

        # Modeli eğit ve predict_proba kullan
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli sadece eğitim verileri üzerinde eğit
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
       # model = RandomForestClassifier()  # Varsayılan olarak Random Forest kullanılıyor
       # model.fit(X, y)

        probabilities = model.predict_proba(input_features)
        survival_prob = probabilities[0][1]  # Hayatta kalma olasılığı
        death_prob = probabilities[0][0]  # Ölüm olasılığı

        # Sonuçları label_10'da göster
        self.label_10.setText(f"Survival: {survival_prob:.2%}, Death: {death_prob:.2%}")


    def load_dataset(self):
        file_path = r"C:\\Users\\Acer\\Desktop\\Makine Projesi_\\verisetim_guncel.csv"
        self.dataset = pd.read_csv(file_path)
        self.target_column = "Survived"
        self.show_data_in_table()

    def show_data_in_table(self):
        self.tableWidget.setColumnCount(len(self.dataset.columns))
        self.tableWidget.setRowCount(len(self.dataset))
        self.tableWidget.setHorizontalHeaderLabels(self.dataset.columns)
        
        for i in range(len(self.dataset)):
            for j in range(len(self.dataset.columns)):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.dataset.iloc[i, j])))

    def add_nans(self):
        percentage = self.spinBox.value()
        n_nan = int(len(self.dataset) * len(self.dataset.columns) * percentage / 100)
        np.random.seed(0)
        for _ in range(n_nan):
            row = np.random.randint(0, len(self.dataset))
            col = np.random.randint(0, len(self.dataset.columns))
            if self.dataset.columns[col] != self.target_column:
                self.dataset.iloc[row, col] = np.nan
        self.show_data_in_table()

    def impute_nans(self):
        # pushButton_3 için tıklama işlevi
        method = self.comboBox.currentText()

        if method == "Forward Fill":
            self.dataset.fillna(method='ffill', inplace=True)
        elif method == "Backward Fill":
            self.dataset.fillna(method='bfill', inplace=True)
        elif method == "Mean Fill":
            imputer = SimpleImputer(strategy='mean')
            self.dataset.iloc[:, :] = imputer.fit_transform(self.dataset)
        elif method == "Remove Rows":
            self.dataset.dropna(inplace=True)
        elif method == "Fill with 0":
            self.dataset.fillna(0, inplace=True)

        self.show_data_in_table()
        self.evaluate_model_metrics(label=self.label)  # pushButton_3'ün sonuçlarını yazdır

    def normalize_data(self):
        # pushButton_4 için tıklama işlevi
        method = self.comboBox_2.currentText()
        scaler = MinMaxScaler() if method == "Min-Max Scaling" else StandardScaler()
        features = self.dataset.drop(columns=[self.target_column])
        scaled_features = scaler.fit_transform(features)
        self.dataset.loc[:, features.columns] = scaled_features
        self.show_data_in_table()
        self.evaluate_model_metrics(label=self.label_2)  # pushButton_4'ün sonuçlarını yazdır

    def classify_data(self):
        # pushButton_5 için tıklama işlevi
        method = self.comboBox_3.currentText()
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if method == "KNN":
            model = KNeighborsClassifier()
        elif method == "Random Forest":
            model = RandomForestClassifier()
        elif method == "Decision Tree":
            model = DecisionTreeClassifier()

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        
        cm = confusion_matrix(y_test, y_pred)
        cm_text = f"Matrix: [{cm[0, 0]}, {cm[0, 1]} | {cm[1, 0]}, {cm[1, 1]}]"
       
        self.label_3.setText(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Specificity: {specificity:.2f}"
            f"{cm_text}"
        )

    def evaluate_model_metrics(self, label):
        # Ortak metrik hesaplama fonksiyonu
        X = self.dataset.drop(columns=[self.target_column])
        y = self.dataset[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model = RandomForestClassifier()  # Varsayılan model olarak Random Forest
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)

# Karışıklık Matrisi
        cm = confusion_matrix(y_test, y_pred)
        cm_text = f"Matrix: [{cm[0, 0]}, {cm[0, 1]} | {cm[1, 0]}, {cm[1, 1]}]"

        # Metin olarak sonuçlar
        label.setText(
            f"Accuracy: {accuracy:.2f}, "
            f"Precision: {precision:.2f}, "
            f"Recall: {recall:.2f}, "
            f"F1-Score: {f1:.2f}, "
            f"Specificity: {specificity:.2f} | \n"
            f"{cm_text}"
    
        )

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = Ui_Form()  # Ana pencereyi oluştur
    window.show()  # Pencereyi göster
    sys.exit(app.exec_())  # Uygulama çalıştır