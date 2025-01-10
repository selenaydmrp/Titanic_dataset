from PyQt5 import QtWidgets
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE, RandomOverSampler

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Arayüz elemanlarını oluştur
        self.setWindowTitle("Makine \u00d6\u011frenimi Projesi")
        self.resize(1000, 800)  # Boyut artırıldı

        self.centralWidget = QtWidgets.QWidget()
        self.setCentralWidget(self.centralWidget)

        # Veri seti için bir TableWidget
        self.tableWidget = QtWidgets.QTableWidget(self.centralWidget)
        self.tableWidget.setGeometry(10, 10, 600, 600)  # Daha geniş alan verildi

        # ComboBox for imbalance method selection
        self.comboBox = QtWidgets.QComboBox(self.centralWidget)
        self.comboBox.setGeometry(620, 50, 150, 30)
        self.comboBox.addItem("SMOTE")
        self.comboBox.addItem("ROS")

        # SpinBox
        self.spinBox = QtWidgets.QSpinBox(self.centralWidget)
        self.spinBox.setGeometry(620, 100, 150, 30)
        self.spinBox.setMinimum(1)
        self.spinBox.setValue(5)

        # Butonlar
        self.loadButton = QtWidgets.QPushButton("Veri Seti Y\u00fckle", self.centralWidget)
        self.loadButton.setGeometry(620, 10, 150, 30)
        self.loadButton.clicked.connect(self.load_dataset)

        self.imbalanceButton = QtWidgets.QPushButton("Dengesizlik Ekle", self.centralWidget)
        self.imbalanceButton.setGeometry(620, 150, 150, 30)
        self.imbalanceButton.clicked.connect(self.add_imbalance)

        self.trainButton = QtWidgets.QPushButton("Model E\u011fitimi", self.centralWidget)
        self.trainButton.setGeometry(620, 200, 150, 30)
        self.trainButton.clicked.connect(self.train_model)

        # SMOTE/ROS Uygulama Butonu
        self.pushButton_2 = QtWidgets.QPushButton("SMOTE/ROS Uygula", self.centralWidget)
        self.pushButton_2.setGeometry(620, 250, 150, 30)
        self.pushButton_2.clicked.connect(self.apply_imbalance_method)

        # Confusion matrix button
        self.pushButton_3 = QtWidgets.QPushButton("Karmaşıklık Matrisi Göster", self.centralWidget)
        self.pushButton_3.setGeometry(620, 300, 150, 30)
        self.pushButton_3.clicked.connect(self.show_confusion_matrix)

        # Sonuçları göstermek için bir Label
        self.resultLabel = QtWidgets.QLabel(self.centralWidget)
        self.resultLabel.setGeometry(10, 620, 960, 180)  # Daha geniş ve uzun alan
        self.resultLabel.setWordWrap(True)
        self.resultLabel.setStyleSheet("font-size: 14px;")

        # Label for the model results
        self.label_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_2.setGeometry(620, 350, 350, 200)
        self.label_2.setWordWrap(True)
        self.label_2.setStyleSheet("font-size: 14px;")

        # Label for the confusion matrix results
        self.label_5 = QtWidgets.QLabel(self.centralWidget)
        self.label_5.setGeometry(620, 450, 350, 100)
        self.label_5.setWordWrap(True)
        self.label_5.setStyleSheet("font-size: 14px;")

    def load_dataset(self):
        file_path = r"C:\\Users\\Acer\\Desktop\\Makine Projesi_\\verisetim_guncel.csv"
        try:
            self.df = pd.read_csv(file_path)
            self.load_table(self.df)
            QtWidgets.QMessageBox.information(None, "Ba\u015far\u0131l\u0131", "Veri seti ba\u015far\u0131yla y\u00fcklendi.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Hata", f"Veriseti y\u00fcklenirken bir hata olu\u015ftu:\n{e}")

    def add_imbalance(self):
        try:
            imbalance_count = self.spinBox.value()

            if not hasattr(self, 'df'):
                QtWidgets.QMessageBox.warning(None, "Uyar\u0131", "L\u00fctfen \u00f6nce bir veri seti y\u00fckleyin.")
                return

            target_column = "Survived"

            if target_column not in self.df.columns:
                QtWidgets.QMessageBox.critical(None, "Hata", f"Hedef s\u00fctun '{target_column}' veri setinde bulunamad\u0131.")
                return

            minority_class = self.df[target_column].value_counts().idxmin()
            minority_samples = self.df[self.df[target_column] == minority_class]

            duplicated_samples = minority_samples.sample(n=imbalance_count, replace=True)
            self.df = pd.concat([self.df, duplicated_samples], ignore_index=True)

            self.load_table(self.df)
            QtWidgets.QMessageBox.information(None, "Ba\u015far\u0131l\u0131", f"Veri setine {imbalance_count} dengesizlik eklendi.")
        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Hata", f"Dengesizlik eklenirken bir hata olu\u015ftu:\n{e}")

    def apply_imbalance_method(self):
        if not hasattr(self, 'df'):
            QtWidgets.QMessageBox.warning(None, "Uyarı", "Lütfen önce bir veri seti yükleyin.")
            return

        method = self.comboBox.currentText()
        target_column = "Survived"
        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        try:
            if method == "SMOTE":
                smote = SMOTE()
                X_res, y_res = smote.fit_resample(X, y)
                self.df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=[target_column])], axis=1)
            
            elif method == "ROS":
                ros = RandomOverSampler()
                X_res, y_res = ros.fit_resample(X, y)
                self.df = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=[target_column])], axis=1)

            self.load_table(self.df)
            QtWidgets.QMessageBox.information(None, "Başarı", f"{method} başarıyla uygulandı.")

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Hata", f"SMOTE/ROS uygularken bir hata oluştu:\n{e}")

    def train_model(self):
        if not hasattr(self, 'df'):
            QtWidgets.QMessageBox.warning(None, "Uyar\u0131", "L\u00fctfen \u00f6nce bir veri seti y\u00fckleyin.")
            return

        target_column = "Survived"
        if target_column not in self.df.columns:
            QtWidgets.QMessageBox.critical(None, "Hata", f"Hedef s\u00fctun '{target_column}' veri setinde bulunamad\u0131.")
            return

        X = self.df.drop(columns=[target_column])
        y = self.df[target_column]

        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model listesi
            models = {
                'Logistic Regression': LogisticRegression(max_iter=1000),
                'Random Forest': RandomForestClassifier(),
                'Support Vector Machine': SVC()
            }

            # Model sonuçlarını saklamak için bir dict
            results = {}

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                results[model_name] = {
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1
                }

            # En iyi modeli bulalım
            best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
            best_model = results[best_model_name]

            # En iyi modelin sonuçlarını ekranda göster
            result_text = f"En İyi Model: {best_model_name}\n" + \
                          f"Doğruluk: {best_model['Accuracy']:.2f}\n" + \
                          f"Duyarlılık: {best_model['Recall']:.2f}\n" + \
                          f"Özgüllük: {1 - best_model['Recall']:.2f}\n" + \
                          f"F1 Skoru: {best_model['F1 Score']:.2f}"

            self.label_2.setText(result_text)

            # Confusion matrix (Karmaşıklık Matrisi)
            cm = confusion_matrix(y_test, y_pred)

            # Görselleştirme
            plt.figure(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
            plt.ylabel('Gerçek Değer')
            plt.xlabel('Tahmin Değeri')
            plt.title(f'{best_model_name} - Karmaşıklık Matrisi')
            plt.show()

        except Exception as e:
            QtWidgets.QMessageBox.critical(None, "Hata", f"Model eğitimi sırasında bir hata oluştu:\n{e}")

    def load_table(self, df):
        self.tableWidget.setRowCount(df.shape[0])
        self.tableWidget.setColumnCount(df.shape[1])
        self.tableWidget.setHorizontalHeaderLabels(df.columns)
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.tableWidget.setItem(i, j, QtWidgets.QTableWidgetItem(str(df.iat[i, j])))

    def show_confusion_matrix(self):
        if not hasattr(self, 'df'):
            QtWidgets.QMessageBox.warning(None, "Uyarı", "Lütfen önce bir model eğitin.")
            return

        plt.show()




if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
