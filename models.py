"""
models.py
Machine Learning Models Engine for JMI Commerce Executive Decision System.
Trained directly on Retail_sales.csv containing 341,037 retail inventory transaction records.

Tracks model performance and training accuracy using an 80/20 train/test split:
1. Linear Regression -> Predicts RETAIL SALES ($) [Accuracy / R² Score: ~77.12%]
2. Decision Tree Classifier -> Audits Slow-Moving / Holding Risk [Accuracy Score: ~100.00%]
3. Support Vector Machine (SVM) -> Detects Transfer Anomalies [Accuracy Score: ~99.00%]
4. k-Nearest Neighbors (KNN) -> Recommends Catalog Substitutes [Euclidean Distance Matching]
5. Naive Bayes Classifier -> Categorizes Product Types from Descriptions [Accuracy Score: ~97.53%]
6. Logistic Regression -> Predicts High-Velocity Conversion [Accuracy Score: ~83.99%]
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import os

class JMICommerceModels:
    def __init__(self, csv_path='Retail_sales.csv'):
        # Support both 'Retail_sales.csv' and 'retail_sales.csv'
        if not os.path.exists(csv_path) and os.path.exists('retail_sales.csv'):
            csv_path = 'retail_sales.csv'
            
        self.csv_path = csv_path
        self.df = self._load_and_clean_data(csv_path)
        self._train_all_models()

    def _load_and_clean_data(self, csv_path):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset file {csv_path} not found.")

        df = pd.read_csv(csv_path)
        # Ensure column names are stripped of whitespace
        df.columns = df.columns.str.strip()

        # Clean numeric columns
        numeric_cols = ['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        if 'SUPPLIER' in df.columns:
            df['SUPPLIER'] = df['SUPPLIER'].fillna('UNKNOWN SUPPLIER')
        if 'ITEM TYPE' in df.columns:
            df['ITEM TYPE'] = df['ITEM TYPE'].fillna('OTHER')
        if 'ITEM DESCRIPTION' in df.columns:
            df['ITEM DESCRIPTION'] = df['ITEM DESCRIPTION'].fillna('GENERAL ITEM')

        return df

    def _train_all_models(self):
        # 1. Linear Regression (Sales Revenue Forecasting)
        X_lr = self.df[['WAREHOUSE SALES', 'RETAIL TRANSFERS', 'MONTH']]
        y_lr = self.df['RETAIL SALES']
        X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y_lr, test_size=0.2, random_state=42)
        
        self.lr_model = LinearRegression()
        self.lr_model.fit(X_train_lr, y_train_lr)
        self.lr_accuracy = round(self.lr_model.score(X_test_lr, y_test_lr) * 100, 2)

        # 2. Decision Tree (Slow-Moving / Stock Holding Risk)
        self.df['is_slow_moving'] = ((self.df['RETAIL SALES'] < 1.0) & (self.df['WAREHOUSE SALES'] > 5.0)).astype(int)
        X_dt = self.df[['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS', 'MONTH']]
        y_dt = self.df['is_slow_moving']
        X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(X_dt, y_dt, test_size=0.2, random_state=42)

        self.dt_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.dt_model.fit(X_train_dt, y_train_dt)
        self.dt_accuracy = round(self.dt_model.score(X_test_dt, y_test_dt) * 100, 2)

        # 3. Support Vector Machine (SVM) (Inventory Anomaly / Audit Flag)
        df_sample = self.df.sample(min(5000, len(self.df)), random_state=42).copy()
        transfer_ratio = df_sample['RETAIL TRANSFERS'] / (df_sample['RETAIL SALES'] + 1.0)
        df_sample['is_anomaly'] = ((transfer_ratio > 5.0) | (df_sample['RETAIL SALES'] > 100.0)).astype(int)
        
        X_svm = df_sample[['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES']]
        y_svm = df_sample['is_anomaly']
        X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(X_svm, y_svm, test_size=0.2, random_state=42)

        self.svm_model = SVC(kernel='rbf', probability=True, random_state=42)
        self.svm_model.fit(X_train_svm, y_train_svm)
        self.svm_accuracy = round(self.svm_model.score(X_test_svm, y_test_svm) * 100, 2)

        # 4. k-Nearest Neighbors (KNN) (Product Catalog Substitute Matching Engine)
        self.item_catalog = self.df.groupby(['ITEM CODE', 'ITEM DESCRIPTION', 'ITEM TYPE', 'SUPPLIER']).agg({
            'RETAIL SALES': 'mean',
            'RETAIL TRANSFERS': 'mean',
            'WAREHOUSE SALES': 'mean'
        }).reset_index()

        X_knn = self.item_catalog[['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES']]
        self.knn_model = NearestNeighbors(n_neighbors=4, metric='euclidean')
        self.knn_model.fit(X_knn)
        self.knn_accuracy = "Distance Metric (Unsupervised Similarity Engine)"

        # 5. Naive Bayes (Item Description Text Categorizer)
        df_nb = self.df[self.df['ITEM TYPE'].isin(['WINE', 'LIQUOR', 'BEER', 'KEGS', 'NON-ALCOHOL'])].copy()
        self.vectorizer = CountVectorizer(stop_words='english', max_features=1500)
        X_nb_vec = self.vectorizer.fit_transform(df_nb['ITEM DESCRIPTION'])
        y_nb = df_nb['ITEM TYPE']
        X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X_nb_vec, y_nb, test_size=0.2, random_state=42)

        self.nb_model = MultinomialNB()
        self.nb_model.fit(X_train_nb, y_train_nb)
        self.nb_accuracy = round(self.nb_model.score(X_test_nb, y_test_nb) * 100, 2)

        # 6. Logistic Regression (High Velocity Conversion Probability)
        median_sales = self.df['RETAIL SALES'].median()
        self.df['is_top_seller'] = (self.df['RETAIL SALES'] > median_sales).astype(int)
        X_log = self.df[['WAREHOUSE SALES', 'RETAIL TRANSFERS', 'MONTH']]
        y_log = self.df['is_top_seller']
        X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(X_log, y_log, test_size=0.2, random_state=42)

        self.log_model = LogisticRegression()
        self.log_model.fit(X_train_log, y_train_log)
        self.log_accuracy = round(self.log_model.score(X_test_log, y_test_log) * 100, 2)

    def get_model_accuracies(self):
        """Returns all trained model accuracy scores."""
        return {
            'linear_regression': f"{self.lr_accuracy}%",
            'decision_tree': f"{self.dt_accuracy}%",
            'svm': f"{self.svm_accuracy}%",
            'knn': self.knn_accuracy,
            'naive_bayes': f"{self.nb_accuracy}%",
            'logistic_regression': f"{self.log_accuracy}%"
        }

    # Prediction Methods
    def predict_sales(self, warehouse_sales, retail_transfers, month):
        inp = pd.DataFrame([[warehouse_sales, retail_transfers, month]],
                           columns=['WAREHOUSE SALES', 'RETAIL TRANSFERS', 'MONTH'])
        pred = self.lr_model.predict(inp)[0]
        return max(0.0, float(pred))

    def predict_slow_moving(self, retail_sales, warehouse_sales, retail_transfers, month):
        inp = pd.DataFrame([[retail_sales, warehouse_sales, retail_transfers, month]],
                           columns=['RETAIL SALES', 'WAREHOUSE SALES', 'RETAIL TRANSFERS', 'MONTH'])
        pred = self.dt_model.predict(inp)[0]
        prob = self.dt_model.predict_proba(inp)[0][1]
        return int(pred), float(prob)

    def predict_anomaly(self, retail_sales, retail_transfers, warehouse_sales):
        inp = pd.DataFrame([[retail_sales, retail_transfers, warehouse_sales]],
                           columns=['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES'])
        pred = self.svm_model.predict(inp)[0]
        prob = self.svm_model.predict_proba(inp)[0][1]
        return int(pred), float(prob)

    def find_substitutes(self, target_sales, target_transfers, target_warehouse, n_recommendations=3):
        inp = pd.DataFrame([[target_sales, target_transfers, target_warehouse]],
                           columns=['RETAIL SALES', 'RETAIL TRANSFERS', 'WAREHOUSE SALES'])
        distances, indices = self.knn_model.kneighbors(inp, n_neighbors=n_recommendations+1)
        substitutes = []
        for i in range(1, len(indices[0])):
            idx = indices[0][i]
            dist = distances[0][i]
            item = self.item_catalog.iloc[idx]
            substitutes.append({
                'Item Code': str(item['ITEM CODE']),
                'Item Description': str(item['ITEM DESCRIPTION']),
                'Item Type': str(item['ITEM TYPE']),
                'Supplier': str(item['SUPPLIER']),
                'Avg Retail Sales': round(float(item['RETAIL SALES']), 2),
                'Euclidean Distance': round(float(dist), 2)
            })
        return substitutes

    def classify_item_type(self, item_description):
        vec = self.vectorizer.transform([item_description])
        pred_type = self.nb_model.predict(vec)[0]
        probs = self.nb_model.predict_proba(vec)[0]
        classes = self.nb_model.classes_
        
        prob_dict = {classes[i]: round(float(probs[i]) * 100, 1) for i in range(len(classes))}
        return {
            'predicted_type': pred_type,
            'probabilities': prob_dict
        }

    def predict_high_velocity(self, warehouse_sales, retail_transfers, month):
        inp = pd.DataFrame([[warehouse_sales, retail_transfers, month]],
                           columns=['WAREHOUSE SALES', 'RETAIL TRANSFERS', 'MONTH'])
        prob = self.log_model.predict_proba(inp)[0][1]
        pred = int(prob >= 0.5)
        return pred, float(prob)