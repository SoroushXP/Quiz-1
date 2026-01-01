# سؤال 3: استخراج و انتخاب ویژگی

import pandas as pd
import numpy as np
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


print("=== سؤال 3: PCA و Feature Selection ===")

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df_wine = pd.read_csv(url, sep=';')

X_wine = df_wine.drop('quality', axis=1)
y_wine = (df_wine['quality'] > 5).astype(int)  # تبدیل به مسئله باینری

# استانداردسازی
scaler_wine = StandardScaler()
X_scaled = scaler_wine.fit_transform(X_wine)

# PCA - نگه‌داشتن 95% واریانس
pca = PCA()
X_pca_full = pca.fit_transform(X_scaled)
cum_var = np.cumsum(pca.explained_variance_ratio_)
n_comp_95 = np.searchsorted(cum_var, 0.95) + 1
print(f"تعداد مؤلفه‌های لازم برای 95% واریانس: {n_comp_95}")

X_pca_95 = X_pca_full[:, :n_comp_95]

# Feature Selection با Mutual Information (2 ویژگی برتر)
selector = SelectKBest(mutual_info_classif, k=2)
X_selected = selector.fit_transform(X_scaled, y_wine)
selected_features = X_wine.columns[selector.get_support()].tolist()
print(f"ویژگی‌های انتخاب‌شده: {selected_features}")

# تابع آموزش و ارزیابی Logistic Regression
def train_evaluate(X_data, name):
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_wine,
                                              test_size=0.2, random_state=42, stratify=y_wine)
    start = time()
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_tr, y_tr)
    y_pred = lr.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    elapsed = time() - start
    print(f"{name} - Accuracy: {acc:.4f} | زمان: {elapsed:.4f}s | تعداد ویژگی: {X_data.shape[1]}")
    return acc, elapsed

train_evaluate(X_scaled, "داده کامل (11 ویژگی)")
train_evaluate(X_pca_95, f"PCA ({n_comp_95} مؤلفه)")
train_evaluate(X_selected, "دو ویژگی انتخاب‌شده")
