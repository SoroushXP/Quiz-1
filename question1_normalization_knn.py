# سؤال 1: نرمال‌سازی و اثر آن روی KNN

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import arabic_reshaper
from bidi.algorithm import get_display

# تنظیم فونت برای نمایش صحیح فارسی
mpl.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False

def persian_text(text):
    """Convert Persian text for proper RTL display in matplotlib"""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score


print("=== سؤال 1: نرمال‌سازی و KNN ===")

df_diamonds = sns.load_dataset("diamonds")
numerical_features = ['carat', 'depth', 'table', 'price', 'x', 'y', 'z']

# رسم هیستوگرام و باکس‌پلات
for col in numerical_features:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df_diamonds[col], kde=True, color='skyblue')
    plt.title(persian_text(f'هیستوگرام {col}'))
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df_diamonds[col], color='lightgreen')
    plt.title(persian_text(f'باکس‌پلات {col}'))
    plt.tight_layout()
    plt.show()

# آماده‌سازی داده برای KNN (پیش‌بینی cut)
X = df_diamonds[numerical_features]
y = df_diamonds['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2,
                             'Premium': 3, 'Ideal': 4})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# تابع ارزیابی KNN
def evaluate_knn(X_tr, X_te, name):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_tr, y_train)
    y_pred = knn.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    print(f"{name} - Accuracy: {acc:.4f}")
    return acc

# بدون نرمال‌سازی
acc_raw = evaluate_knn(X_train, X_test, "بدون نرمال‌سازی")

# Min-Max Scaling
mm_scaler = MinMaxScaler()
X_train_mm = mm_scaler.fit_transform(X_train)
X_test_mm = mm_scaler.transform(X_test)
acc_mm = evaluate_knn(X_train_mm, X_test_mm, "Min-Max Scaling")

# Z-score (StandardScaler)
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)
acc_std = evaluate_knn(X_train_std, X_test_std, "Z-score Scaling")
