# سؤال 4: مدیریت داده‌های نامتوازن

import pandas as pd
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler


print("=== سؤال 4: داده‌های نامتوازن ===")

data = load_breast_cancer()
df_bc = pd.DataFrame(data.data, columns=data.feature_names)
df_bc['target'] = data.target  # 0: malignant, 1: benign

# توزیع اولیه
print("توزیع اولیه کلاس‌ها:")
print(df_bc['target'].value_counts(normalize=True))

# نمودار پای
plt.figure(figsize=(6, 6))
plt.pie(df_bc['target'].value_counts(), labels=['Benign (1)', 'Malignant (0)'],
        autopct='%1.1f%%', colors=['lightblue', 'salmon'])
plt.title(persian_text('توزیع کلاس‌ها در دیتاست') + ' breast cancer')
plt.show()

# ایجاد عدم تعادل بیشتر (80% benign - 20% malignant)
# نمونه‌برداری دستی برای ایجاد نسبت 80/20
df_benign = df_bc[df_bc['target'] == 1]
df_malignant = df_bc[df_bc['target'] == 0]

# کاهش تعداد malignant به 25% از benign (برای رسیدن به 80/20)
n_malignant_new = int(len(df_benign) * 0.25)
df_malignant_sampled = df_malignant.sample(n=n_malignant_new, random_state=42)

df_imbalanced = pd.concat([df_benign, df_malignant_sampled])
X_res = df_imbalanced.drop('target', axis=1)
y_res = df_imbalanced['target']

print("\nتوزیع پس از ایجاد عدم تعادل:")
print(pd.Series(y_res).value_counts(normalize=True))

# آموزش مدل روی داده نامتوازن
X_tr, X_te, y_tr, y_te = train_test_split(X_res, y_res, test_size=0.2,
                                          random_state=42, stratify=y_res)

# استانداردسازی برای همگرایی بهتر
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_te_scaled = scaler.transform(X_te)

lr_imbalanced = LogisticRegression(max_iter=1000)
lr_imbalanced.fit(X_tr_scaled, y_tr)
y_pred = lr_imbalanced.predict(X_te_scaled)

print("\nنتایج مدل Logistic Regression روی داده نامتوازن:")
print(f"Accuracy:  {accuracy_score(y_te, y_pred):.4f}")
print(f"Precision: {precision_score(y_te, y_pred):.4f}")
print(f"Recall:    {recall_score(y_te, y_pred):.4f}")
print(f"F1-score:  {f1_score(y_te, y_pred):.4f}")

print("\nتوضیح: Accuracy معیار مناسبی نیست زیرا مدل می‌تواند فقط کلاس اکثریت را پیش‌بینی کند و همچنان دقت بالایی داشته باشد.")
print("در مسائل پزشکی، Recall (تشخیص درست موارد مثبت) اهمیت بیشتری دارد.")
