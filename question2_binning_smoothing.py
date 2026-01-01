# سؤال 2: باینینگ و هموارسازی

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display

# تنظیم فونت برای نمایش صحیح فارسی
mpl.rcParams['font.family'] = 'Tahoma'
plt.rcParams['axes.unicode_minus'] = False

def persian_text(text):
    """Convert Persian text for proper RTL display in matplotlib"""
    reshaped = arabic_reshaper.reshape(text)
    return get_display(reshaped)


print("=== سؤال 2: باینینگ و هموارسازی ===")

df_titanic = sns.load_dataset("titanic")

# Equal-Width Binning (4 bin)
bins_width = [0, 20, 40, 60, 80]
labels_width = ['0-20', '20-40', '40-60', '60-80']
df_titanic['age_eq_width'] = pd.cut(df_titanic['age'],
                                   bins=bins_width, labels=labels_width, include_lowest=True)
print("Equal-Width Binning:")
print(df_titanic['age_eq_width'].value_counts().sort_index())

# Equal-Frequency Binning (4 bin)
df_titanic['age_eq_freq'] = pd.qcut(df_titanic['age'], q=4,
                                   labels=['Q1', 'Q2', 'Q3', 'Q4'])
print("\nEqual-Frequency Binning:")
print(df_titanic['age_eq_freq'].value_counts().sort_index())

# تقسیم به 5 bin برای هموارسازی
df_titanic['age_bin'] = pd.cut(df_titanic['age'], bins=5)

# Smoothing by bin mean
df_titanic['age_smooth_mean'] = df_titanic.groupby('age_bin')['age'].transform('mean')

# Smoothing by bin median
df_titanic['age_smooth_median'] = df_titanic.groupby('age_bin')['age'].transform('median')

# Smoothing by bin boundaries
def smooth_by_boundaries(row):
    if pd.isna(row['age']):
        return np.nan
    interval = row['age_bin']
    left = interval.left
    right = interval.right
    return left if (row['age'] - left) < (right - row['age']) else right

df_titanic['age_smooth_bound'] = df_titanic.apply(smooth_by_boundaries, axis=1)

# نمایش هیستوگرام‌ها (اختیاری - برای مقایسه)
cols_smooth = ['age', 'age_smooth_mean', 'age_smooth_median', 'age_smooth_bound']
titles = [persian_text('اصلی'), 'Mean', 'Median', 'Boundaries']
plt.figure(figsize=(12, 8))
for i, col in enumerate(cols_smooth):
    plt.subplot(2, 2, i+1)
    df_titanic[col].dropna().hist(bins=30, alpha=0.7)
    plt.title(titles[i])
plt.suptitle(persian_text('مقایسه هیستوگرام قبل و بعد از هموارسازی'))
plt.tight_layout()
plt.show()
