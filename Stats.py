import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# خواندن فایل‌ها
hbn_data = pd.read_csv('HBN.csv')
data_dictionary = pd.read_csv('data_dictionary.csv')

# بررسی توزیع سن
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['Age'], kde=True, bins=20, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# بررسی توزیع جنسیت
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=hbn_data, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.grid()
plt.show()

# بررسی توزیع BMI (شاخص توده بدنی)
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['Physical-BMI'], kde=True, bins=20, color='green')
plt.title('Distribution of Physical-BMI')
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# بررسی نمره‌ی کلی PCIAT (استفاده از اینترنت)
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['PCIAT-PCIAT_Total'], kde=True, bins=20, color='purple')
plt.title('Distribution of PCIAT Total Score')
plt.xlabel('PCIAT Total Score')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# نمایش اطلاعات کلی دیتاست
print("Dataset Info:")
print(hbn_data.info())

# نمودار ماتریس مقادیر گمشده
plt.figure(figsize=(10, 6))
msno.matrix(hbn_data)
plt.title("Missing Data Matrix")
plt.show()

# نمودار Heatmap مقادیر گمشده
plt.figure(figsize=(10, 6))
msno.heatmap(hbn_data)
plt.title("Missing Data Heatmap")
plt.show()

# نمودار Barplot برای درصد مقادیر گمشده در هر ستون
plt.figure(figsize=(10, 6))
msno.bar(hbn_data)
plt.title("Missing Data Barplot")
plt.show()
