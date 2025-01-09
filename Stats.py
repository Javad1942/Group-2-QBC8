import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# خواندن فایل‌ها
hbn_data = pd.read_csv('HBN.csv')
data_dictionary = pd.read_csv('data_dictionary.csv')
parquet_file_path = 'E:/programing/code/quera/first project/series.parquet'
series_data = pd.read_parquet(parquet_file_path)
# ادغام دو دیتاست براساس ستون id
combined_data = pd.merge(hbn_data, series_data, on='id', how='inner')

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

#presenting what we should know about dataset

# توزیع سن
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['Age'], kde=True, bins=20, color='blue')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# توزیع جنسیت
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', data=hbn_data, palette='Set2')
plt.title('Gender Distribution')
plt.xlabel('Sex (0 = Male, 1 = Female)')
plt.ylabel('Count')
plt.grid()
plt.show()

# توزیع وزن
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['Physical-Weight'], kde=True, bins=20, color='orange')
plt.title('Distribution of Weight')
plt.xlabel('Weight (lbs)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# توزیع قد
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['Physical-Height'], kde=True, bins=20, color='green')
plt.title('Distribution of Height')
plt.xlabel('Height (inches)')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# توزیع مصرف اینترنت
plt.figure(figsize=(8, 5))
sns.histplot(hbn_data['PCIAT-PCIAT_Total'], kde=True, bins=20, color='purple')
plt.title('Distribution of Internet Addiction (PCIAT Total Score)')
plt.xlabel('PCIAT Total Score')
plt.ylabel('Frequency')
plt.grid()
plt.show()

# توزیع توده عضلانی بر حسب جنسیت
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='BIA-BIA_SMM', data=hbn_data, palette='Set2')
plt.title('Muscle Mass by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('Skeletal Muscle Mass')
plt.grid()
plt.show()

# توزیع نرخ متابولیسم پایه بر حسب جنسیت
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='BIA-BIA_BMR', data=hbn_data, palette='Set1')
plt.title('Basal Metabolic Rate (BMR) by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('BMR')
plt.grid()
plt.show()


# توزیع میزان عملکرد ذهنی بر حسب جنسیت
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sex', y='CGAS-Score', data=hbn_data, palette='Set3')
plt.title('Mental Performance by Gender')
plt.xlabel('Gender (0 = Male, 1 = Female)')
plt.ylabel('CGAS Score')
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
sns.scatterplot(x='Age', y='PAQ_C-PAQ_C_Total', data=hbn_data, color='teal')
plt.title('Physical Fitness Score by Age')
plt.xlabel('Age')
plt.ylabel('Physical Fitness Score')
plt.grid()
plt.show()

# دسته‌بندی اختلالات خواب
combined_data['Sleep_Disorder'] = combined_data['SDS-SDS_Total_T'].apply(lambda x: 'No Disorder' if x <= 50 else 'Disorder')

# توزیع نور محیطی بر اساس اختلال خواب
plt.figure(figsize=(8, 5))
sns.boxplot(x='Sleep_Disorder', y='light', data=combined_data, palette='coolwarm')
plt.title('Ambient Light Distribution by Sleep Disorder')
plt.xlabel('Sleep Disorder Category')
plt.ylabel('Average Ambient Light')
plt.grid()
plt.show()

correlation_columns = [
    'Age', 'Physical-BMI', 'BIA-BIA_SMM', 'BIA-BIA_BMR',
    'light', 'BIA-BIA_DEE', 'BIA-BIA_FFM', 'step', 'SDS-SDS_Total_T'
]
correlation_data = combined_data[correlation_columns]

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Key Variables')
plt.show()

# توزیع تعداد گام‌ها بر اساس روز هفته
plt.figure(figsize=(8, 5))
sns.boxplot(x='weekday', y='step', data=combined_data, palette='Set2')
plt.title('Step Count by Weekday')
plt.xlabel('Day of the Week')
plt.ylabel('Step Count')
plt.grid()
plt.show()

# میانگین تعداد گام‌ها برای فصل‌ها
avg_steps_per_season = combined_data.groupby('quarter')['step'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='quarter', y='step', data=avg_steps_per_season, palette='viridis')
plt.title('Average Step Count by Season')
plt.xlabel('Season')
plt.ylabel('Average Step Count')
plt.grid()
plt.show()
