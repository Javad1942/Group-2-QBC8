import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# عنوان اصلی اپلیکیشن
st.title("تحلیل داده‌ها و آزمون‌های آماری")

# آپلود فایل داده
uploaded_file = st.file_uploader("یک فایل داده آپلود کنید (CSV یا Parquet):", type=["csv", "parquet"])

if uploaded_file is not None:
    # خواندن داده بر اساس فرمت فایل
    file_extension = uploaded_file.name.split(".")[-1]
    if file_extension == "csv":
        df = pd.read_csv(uploaded_file)
    elif file_extension == "parquet":
        df = pd.read_parquet(uploaded_file)

    # نمایش اطلاعات کلی دیتاست
    st.subheader("اطلاعات کلی داده‌ها")
    st.write(df.head())

    # نمایش خلاصه اطلاعات داده‌ها
    st.write("اطلاعات دیتاست:")
    buffer = st.empty()
    buffer.text(df.info())

    # بخش 1: نمایش داده‌های گمشده
    st.subheader("بصری‌سازی داده‌های گمشده")
    missing_plot_option = st.selectbox("یک روش نمایش انتخاب کنید:", ["Matrix", "Heatmap", "Barplot"])

    if missing_plot_option == "Matrix":
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.matrix(df, ax=ax)
        st.pyplot(fig)
    elif missing_plot_option == "Heatmap":
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.heatmap(df, ax=ax)
        st.pyplot(fig)
    elif missing_plot_option == "Barplot":
        fig, ax = plt.subplots(figsize=(10, 6))
        msno.bar(df, ax=ax)
        st.pyplot(fig)

    # بخش 2: تحلیل توزیع متغیرها
    st.subheader("تحلیل توزیع متغیرها")
    numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns
    selected_column = st.selectbox("یک متغیر انتخاب کنید:", numeric_columns)

    if selected_column:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df[selected_column].dropna(), kde=True, ax=ax, color="blue")
        ax.set_title(f"توزیع {selected_column}")
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    # بخش 3: تحلیل بر اساس جنسیت
    st.subheader("تحلیل جنسیت")
    gender_column = "Sex"  # فرض می‌کنیم ستون جنسیت این نام را دارد
    if gender_column in df.columns:
        gender_analysis_option = st.selectbox("یک تحلیل انتخاب کنید:", ["Muscle Mass", "Basal Metabolic Rate", "Mental Performance"])

        if gender_analysis_option == "Muscle Mass":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=gender_column, y='BIA-BIA_SMM', data=df, ax=ax, palette='Set2')
            ax.set_title('Muscle Mass by Gender')
            ax.set_xlabel('Gender (0 = Male, 1 = Female)')
            ax.set_ylabel('Skeletal Muscle Mass')
            st.pyplot(fig)

        elif gender_analysis_option == "Basal Metabolic Rate":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=gender_column, y='BIA-BIA_BMR', data=df, ax=ax, palette='Set1')
            ax.set_title('Basal Metabolic Rate (BMR) by Gender')
            ax.set_xlabel('Gender (0 = Male, 1 = Female)')
            ax.set_ylabel('BMR')
            st.pyplot(fig)

        elif gender_analysis_option == "Mental Performance":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x=gender_column, y='CGAS-Score', data=df, ax=ax, palette='Set3')
            ax.set_title('Mental Performance by Gender')
            ax.set_xlabel('Gender (0 = Male, 1 = Female)')
            ax.set_ylabel('CGAS Score')
            st.pyplot(fig)

    # بخش 4: ماتریس همبستگی
    st.subheader("ماتریس همبستگی")
    correlation_columns = st.multiselect("ستون‌های مورد نظر برای همبستگی را انتخاب کنید:", numeric_columns)

    if len(correlation_columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df[correlation_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

    # بخش 5: تحلیل اختلال خواب
    st.subheader("تحلیل اختلال خواب")
    if 'SDS-SDS_Total_T' in df.columns:
        df['Sleep_Disorder'] = df['SDS-SDS_Total_T'].apply(lambda x: 'No Disorder' if x <= 50 else 'Disorder')
        if 'light' in df.columns:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='Sleep_Disorder', y='light', data=df, palette='coolwarm', ax=ax)
            ax.set_title('Ambient Light by Sleep Disorder')
            ax.set_xlabel('Sleep Disorder Category')
            ax.set_ylabel('Average Ambient Light')
            st.pyplot(fig)

    # بخش 6: تحلیل گام‌ها
    st.subheader("تحلیل تعداد گام‌ها")
    if 'weekday' in df.columns and 'step' in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='weekday', y='step', data=df, ax=ax, palette='Set2')
        ax.set_title('Step Count by Weekday')
        ax.set_xlabel('Day of the Week')
        ax.set_ylabel('Step Count')
        st.pyplot(fig)

    if 'quarter' in df.columns and 'step' in df.columns:
        avg_steps_per_season = df.groupby('quarter')['step'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x='quarter', y='step', data=avg_steps_per_season, ax=ax, palette='viridis')
        ax.set_title('Average Step Count by Season')
        ax.set_xlabel('Season')
        ax.set_ylabel('Average Step Count')
        st.pyplot(fig)
