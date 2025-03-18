import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
from sklearn.cluster import KMeans, DBSCAN
import umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTENC, ADASYN, BorderlineSMOTE
from sklearn.metrics import auc, roc_auc_score, roc_curve, classification_report
import matplotlib.lines as mlines
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer

# Streamlit setup
st.write('Data Housekeeping')
st.write('Upload your data')

# File uploader to accept CSV, TSV, and Excel files
uploaded_file = st.file_uploader("Choose a file", type=["csv", "tsv", "xlsx", "xls"])

# Check if the user uploaded a file
if uploaded_file is not None:
    # For CSV files
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=',')
        st.write("Displaying CSV file data:")
        st.write(df)
        
    # For TSV files
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, sep='\t')
        st.write("Displaying TSV file data:")
        st.write(df)
        
    # For Excel files
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        sheet = st.selectbox("Select a sheet", sheet_names)
        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        st.write(f"Displaying data from sheet: {sheet}")
        st.write(f'### Data shape {df.shape}')
        index_column = st.selectbox("Select a Column to Set as Index:", options=df.columns)
        df = df.set_index(index_column)
        st.write(df)

    # Handle missing values
    if df is not None:
        st.write(f'Missing values check')
        operation = st.radio("Choose an operation for missing values", ["Sum", "Mean"])

        # Calculate missing values based on user's choice
        if operation == "Sum":
            missing_values = pd.DataFrame(df.isna().sum(), columns=["Missing Values"])
        elif operation == "Mean":
            missing_values = pd.DataFrame(df.isna().mean(), columns=["Mean of Missing Values"])

        # Display the result
        st.write(missing_values)

        # Sort the missing values in descending order
        missing_values = missing_values.sort_values(by=missing_values.columns[0], ascending=False)

        # Create the plot
        plot_type = st.radio("Choose a plot type", ["Bar Plot", "Scatter Plot"])
        fig, ax = plt.subplots(figsize=(8, 6))
        x = missing_values.index  # Columns of the dataframe
        y = missing_values.iloc[:, 0]  # Missing values (Sum or Mean)

        if plot_type == "Bar Plot":
            ax.bar(x, y, color='skyblue')
            ax.set_ylabel('Missing Values')
            ax.set_title(f'Sorted Missing Values ({operation}) - Bar Plot')
        elif plot_type == "Scatter Plot":
            ax.scatter(x, y, color='orange', s=100)
            ax.set_ylabel('Missing Values')
            ax.set_title(f'Sorted Missing Values ({operation}) - Scatter Plot')

        ax.set_xlabel('Columns')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        st.write("### Dynamic Threshold Adjustment")
        threshold = st.slider("Set the threshold for missing values (percentage)", 0, 100, 50)

        # Filter dataframe based on the threshold
        df_filtered = df.loc[:, df.isna().mean() < (threshold / 100)]

        # Display the filtered dataframe
        st.write(f"### Data After Dropping Columns with More Than {threshold}% Missing Values")
        st.write(df_filtered)
        st.write(f'### Data shape after cleaning {df_filtered.shape}')

        # Compute missing values (mean) for the plot
        missing_values = pd.DataFrame(df_filtered.isna().mean(), columns=["Missing Percentage"])
        missing_values["Missing Percentage"] *= 100  # Convert to percentage
        missing_values = missing_values.sort_values(by="Missing Percentage", ascending=False)
        st.write(pd.DataFrame(missing_values))

        # Features type
        selected_dtype = st.multiselect(
            "Select Data Types to Include:",
            options=["object", "float", "int"],
            default=["object", "float", "int"]
        )

        df_filtered_by_dtype = df_filtered.select_dtypes(include=selected_dtype)

        # Display the filtered dataframe
        st.write(f"### Data After Selecting Data Types: {selected_dtype}")
        st.write(df_filtered_by_dtype)
        st.write(f"### Data shape after selecting types: {df_filtered_by_dtype.shape}")

        # Allow the user to choose the target type for each column
        columns_to_transform = df_filtered_by_dtype.columns
        target_types = ['int', 'float', 'str']
        st.write("### Choose Data Type for Each Feature")
        transformation_choices = {}

        # Dynamically create a selectbox for each column to allow type selection
        for col in columns_to_transform:
            current_dtype = str(df_filtered_by_dtype[col].dtype)
            default_index = target_types.index('str') if 'object' in current_dtype else (
                target_types.index('int') if 'int' in current_dtype else (
                    target_types.index('float') if 'float' in current_dtype else 0
                )
            )
            
            transformation_choices[col] = st.selectbox(
                f"Select data type for column `{col}`:",
                options=target_types,
                index=default_index
            )

        # Apply transformations
        try:
            transformed_columns = {}  # Store transformed columns temporarily
            for col, dtype in transformation_choices.items():
                transformed_columns[col] = df_filtered_by_dtype[col].astype(dtype)

            df_filtered_by_dtype = pd.DataFrame(transformed_columns)

            # Display the transformed DataFrame
            st.write("### Transformed DataFrame")
            st.write(df_filtered_by_dtype)
            st.write(f"### Updated Data Shape: {df_filtered_by_dtype.shape}")
        except Exception as e:
            st.error(f"Error: Could not complete transformations. Details: {e}")

        # Plot every feature
        st.write("### Data Exploration")
        for i in df_filtered_by_dtype.columns:
            if df_filtered_by_dtype[i].dtype == 'object':
                st.write(f"#### Count Plot for `{i}`")
                fig, ax = plt.subplots()
                sns.countplot(data=df_filtered_by_dtype, x=i, ax=ax, palette='Blues')
                st.pyplot(fig)
            elif df_filtered_by_dtype[i].dtype in ['float64', 'int64']:
                st.write(f"#### Distribution Plot for `{i}`")
                fig, ax = plt.subplots()
                sns.histplot(df_filtered_by_dtype[i], kde=True, ax=ax, color='blue')
                st.pyplot(fig)
            else:
                st.write(f"#### `{i}`: Data type not suitable for plotting")



# Streamlit UI
st.write("### Missing Values Handling")
method = st.radio("Select how to handle missing values:", ["Drop", "Imputation", "Dummies"])

df_filtered_by_dtype = pd.DataFrame({})  # Placeholder, sostituire con il DataFrame reale

if method == "Drop":
    drop_threshold = st.slider(
        "Select the percentage threshold for dropping rows with missing values (0% to 100%):",
        min_value=0, max_value=100, value=100
    )
    
elif method == 'Imputation': 
    object_method = st.selectbox("Select imputation method for object columns:", options=['most_frequent', 'constant'])
    numeric_method = st.selectbox("Select imputation method for numeric columns:", options=['mean', 'median', 'most_frequent', 'constant'])
    
    if numeric_method == 'constant':
        numeric_constant = st.number_input("Enter the constant value for numeric columns:", value=0)
    if object_method == 'constant':
        object_constant = st.text_input("Enter the constant value for object columns:", value="missing")
    
elif method == 'Dummies':
    sep = st.text_input("Specify separator for dummies (default '_'):", value="-")
    drop_original = st.checkbox("Drop original column after creating dummies", value=True)

if st.button("Apply Missing Value Handling"):
    if method == "Drop" and drop_threshold < 100:
        non_missing_threshold = len(df_filtered_by_dtype.columns) * (1 - drop_threshold / 100)
        df_filtered_by_dtype = df_filtered_by_dtype.dropna(thresh=non_missing_threshold)
    
    elif method == "Imputation":
        if 'object' in df_filtered_by_dtype.select_dtypes(include='object').dtypes.values:
            object_imputer = SimpleImputer(strategy=object_method, fill_value=object_constant if object_method == 'constant' else None)
            df_filtered_by_dtype[df_filtered_by_dtype.select_dtypes(include='object').columns] = object_imputer.fit_transform(df_filtered_by_dtype.select_dtypes(include='object'))
        
        if any(dtype in ['float64', 'int64'] for dtype in df_filtered_by_dtype.dtypes):
            numeric_imputer = SimpleImputer(strategy=numeric_method, fill_value=numeric_constant if numeric_method == 'constant' else None)
            df_filtered_by_dtype[df_filtered_by_dtype.select_dtypes(include=['int', 'float']).columns] = numeric_imputer.fit_transform(df_filtered_by_dtype.select_dtypes(include=['int', 'float']))
    
elif method == 'Dummies':

    def handle_missing_values_with_dummies(df, sep=None, drop_original=True):
        df = df.copy()  # Evita modifiche in-place
        columns_with_nans = df.columns[df.isna().any()].tolist()

        for col in columns_with_nans:
            prefix = f"{col}{sep}" if sep else col
            dummies = pd.get_dummies(df[col], prefix=prefix, prefix_sep='' if sep is None else sep, dtype=int)

            col_idx = df.columns.get_loc(col)  # Indice della colonna originale
            df = pd.concat([df.iloc[:, :col_idx + 1], dummies, df.iloc[:, col_idx + 1:]], axis=1)

            if drop_original:
                df.drop(columns=[col], inplace=True)

        return df

    sep = st.text_input("Enter separator for dummies (leave empty for default):", value="")
    sep = None if sep == "" else sep  # Se vuoto, non usare separatori

    if st.button("Apply Missing Value Handling"):
        df_filtered_by_dtype = handle_missing_values_with_dummies(df_filtered_by_dtype, sep)

# Display the updated DataFrame
st.write("### DataFrame After Missing Value Handling")
st.write(df_filtered_by_dtype)
st.write(f'### Remaining missing values: {df_filtered_by_dtype.isna().sum().sum()}')
