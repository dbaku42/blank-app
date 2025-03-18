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
    df = None  # Inizializza df per evitare problemi

    # For CSV files
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=',')
        st.write("Displaying CSV file data:")

    # For TSV files
    elif uploaded_file.name.endswith('.tsv'):
        df = pd.read_csv(uploaded_file, sep='\t')
        st.write("Displaying TSV file data:")

    # For Excel files
    elif uploaded_file.name.endswith(('.xlsx', '.xls')):
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        sheet = st.selectbox("Select a sheet", sheet_names)

        df = pd.read_excel(uploaded_file, sheet_name=sheet)
        st.write(f"Displaying data from sheet: {sheet}")
        st.write(f'### Data shape {df.shape}')

        index_column = st.selectbox("Select a Column to Set as Index:", options=df.columns)

        if "modified_df" not in st.session_state:
            st.session_state.modified_df = df  # Initialize session state

        if st.button("Apply Index"):
            st.session_state.modified_df = df.set_index(index_column)  # Corretto

        df = st.session_state.modified_df  # Aggiorna df per i passaggi successivi

    if df is not None:
        st.write(df)

        # Handle missing values
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

# Select missing value threshold using slider
threshold = st.slider("Select missing value threshold (%)", min_value=0, max_value=100, value=30)

# Filter dataframe based on the threshold
df_filtered = df.loc[:, df.isna().mean() < (threshold / 100)]

# Display the filtered dataframe
st.write(f"### Data After Dropping Columns with More Than {threshold}% Missing Values")
st.write(df_filtered)
st.write(f'### Data shape after cleaning {df_filtered.shape}')

# Compute missing values (mean) for the plot
missing_values = pd.DataFrame(df_filtered.isna().mean() * 100, columns=["Missing Percentage"])
missing_values = missing_values.sort_values(by="Missing Percentage", ascending=False)
st.write(missing_values)

# Features type selection
selected_dtype = st.multiselect(
    "Select Data Types to Include:",
    options=["object", "float", "int"],
    default=["object", "float", "int"]
)

# Filter dataframe by selected data types
df_filtered_by_dtype = df_filtered.select_dtypes(include=selected_dtype)

# Display the filtered dataframe
st.write(f"### Data After Selecting Data Types: {selected_dtype}")
st.write(df_filtered_by_dtype)
st.write(f"### Data shape after selecting types: {df_filtered_by_dtype.shape}")

# Allow the user to choose the target type for each column
columns_to_transform = df_filtered_by_dtype.columns
target_types = ['int', 'float', 'str']
st.write("### Choose Data Type for Each Feature")

# Dictionary for user-selected transformations
transformation_choices = {}

for col in columns_to_transform:
    current_dtype = str(df_filtered_by_dtype[col].dtype)
    default_index = target_types.index('str') if 'object' in current_dtype else (
        target_types.index('int') if 'int' in current_dtype else target_types.index('float')
    )

    transformation_choices[col] = st.selectbox(
        f"Select data type for column `{col}`:",
        options=target_types,
        index=default_index
    )

# Apply transformations safely
try:
    df_transformed = df_filtered_by_dtype.copy()

    # Apply the selected transformations
    for col, dtype in transformation_choices.items():
        df_transformed[col] = df_transformed[col].astype(dtype)

    # Display the transformed DataFrame
    st.write("### Transformed DataFrame")
    st.write(df_transformed)
    st.write(f"### Updated Data Shape: {df_transformed.shape}")

except Exception as e:
    st.error(f"Error: Could not complete transformations. Details: {e}")

# Plot every feature
st.write("### Data Exploration")
for col in df_transformed.columns:
    # For categorical columns (object type)
    if df_transformed[col].dtype == 'object':
        st.write(f"#### Count Plot for `{col}`")
        fig, ax = plt.subplots()
        sns.countplot(data=df_transformed, x=col, ax=ax, palette='Blues')
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # For numerical columns (float or int type)
    elif df_transformed[col].dtype in ['float64', 'int64']:
        st.write(f"#### Distribution Plot for `{col}`")
        fig, ax = plt.subplots()
        sns.histplot(df_transformed[col], kde=True, ax=ax, color='blue')
        st.pyplot(fig)

    # If the column's data type is not suitable for plotting
    else:
        st.write(f"#### `{col}`: Data type not suitable for plotting")

from sklearn.impute import SimpleImputer
import pandas as pd
import streamlit as st

# Ensure there's a session state for storing the modified dataframe
if 'modified_df' not in st.session_state:
    st.session_state.modified_df = df_filtered_by_dtype.copy()

# Missing Values Handling
st.write("### Missing Values Handling")
method = st.radio("Select how to handle missing values:", ["Drop", "Imputation", "Dummies"])

# Drop method
if method == "Drop":
    drop_threshold = st.slider(
        "Select the percentage threshold for dropping rows with missing values (0% to 100%):",
        min_value=0, max_value=100, value=100
    )
    
    if st.button("Apply Drop Missing Values"):
        non_missing_threshold = len(st.session_state.modified_df.columns) * (1 - drop_threshold / 100)
        st.session_state.modified_df = st.session_state.modified_df.dropna(thresh=non_missing_threshold)

# Imputation method
elif method == 'Imputation':
    object_method = st.selectbox("Select imputation method for object columns:", options=['most_frequent', 'constant'])
    numeric_method = st.selectbox("Select imputation method for numeric columns:", options=['mean', 'median', 'most_frequent', 'constant'])
    
    # Handle constant imputation
    if numeric_method == 'constant':
        numeric_constant = st.number_input("Enter the constant value for numeric columns:", value=0)
    if object_method == 'constant':
        object_constant = st.text_input("Enter the constant value for object columns:", value="missing")
    
    if st.button("Apply Imputation"):
        df = st.session_state.modified_df.copy()

        # Impute object columns
        if 'object' in df.select_dtypes(include='object').dtypes.values:
            object_imputer = SimpleImputer(strategy=object_method, fill_value=object_constant if object_method == 'constant' else None)
            df[df.select_dtypes(include='object').columns] = object_imputer.fit_transform(df.select_dtypes(include='object'))

        # Impute numeric columns
        if any(dtype in ['float64', 'int64'] for dtype in df.dtypes):
            numeric_imputer = SimpleImputer(strategy=numeric_method, fill_value=numeric_constant if numeric_method == 'constant' else None)
            df[df.select_dtypes(include=['int', 'float']).columns] = numeric_imputer.fit_transform(df.select_dtypes(include=['int', 'float']))
        
        st.session_state.modified_df = df

# Dummies method
elif method == 'Dummies':
    sep = st.text_input("Specify separator for dummies (leave empty for none):", value="")
    drop_original = st.checkbox("Drop original column after creating dummies", value=True)
    
    # Function to handle missing values and create dummies
    def handle_missing_values_with_dummies(df, sep=None, drop_original=True):
        columns_with_nans = df.columns[df.isna().any()].tolist()

        for col in columns_with_nans:
            prefix = f"{col}{sep}" if sep else col
            dummies = pd.get_dummies(df[col], prefix=prefix, prefix_sep='' if not sep else sep, dtype=int)
            
            col_idx = df.columns.get_loc(col)
            df = pd.concat([df.iloc[:, :col_idx + 1], dummies, df.iloc[:, col_idx + 1:]], axis=1)
            
            if drop_original:
                df.drop(columns=[col], inplace=True)

        return df
    
    if st.button("Apply Dummies Encoding"):
        st.session_state.modified_df = handle_missing_values_with_dummies(st.session_state.modified_df, sep, drop_original)

# Display the updated DataFrame
st.write("### DataFrame After Missing Value Handling")
st.write(st.session_state.modified_df)

# Display remaining missing values
remaining_missing = st.session_state.modified_df.isna().sum().sum()
st.write(f'### Remaining missing values: {remaining_missing}')
