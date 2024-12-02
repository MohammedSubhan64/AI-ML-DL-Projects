import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.cluster import KMeans
from scipy import stats

# Add a title to the app
st.title("Streamlit Data Visualization and Cleaning App")

# Sidebar for file upload
st.sidebar.header("Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Initialize an empty DataFrame
data = None
df = None  # Ensure df is initialized to None initially

# Check if file is uploaded
if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
        df = data.copy()  # Make sure df is initialized from the uploaded data
    except Exception as e:
        st.sidebar.error(f"Error reading the file: {e}")
else:
    st.sidebar.info("Please upload a CSV file to proceed.")

# Display the uploaded data
if data is not None:
    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    # Data Cleaning Section
    if st.sidebar.button("Clean Data"):
        # Fill missing values for numeric columns with the mean
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data[col].fillna(data[col].mean(), inplace=True)
        
        # Fill missing values for categorical columns with the mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)
        
        # Label Encoding for categorical columns
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            data[col] = label_encoder.fit_transform(data[col])

        # Remove Outliers
        remove_outliers = st.sidebar.checkbox("Remove Outliers", value=True)
        if remove_outliers:
            st.sidebar.subheader("Outlier Removal Using Z-Score")
            z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
            data = data[(z_scores < 3).all(axis=1)]  # Keep rows with z-score < 3

        # Update df to the cleaned data
        df = data.copy()

        # Display cleaned data
        st.write("### Cleaned Data:")
        st.dataframe(data.head())

    # Ensure every column can be used for plotting
    def preprocess_for_visualization(df):
        # Label Encoding for categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Ensure all columns are numeric where necessary
        df = df.apply(pd.to_numeric, errors='ignore')
        return df

    # Apply preprocessing for visualization (if df is initialized)
    if df is not None:
        df = preprocess_for_visualization(df)

        # Visualization Section
        st.sidebar.header("Data Visualization Options")

        # Visualization options
        visualization_option = st.sidebar.selectbox("Select Visualization", [
            "None", "Histogram", "Correlation Heatmap", "Boxplot", "Pairplot", "Scatter Plot", "Violin Plot", "Clustering Heatmap"])

        if visualization_option == "Histogram":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_column = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
            st.write(f"Histogram of {selected_column}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.histplot(df[selected_column], kde=True, ax=ax)
            st.pyplot(fig)

        if visualization_option == "Correlation Heatmap":
            st.write("Correlation Heatmap of Features")
            fig, ax = plt.subplots(figsize=(10, 8))
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

        if visualization_option == "Boxplot":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_column = st.sidebar.selectbox("Select Column for Boxplot", numeric_columns)
            st.write(f"Boxplot of {selected_column}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(x=df[selected_column], ax=ax)
            st.pyplot(fig)

        if visualization_option == "Pairplot":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_columns = st.sidebar.multiselect("Select Columns for Pairplot", numeric_columns)
            if len(selected_columns) > 1:
                st.write(f"Pairplot of {', '.join(selected_columns)}")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.pairplot(df[selected_columns])
                st.pyplot(fig)

        if visualization_option == "Scatter Plot":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            x_column = st.sidebar.selectbox("Select X-axis for Scatter Plot", numeric_columns)
            y_column = st.sidebar.selectbox("Select Y-axis for Scatter Plot", numeric_columns)
            st.write(f"Scatter Plot of {x_column} vs {y_column}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(x=df[x_column], y=df[y_column], ax=ax)
            st.pyplot(fig)

        if visualization_option == "Violin Plot":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_column = st.sidebar.selectbox("Select Column for Violin Plot", numeric_columns)
            st.write(f"Violin Plot of {selected_column}")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.violinplot(x=df[selected_column], ax=ax)
            st.pyplot(fig)

        if visualization_option == "Clustering Heatmap":
            st.write("Clustering Heatmap Using KMeans Clustering")
            n_clusters = st.sidebar.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(df.select_dtypes(include=[np.number]))
            df['Cluster'] = cluster_labels
            corr = df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)

# Option to export the cleaned data
if df is not None:
    st.sidebar.download_button(
        label="Download Cleaned Data",
        data=df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )
