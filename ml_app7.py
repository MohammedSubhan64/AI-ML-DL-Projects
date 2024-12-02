import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
import joblib
from sklearn.impute import SimpleImputer

# Streamlit App
st.title("Machine Learning Algorithm Performance")

# Sidebar for navigation
page = st.sidebar.radio("Select a page", ["Model Evaluation", "Data Visualization"])

# Model Evaluation Page
if page == "Model Evaluation":
    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Preprocess the data
        st.write("Handling missing values...")

        # Identify categorical and numerical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

        # Impute missing values
        imputer_cat = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

        imputer_num = SimpleImputer(strategy='mean')
        df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

        # Remove duplicates
        df = df.drop_duplicates()

        st.write(f"Dataset after handling missing values and removing duplicates:")
        st.dataframe(df.head())

        # Display available columns
        st.write(f"Available Columns in Dataset: {df.columns.tolist()}")

        # Sidebar for feature and target selection
        st.sidebar.header("Select Features and Target Column")
        feature_columns = st.sidebar.multiselect("Select Feature Columns", df.columns.tolist(), default=df.columns.tolist()[:-1])
        target_column = st.sidebar.selectbox("Select Target Column", df.columns.tolist())

        if target_column not in df.columns or any(col not in df.columns for col in feature_columns):
            st.error("One or more of the selected columns are not present in the dataset!")
        else:
            # Label Encoding for categorical columns
            label_encoder = LabelEncoder()
            if df[target_column].dtype == 'object':
                df[target_column] = label_encoder.fit_transform(df[target_column])

            categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                df[col] = label_encoder.fit_transform(df[col])

            # Split data
            X = df[feature_columns]
            y = df[target_column]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            # Scaling options
            scaling_option = st.sidebar.selectbox("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
            if scaling_option == "StandardScaler":
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            elif scaling_option == "MinMaxScaler":
                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            # SMOTE option for class imbalance
            if st.sidebar.checkbox("Apply SMOTE for Class Imbalance", False):
                smote = SMOTE(sampling_strategy='auto', random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                st.write("SMOTE applied!")

            # Model selection from sidebar
            st.sidebar.header("Select Algorithm for Evaluation")
            algorithm = st.sidebar.selectbox("Select Algorithm", [
                "Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "Naive Bayes"])

            # Define models
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(probability=True),
                "KNN": KNeighborsClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Naive Bayes": GaussianNB()
            }

            # Track the best model
            best_model = None
            best_accuracy = 0
            best_model_name = ""

            # K-Fold Cross Validation and Performance
            if st.sidebar.checkbox("Perform K-Fold Cross Validation", False):
                st.write("Performing K-Fold Cross Validation...")

                kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_results = cross_val_score(models[algorithm], X, y, cv=kfold, scoring='accuracy')
                st.write(f"Accuracy: {cv_results.mean():.2f} (+/- {cv_results.std():.2f})")

            # Bagging and Boosting
            st.sidebar.header("Bagging & Boosting")
            try_bagging = st.sidebar.checkbox("Try Bagging Model", False)
            try_boosting = st.sidebar.checkbox("Try Boosting Model", False)

            if try_bagging:
                st.write("Performing Bagging...")
                bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
                bagging_model.fit(X_train, y_train)
                bagging_pred = bagging_model.predict(X_test)
                bagging_accuracy = accuracy_score(y_test, bagging_pred)
                st.write(f"Bagging Accuracy: {bagging_accuracy * 100:.2f}%")

                # Check if this is the best model
                if bagging_accuracy > best_accuracy:
                    best_accuracy = bagging_accuracy
                    best_model = bagging_model
                    best_model_name = "Bagging Model"

                # Save Model Button
                if st.sidebar.button("Save Bagging Model"):
                    joblib.dump(bagging_model, "bagging_model.pkl")
                    st.success("Bagging Model Saved!")

            if try_boosting:
                st.write("Performing Boosting...")
                boosting_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
                boosting_model.fit(X_train, y_train)
                boosting_pred = boosting_model.predict(X_test)
                boosting_accuracy = accuracy_score(y_test, boosting_pred)
                st.write(f"Boosting Accuracy: {boosting_accuracy * 100:.2f}%")

                # Check if this is the best model
                if boosting_accuracy > best_accuracy:
                    best_accuracy = boosting_accuracy
                    best_model = boosting_model
                    best_model_name = "Boosting Model"

                # Save Model Button
                if st.sidebar.button("Save Boosting Model"):
                    joblib.dump(boosting_model, "boosting_model.pkl")
                    st.success("Boosting Model Saved!")

            # Evaluate Model
            if algorithm in models:
                model = models[algorithm]
                model.fit(X_train, y_train)

                # Predictions and Evaluation
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)

                # Display accuracy
                st.subheader(f"Accuracy of {algorithm}: {accuracy * 100:.2f}%")

                # Check if this is the best model
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
                    best_model_name = algorithm

                # Confusion Matrix
                cm = confusion_matrix(y_test, y_pred)
                st.subheader("Confusion Matrix:")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel('Predicted Labels')
                ax.set_ylabel('True Labels')
                ax.set_title('Confusion Matrix')
                st.pyplot(fig)

                # Classification Report
                report = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.subheader("Classification Report:")
                st.dataframe(report_df)

            # Save Best Model by Name
            if best_model is not None:
                st.subheader(f"Saving Best Model: {best_model_name} with Accuracy: {best_accuracy * 100:.2f}%")
                save_best_model = st.sidebar.text_input("Enter the name to save the best model (e.g., 'best_model.pkl')", "best_model.pkl")
                if st.sidebar.button("Save Best Model"):
                    joblib.dump(best_model, save_best_model)
                    st.success(f"Best Model Saved as {save_best_model}")

# Data Visualization Page
elif page == "Data Visualization":
    st.sidebar.header("Data Visualization Options")
    uploaded_file = st.sidebar.file_uploader("Upload CSV for Visualization", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(df.head())

        # Preprocess data (if necessary)
        # Handling missing values, duplicate values, and encoding as needed before visualizations
        df = df.drop_duplicates()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Encode categorical columns for visualization
        label_encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = label_encoder.fit_transform(df[col])

        # Visualization options
        visualization_option = st.sidebar.selectbox("Select Visualization", [
            "None", "Histogram", "Correlation Heatmap", "Boxplot", "Pairplot"])

        if visualization_option == "Histogram":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_column = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
            st.write(f"Histogram of {selected_column}")
            fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
            sns.histplot(df[selected_column], kde=True, ax=ax)  # Pass ax to sns
            st.pyplot(fig)  # Now passing the figure object

        if visualization_option == "Correlation Heatmap":
            st.write("Correlation Heatmap of Features")
            fig, ax = plt.subplots(figsize=(10, 8))  # Creating a figure object
            corr = df.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)  # Pass ax to sns
            st.pyplot(fig)  # Now passing the figure object

        if visualization_option == "Boxplot":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_column = st.sidebar.selectbox("Select Column for Boxplot", numeric_columns)
            st.write(f"Boxplot of {selected_column}")
            fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
            sns.boxplot(x=df[selected_column], ax=ax)  # Pass ax to sns
            st.pyplot(fig)  # Now passing the figure object

        if visualization_option == "Pairplot":
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            selected_columns = st.sidebar.multiselect("Select Columns for Pairplot", numeric_columns)
            if len(selected_columns) > 1:
                st.write(f"Pairplot of {', '.join(selected_columns)}")
                fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
                sns.pairplot(df[selected_columns])  # Pairplot doesn't need ax, it creates its own
                st.pyplot(fig)  # Now passing the figure object



# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from imblearn.over_sampling import SMOTE
# import joblib
# from sklearn.impute import SimpleImputer

# # Streamlit App
# st.title("Machine Learning Algorithm Performance")

# # Sidebar for navigation
# page = st.sidebar.radio("Select a page", ["Model Evaluation", "Data Visualization"])

# # Model Evaluation Page
# if page == "Model Evaluation":
#     st.sidebar.header("Upload Dataset")
#     uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("Dataset Preview:")
#         st.dataframe(df.head())

#         # Preprocess the data
#         st.write("Handling missing values...")

#         # Identify categorical and numerical columns
#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

#         # Impute missing values
#         imputer_cat = SimpleImputer(strategy='most_frequent')
#         df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

#         imputer_num = SimpleImputer(strategy='mean')
#         df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

#         # Remove duplicates
#         df = df.drop_duplicates()

#         st.write(f"Dataset after handling missing values and removing duplicates:")
#         st.dataframe(df.head())

#         # Display available columns
#         st.write(f"Available Columns in Dataset: {df.columns.tolist()}")

#         # Sidebar for feature and target selection
#         st.sidebar.header("Select Features and Target Column")
#         feature_columns = st.sidebar.multiselect("Select Feature Columns", df.columns.tolist(), default=df.columns.tolist()[:-1])
#         target_column = st.sidebar.selectbox("Select Target Column", df.columns.tolist())

#         if target_column not in df.columns or any(col not in df.columns for col in feature_columns):
#             st.error("One or more of the selected columns are not present in the dataset!")
#         else:
#             # Label Encoding for categorical columns
#             label_encoder = LabelEncoder()
#             if df[target_column].dtype == 'object':
#                 df[target_column] = label_encoder.fit_transform(df[target_column])

#             categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
#             for col in categorical_cols:
#                 df[col] = label_encoder.fit_transform(df[col])

#             # Split data
#             X = df[feature_columns]
#             y = df[target_column]
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#             # Scaling options
#             scaling_option = st.sidebar.selectbox("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
#             if scaling_option == "StandardScaler":
#                 scaler = StandardScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)
#             elif scaling_option == "MinMaxScaler":
#                 scaler = MinMaxScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)

#             # SMOTE option for class imbalance
#             if st.sidebar.checkbox("Apply SMOTE for Class Imbalance", False):
#                 smote = SMOTE(sampling_strategy='auto', random_state=42)
#                 X_train, y_train = smote.fit_resample(X_train, y_train)
#                 st.write("SMOTE applied!")

#             # Model selection from sidebar
#             st.sidebar.header("Select Algorithm for Evaluation")
#             algorithm = st.sidebar.selectbox("Select Algorithm", [
#                 "Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "Naive Bayes"])

#             # Define models
#             models = {
#                 "Logistic Regression": LogisticRegression(max_iter=1000),
#                 "Random Forest": RandomForestClassifier(),
#                 "SVM": SVC(probability=True),
#                 "KNN": KNeighborsClassifier(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Naive Bayes": GaussianNB()
#             }

#             # K-Fold Cross Validation and Performance
#             if st.sidebar.checkbox("Perform K-Fold Cross Validation", False):
#                 st.write("Performing K-Fold Cross Validation...")

#                 kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#                 cv_results = cross_val_score(models[algorithm], X, y, cv=kfold, scoring='accuracy')
#                 st.write(f"Accuracy: {cv_results.mean():.2f} (+/- {cv_results.std():.2f})")

#             # Bagging and Boosting
#             st.sidebar.header("Bagging & Boosting")
#             try_bagging = st.sidebar.checkbox("Try Bagging Model", False)
#             try_boosting = st.sidebar.checkbox("Try Boosting Model", False)

#             if try_bagging:
#                 st.write("Performing Bagging...")
#                 bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
#                 bagging_model.fit(X_train, y_train)
#                 bagging_pred = bagging_model.predict(X_test)
#                 bagging_accuracy = accuracy_score(y_test, bagging_pred)
#                 st.write(f"Bagging Accuracy: {bagging_accuracy * 100:.2f}%")

#                 # Save Model Button
#                 if st.sidebar.button("Save Bagging Model"):
#                     joblib.dump(bagging_model, "bagging_model.pkl")
#                     st.success("Bagging Model Saved!")

#             if try_boosting:
#                 st.write("Performing Boosting...")
#                 boosting_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
#                 boosting_model.fit(X_train, y_train)
#                 boosting_pred = boosting_model.predict(X_test)
#                 boosting_accuracy = accuracy_score(y_test, boosting_pred)
#                 st.write(f"Boosting Accuracy: {boosting_accuracy * 100:.2f}%")

#                 # Save Model Button
#                 if st.sidebar.button("Save Boosting Model"):
#                     joblib.dump(boosting_model, "boosting_model.pkl")
#                     st.success("Boosting Model Saved!")

#             # Evaluate Model
#             if algorithm in models:
#                 model = models[algorithm]
#                 model.fit(X_train, y_train)

#                 # Predictions and Evaluation
#                 y_pred = model.predict(X_test)
#                 accuracy = accuracy_score(y_test, y_pred)

#                 # Display accuracy
#                 st.subheader(f"Accuracy of {algorithm}: {accuracy * 100:.2f}%")

#                 # Confusion Matrix
#                 cm = confusion_matrix(y_test, y_pred)
#                 st.subheader("Confusion Matrix:")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
#                 ax.set_xlabel('Predicted Labels')
#                 ax.set_ylabel('True Labels')
#                 ax.set_title('Confusion Matrix')
#                 st.pyplot(fig)

#                 # Classification Report
#                 report = classification_report(y_test, y_pred, output_dict=True)
#                 report_df = pd.DataFrame(report).transpose()
#                 st.subheader("Classification Report:")
#                 st.dataframe(report_df)

# # Data Visualization Page
# elif page == "Data Visualization":
#     st.sidebar.header("Data Visualization Options")
#     uploaded_file = st.sidebar.file_uploader("Upload CSV for Visualization", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("Dataset Preview:")
#         st.dataframe(df.head())

#         # Preprocess the data (same as in Model Evaluation)
#         st.write("Handling missing values...")

#         # Identify categorical and numerical columns
#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

#         # Impute missing values
#         imputer_cat = SimpleImputer(strategy='most_frequent')
#         df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

#         imputer_num = SimpleImputer(strategy='mean')
#         df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

#         # Remove duplicates
#         df = df.drop_duplicates()

#         # Label Encoding for categorical columns
#         label_encoder = LabelEncoder()
#         for col in categorical_cols:
#             df[col] = label_encoder.fit_transform(df[col])

#         # Visualization options
#         visualization_option = st.sidebar.selectbox("Select Visualization", [
#             "None", "Histogram", "Correlation Heatmap", "Boxplot", "Pairplot"])

#         if visualization_option == "Histogram":
#             numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#             selected_column = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
#             st.write(f"Histogram of {selected_column}")
#             fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
#             sns.histplot(df[selected_column], kde=True, ax=ax)  # Pass ax to sns
#             st.pyplot(fig)  # Now passing the figure object

#         if visualization_option == "Correlation Heatmap":
#             st.write("Correlation Heatmap of Features")
#             fig, ax = plt.subplots(figsize=(10, 8))  # Creating a figure object
#             corr = df.corr()
#             sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)  # Pass ax to sns
#             st.pyplot(fig)  # Now passing the figure object

#         if visualization_option == "Boxplot":
#             numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#             selected_column = st.sidebar.selectbox("Select Column for Boxplot", numeric_columns)
#             st.write(f"Boxplot of {selected_column}")
#             fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
#             sns.boxplot(x=df[selected_column], ax=ax)  # Pass ax to sns
#             st.pyplot(fig)  # Now passing the figure object

#         if visualization_option == "Pairplot":
#             numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#             selected_columns = st.sidebar.multiselect("Select Columns for Pairplot", numeric_columns)
#             if len(selected_columns) > 1:
#                 st.write(f"Pairplot of {', '.join(selected_columns)}")
#                 fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
#                 sns.pairplot(df[selected_columns])  # Pairplot doesn't need ax, it creates its own
#                 st.pyplot(fig)  # Now passing the figure object




# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
# from imblearn.over_sampling import SMOTE
# import joblib

# # Streamlit App
# st.title("Machine Learning Algorithm Performance")

# # Sidebar for navigation
# page = st.sidebar.radio("Select a page", ["Model Evaluation", "Data Visualization"])

# # Model Evaluation Page
# if page == "Model Evaluation":
#     st.sidebar.header("Upload Dataset")
#     uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("Dataset Preview:")
#         st.dataframe(df.head())

#         # Preprocess the data
#         st.write("Handling missing values...")

#         categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
#         numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

#         # Impute missing values
#         from sklearn.impute import SimpleImputer
#         imputer_cat = SimpleImputer(strategy='most_frequent')
#         df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

#         imputer_num = SimpleImputer(strategy='mean')
#         df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])

#         # Remove duplicates
#         df = df.drop_duplicates()

#         st.write(f"Dataset after handling missing values and removing duplicates:")
#         st.dataframe(df.head())

#         # Display available columns
#         st.write(f"Available Columns in Dataset: {df.columns.tolist()}")

#         # Sidebar for feature and target selection
#         st.sidebar.header("Select Features and Target Column")
#         feature_columns = st.sidebar.multiselect("Select Feature Columns", df.columns.tolist(), default=df.columns.tolist()[:-1])
#         target_column = st.sidebar.selectbox("Select Target Column", df.columns.tolist())

#         if target_column not in df.columns or any(col not in df.columns for col in feature_columns):
#             st.error("One or more of the selected columns are not present in the dataset!")
#         else:
#             # Label Encoding for categorical columns
#             label_encoder = LabelEncoder()
#             if df[target_column].dtype == 'object':
#                 df[target_column] = label_encoder.fit_transform(df[target_column])

#             categorical_cols = df[feature_columns].select_dtypes(include=['object', 'category']).columns.tolist()
#             for col in categorical_cols:
#                 df[col] = label_encoder.fit_transform(df[col])

#             # Split data
#             X = df[feature_columns]
#             y = df[target_column]
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#             # Scaling options
#             scaling_option = st.sidebar.selectbox("Select Scaling Method", ["None", "StandardScaler", "MinMaxScaler"])
#             if scaling_option == "StandardScaler":
#                 scaler = StandardScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)
#             elif scaling_option == "MinMaxScaler":
#                 scaler = MinMaxScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)

#             # SMOTE option for class imbalance
#             if st.sidebar.checkbox("Apply SMOTE for Class Imbalance", False):
#                 smote = SMOTE(sampling_strategy='auto', random_state=42)
#                 X_train, y_train = smote.fit_resample(X_train, y_train)
#                 st.write("SMOTE applied!")

#             # Model selection from sidebar
#             st.sidebar.header("Select Algorithm for Evaluation")
#             algorithm = st.sidebar.selectbox("Select Algorithm", [
#                 "Logistic Regression", "Random Forest", "SVM", "KNN", "Decision Tree", "Naive Bayes"])

#             # Define models
#             models = {
#                 "Logistic Regression": LogisticRegression(max_iter=1000),
#                 "Random Forest": RandomForestClassifier(),
#                 "SVM": SVC(probability=True),
#                 "KNN": KNeighborsClassifier(),
#                 "Decision Tree": DecisionTreeClassifier(),
#                 "Naive Bayes": GaussianNB()
#             }

#             # K-Fold Cross Validation and Performance
#             if st.sidebar.checkbox("Perform K-Fold Cross Validation", False):
#                 st.write("Performing K-Fold Cross Validation...")

#                 kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#                 cv_results = cross_val_score(models[algorithm], X, y, cv=kfold, scoring='accuracy')
#                 st.write(f"Accuracy: {cv_results.mean():.2f} (+/- {cv_results.std():.2f})")

#             # Bagging and Boosting
#             st.sidebar.header("Bagging & Boosting")
#             try_bagging = st.sidebar.checkbox("Try Bagging Model", False)
#             try_boosting = st.sidebar.checkbox("Try Boosting Model", False)

#             if try_bagging:
#                 st.write("Performing Bagging...")
#                 bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
#                 bagging_model.fit(X_train, y_train)
#                 bagging_pred = bagging_model.predict(X_test)
#                 bagging_accuracy = accuracy_score(y_test, bagging_pred)
#                 st.write(f"Bagging Accuracy: {bagging_accuracy * 100:.2f}%")

#                 # Save Model Button
#                 if st.sidebar.button("Save Bagging Model"):
#                     joblib.dump(bagging_model, "bagging_model.pkl")
#                     st.success("Bagging Model Saved!")

#             if try_boosting:
#                 st.write("Performing Boosting...")
#                 boosting_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50)
#                 boosting_model.fit(X_train, y_train)
#                 boosting_pred = boosting_model.predict(X_test)
#                 boosting_accuracy = accuracy_score(y_test, boosting_pred)
#                 st.write(f"Boosting Accuracy: {boosting_accuracy * 100:.2f}%")

#                 # Save Model Button
#                 if st.sidebar.button("Save Boosting Model"):
#                     joblib.dump(boosting_model, "boosting_model.pkl")
#                     st.success("Boosting Model Saved!")

#             # Evaluate Model
#             if algorithm in models:
#                 model = models[algorithm]
#                 model.fit(X_train, y_train)

#                 # Predictions and Evaluation
#                 y_pred = model.predict(X_test)
#                 accuracy = accuracy_score(y_test, y_pred)

#                 # Display accuracy
#                 st.subheader(f"Accuracy of {algorithm}: {accuracy * 100:.2f}%")

#                 # Confusion Matrix
#                 cm = confusion_matrix(y_test, y_pred)
#                 st.subheader("Confusion Matrix:")
#                 fig, ax = plt.subplots(figsize=(8, 6))
#                 sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
#                 ax.set_xlabel('Predicted Labels')
#                 ax.set_ylabel('True Labels')
#                 ax.set_title('Confusion Matrix')
#                 st.pyplot(fig)

#                 # Classification Report
#                 report = classification_report(y_test, y_pred, output_dict=True)
#                 report_df = pd.DataFrame(report).transpose()
#                 st.subheader("Classification Report:")
#                 st.dataframe(report_df)

# # Data Visualization Page
# elif page == "Data Visualization":
#     st.sidebar.header("Data Visualization Options")
#     uploaded_file = st.sidebar.file_uploader("Upload CSV for Visualization", type="csv")

#     if uploaded_file is not None:
#         df = pd.read_csv(uploaded_file)
#         st.write("Dataset Preview:")
#         st.dataframe(df.head())

#         visualization_option = st.sidebar.selectbox("Select Visualization", [
#             "None", "Histogram", "Correlation Heatmap", "Boxplot", "Pairplot"])

#         if visualization_option == "Histogram":
#             numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#             selected_column = st.sidebar.selectbox("Select Column for Histogram", numeric_columns)
#             st.write(f"Histogram of {selected_column}")
#             fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
#             sns.histplot(df[selected_column], kde=True, ax=ax)  # Pass ax to sns
#             st.pyplot(fig)  # Now passing the figure object

#         if visualization_option == "Correlation Heatmap":
#             st.write("Correlation Heatmap of Features")
#             fig, ax = plt.subplots(figsize=(10, 8))  # Creating a figure object
#             corr = df.corr()
#             sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)  # Pass ax to sns
#             st.pyplot(fig)  # Now passing the figure object

#         if visualization_option == "Boxplot":
#             numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#             selected_column = st.sidebar.selectbox("Select Column for Boxplot", numeric_columns)
#             st.write(f"Boxplot of {selected_column}")
#             fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
#             sns.boxplot(x=df[selected_column], ax=ax)  # Pass ax to sns
#             st.pyplot(fig)  # Now passing the figure object

#         if visualization_option == "Pairplot":
#             numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#             selected_columns = st.sidebar.multiselect("Select Columns for Pairplot", numeric_columns)
#             if len(selected_columns) > 1:
#                 st.write(f"Pairplot of {', '.join(selected_columns)}")
#                 fig, ax = plt.subplots(figsize=(8, 6))  # Creating a figure object
#                 sns.pairplot(df[selected_columns])  # Pairplot doesn't need ax, it creates its own
#                 st.pyplot(fig)  # Now passing the figure object
