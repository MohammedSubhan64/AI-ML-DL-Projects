import streamlit as st
import pickle
import numpy as np

# Load the trained model (replace with the correct path to your .pkl file)
model_path = 'models/diabetes_model.pkl'  # Path to your trained model (.pkl file)
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Function to predict diabetes
def predict_diabetes(input_data):
    prediction = model.predict([input_data])
    return prediction[0]

# Streamlit UI setup
st.set_page_config(page_title="Diabetes Prediction", page_icon="🍏", layout="centered")
st.title('Diabetes Prediction System 🍏')
st.markdown("""
    **Welcome to the Diabetes Prediction App!**  
    This tool helps predict whether you are at risk of diabetes based on a set of input features.  
    Please fill out the following information to get your prediction.  
    *Your health is your wealth!* 💚
""", unsafe_allow_html=True)

# Add a diabetes symbol (Using an emoji for the diabetes symbol)
st.markdown("<h3 style='color: #1E90FF;'>Please answer the following questions:</h3>", unsafe_allow_html=True)

# Input Fields with icons and diabetes-related symbols
age = st.number_input('🧑‍⚕️ Age', min_value=0, max_value=120, help="Age in years. Enter your age (e.g., 30, 55).")

gender = st.selectbox('🚻 Gender', options=['Male', 'Female'], help="Gender. Select 'Male' or 'Female'.")
gender = 1 if gender == 'Male' else 0  # Convert 'Male' to 1 and 'Female' to 0

polyuria = st.selectbox('💧 Polyuria (Frequent Urination)', options=['Yes', 'No'], help="Do you experience frequent urination?")
polyuria = 1 if polyuria == 'Yes' else 0

polydipsia = st.selectbox('💦 Polydipsia (Excessive Thirst)', options=['Yes', 'No'], help="Do you experience excessive thirst?")
polydipsia = 1 if polydipsia == 'Yes' else 0

sudden_weight_loss = st.selectbox('⚖️ Sudden Weight Loss', options=['Yes', 'No'], help="Have you experienced sudden weight loss?")
sudden_weight_loss = 1 if sudden_weight_loss == 'Yes' else 0

weakness = st.selectbox('💪 Weakness', options=['Yes', 'No'], help="Do you feel unusually weak?")
weakness = 1 if weakness == 'Yes' else 0

polyphagia = st.selectbox('🍽️ Polyphagia (Excessive Hunger)', options=['Yes', 'No'], help="Do you feel extremely hungry all the time?")
polyphagia = 1 if polyphagia == 'Yes' else 0

genital_thrush = st.selectbox('🦠 Genital Thrush', options=['Yes', 'No'], help="Have you experienced genital thrush?")
genital_thrush = 1 if genital_thrush == 'Yes' else 0

visual_blurring = st.selectbox('👀 Visual Blurring', options=['Yes', 'No'], help="Do you have blurred vision?")
visual_blurring = 1 if visual_blurring == 'Yes' else 0

itching = st.selectbox('🦶 Itching', options=['Yes', 'No'], help="Do you experience excessive itching?")
itching = 1 if itching == 'Yes' else 0

irritability = st.selectbox('😡 Irritability', options=['Yes', 'No'], help="Do you feel unusually irritable?")
irritability = 1 if irritability == 'Yes' else 0

delayed_healing = st.selectbox('⏳ Delayed Healing', options=['Yes', 'No'], help="Have you noticed that your wounds take longer to heal?")
delayed_healing = 1 if delayed_healing == 'Yes' else 0

partial_paresis = st.selectbox('🦵 Partial Paresis (Partial Paralysis)', options=['Yes', 'No'], help="Do you experience partial paralysis or weakness in any part of your body?")
partial_paresis = 1 if partial_paresis == 'Yes' else 0

muscle_stiffness = st.selectbox('💪 Muscle Stiffness', options=['Yes', 'No'], help="Do you experience muscle stiffness or joint pain?")
muscle_stiffness = 1 if muscle_stiffness == 'Yes' else 0

alopecia = st.selectbox('🧑‍🦲 Alopecia (Hair Loss)', options=['Yes', 'No'], help="Have you experienced hair loss?")
alopecia = 1 if alopecia == 'Yes' else 0

obesity = st.selectbox('🍏 Obesity', options=['Yes', 'No'], help="Are you overweight or obese?")
obesity = 1 if obesity == 'Yes' else 0

# Collecting all input data into a numpy array
input_data = np.array([age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
                       polyphagia, genital_thrush, visual_blurring, itching, irritability,
                       delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity])

# Button to trigger prediction
if st.button('Predict Diabetes'):
    prediction = predict_diabetes(input_data)
    
    # Display prediction result with icons and colors
    if prediction == 1:
        st.markdown("""
            <h3 style='color: red;'>Prediction: Diabetes </h3>
            <p style='color: red;'>You are predicted to have diabetes. Please consult a healthcare professional for further diagnosis. 🩺</p>
        """, unsafe_allow_html=True)
        # Add a relevant diabetes icon or symbol
        # st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/0/01/Diabetes_symbol.svg/800px-Diabetes_symbol.svg.png", width=200)

    else:
        st.markdown("""
            <h3 style='color: green;'>Prediction: No Diabetes</h3>
            <p style='color: green;'>You are not predicted to have diabetes, but it's always good to maintain a healthy lifestyle. 💚</p>
        """, unsafe_allow_html=True)
        # Optional: Add a healthy lifestyle icon or symbol
        # st.image("https://upload.wikimedia.org/wikipedia/commons/7/75/Green_Apple.png", width=200)





# import streamlit as st
# import pickle
# import numpy as np

# # Load the trained model (replace with the correct path to your .pkl file)
# model_path = 'models/diabetes_model.pkl'  # Path to your trained model (.pkl file)
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# # Function to predict diabetes
# def predict_diabetes(input_data):
#     prediction = model.predict([input_data])
#     return prediction[0]
# # 
# # Streamlit UI setup
# st.title('Diabetes Prediction')
# st.write('Enter the details below to predict whether you have diabetes or not.')

# # Input fields with tooltips for reference ranges (using Yes/No options in selectbox)
# age = st.number_input('Age', min_value=0, max_value=120, help="Age in years. Enter your age (e.g., 30, 55).")

# gender = st.selectbox('Gender', options=['Male', 'Female'], help="Gender. Select 'Male' or 'Female'.")
# gender = 1 if gender == 'Male' else 0  # Convert 'Male' to 1 and 'Female' to 0

# polyuria = st.selectbox('Polyuria (Frequent Urination)', options=['Yes', 'No'], help="Do you experience frequent urination?")
# polyuria = 1 if polyuria == 'Yes' else 0

# polydipsia = st.selectbox('Polydipsia (Excessive Thirst)', options=['Yes', 'No'], help="Do you experience excessive thirst?")
# polydipsia = 1 if polydipsia == 'Yes' else 0

# sudden_weight_loss = st.selectbox('Sudden Weight Loss', options=['Yes', 'No'], help="Have you experienced sudden weight loss?")
# sudden_weight_loss = 1 if sudden_weight_loss == 'Yes' else 0

# weakness = st.selectbox('Weakness', options=['Yes', 'No'], help="Do you feel unusually weak?")
# weakness = 1 if weakness == 'Yes' else 0

# polyphagia = st.selectbox('Polyphagia (Excessive Hunger)', options=['Yes', 'No'], help="Do you feel extremely hungry all the time?")
# polyphagia = 1 if polyphagia == 'Yes' else 0

# genital_thrush = st.selectbox('Genital Thrush', options=['Yes', 'No'], help="Have you experienced genital thrush?")
# genital_thrush = 1 if genital_thrush == 'Yes' else 0

# visual_blurring = st.selectbox('Visual Blurring', options=['Yes', 'No'], help="Do you have blurred vision?")
# visual_blurring = 1 if visual_blurring == 'Yes' else 0

# itching = st.selectbox('Itching', options=['Yes', 'No'], help="Do you experience excessive itching?")
# itching = 1 if itching == 'Yes' else 0

# irritability = st.selectbox('Irritability', options=['Yes', 'No'], help="Do you feel unusually irritable?")
# irritability = 1 if irritability == 'Yes' else 0

# delayed_healing = st.selectbox('Delayed Healing', options=['Yes', 'No'], help="Have you noticed that your wounds take longer to heal?")
# delayed_healing = 1 if delayed_healing == 'Yes' else 0

# partial_paresis = st.selectbox('Partial Paresis (Partial Paralysis)', options=['Yes', 'No'], help="Do you experience partial paralysis or weakness in any part of your body?")
# partial_paresis = 1 if partial_paresis == 'Yes' else 0

# muscle_stiffness = st.selectbox('Muscle Stiffness', options=['Yes', 'No'], help="Do you experience muscle stiffness or joint pain?")
# muscle_stiffness = 1 if muscle_stiffness == 'Yes' else 0

# alopecia = st.selectbox('Alopecia (Hair Loss)', options=['Yes', 'No'], help="Have you experienced hair loss?")
# alopecia = 1 if alopecia == 'Yes' else 0

# obesity = st.selectbox('Obesity', options=['Yes', 'No'], help="Are you overweight or obese?")
# obesity = 1 if obesity == 'Yes' else 0

# # Collecting all input data into a numpy array
# input_data = np.array([age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
#                        polyphagia, genital_thrush, visual_blurring, itching, irritability,
#                        delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity])

# # Button to trigger prediction
# if st.button('Predict Diabetes'):
#     prediction = predict_diabetes(input_data)
    
#     if prediction == 1:
#         st.write("### Prediction: **Diabetes**")
#         st.write("You are predicted to have diabetes. Please consult a healthcare professional for further diagnosis.")
#     else:
#         st.write("### Prediction: **No Diabetes**")
#         st.write("You are not predicted to have diabetes, but it's always good to maintain a healthy lifestyle.")

# import streamlit as st
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# # Load the trained model (replace with the correct path to your .pkl file)
# model_path = 'models/diabetes_model.pkl'  # Path to your trained model (.pkl file)
# with open(model_path, 'rb') as f:
#     model = pickle.load(f)

# # Function to predict diabetes
# def predict_diabetes(input_data):
#     prediction = model.predict([input_data])
#     return prediction[0]

# # Function to plot confusion matrix using Plotly (Interactive)
# def plot_confusion_matrix_plotly(y_true, y_pred):
#     cm = confusion_matrix(y_true, y_pred)
#     fig = go.Figure(data=go.Heatmap(
#         z=cm,
#         x=['No Diabetes', 'Diabetes'],
#         y=['No Diabetes', 'Diabetes'],
#         colorscale='Blues',
#         colorbar=dict(title="Count"),
#         zmin=0, zmax=cm.max()
#     ))

#     fig.update_layout(
#         title="Confusion Matrix",
#         xaxis_title="Predicted",
#         yaxis_title="Actual",
#         xaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes']),
#         yaxis=dict(tickmode='array', tickvals=[0, 1], ticktext=['No Diabetes', 'Diabetes']),
#         height=400
#     )

#     st.plotly_chart(fig)

# # Function to plot ROC Curve using Plotly (Interactive)
# def plot_roc_curve_plotly(y_true, y_pred_proba):
#     fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
#     roc_auc = auc(fpr, tpr)

#     fig = go.Figure()

#     # Plot ROC curve
#     fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC Curve (AUC = {roc_auc:.2f})"))

#     # Plot diagonal line (random chance)
#     fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name="Random", line=dict(dash='dash')))

#     fig.update_layout(
#         title="Receiver Operating Characteristic (ROC) Curve",
#         xaxis_title="False Positive Rate",
#         yaxis_title="True Positive Rate",
#         height=500,
#         width=600
#     )

#     st.plotly_chart(fig)

# # Function to plot feature importance using Plotly (Interactive)
# def plot_feature_importance_plotly(feature_names, feature_importance):
#     fig = go.Figure()

#     fig.add_trace(go.Bar(
#         x=feature_importance,
#         y=feature_names,
#         orientation='h',
#         marker=dict(color=feature_importance, colorscale='Viridis'),
#         text=feature_importance,
#         hoverinfo='x+text'
#     ))

#     fig.update_layout(
#         title="Feature Importance",
#         xaxis_title="Importance",
#         yaxis_title="Features",
#         height=500,
#         width=800
#     )

#     st.plotly_chart(fig)

# # Streamlit UI setup
# st.title('Diabetes Prediction System')
# st.markdown("""
#     **Welcome to the Diabetes Prediction App!**  
#     This tool helps predict whether you are at risk of diabetes based on a set of input features.  
#     Please fill out the following information to get your prediction and view the model's performance.
# """)

# st.header('Enter Your Information')
# st.write('Provide your details to make a prediction.')

# # Input fields with tooltips for reference ranges (using Yes/No options in selectbox)
# age = st.number_input('Age', min_value=0, max_value=120, help="Age in years. Enter your age (e.g., 30, 55).")

# gender = st.selectbox('Gender', options=['Male', 'Female'], help="Select your gender. 'Male' or 'Female'.")
# gender = 1 if gender == 'Male' else 0  # Convert 'Male' to 1 and 'Female' to 0

# polyuria = st.selectbox('Polyuria (Frequent Urination)', options=['Yes', 'No'], help="Do you experience frequent urination?")
# polyuria = 1 if polyuria == 'Yes' else 0

# polydipsia = st.selectbox('Polydipsia (Excessive Thirst)', options=['Yes', 'No'], help="Do you experience excessive thirst?")
# polydipsia = 1 if polydipsia == 'Yes' else 0

# sudden_weight_loss = st.selectbox('Sudden Weight Loss', options=['Yes', 'No'], help="Have you experienced sudden weight loss?")
# sudden_weight_loss = 1 if sudden_weight_loss == 'Yes' else 0

# weakness = st.selectbox('Weakness', options=['Yes', 'No'], help="Do you feel unusually weak?")
# weakness = 1 if weakness == 'Yes' else 0

# polyphagia = st.selectbox('Polyphagia (Excessive Hunger)', options=['Yes', 'No'], help="Do you feel extremely hungry all the time?")
# polyphagia = 1 if polyphagia == 'Yes' else 0

# genital_thrush = st.selectbox('Genital Thrush', options=['Yes', 'No'], help="Have you experienced genital thrush?")
# genital_thrush = 1 if genital_thrush == 'Yes' else 0

# visual_blurring = st.selectbox('Visual Blurring', options=['Yes', 'No'], help="Do you have blurred vision?")
# visual_blurring = 1 if visual_blurring == 'Yes' else 0

# itching = st.selectbox('Itching', options=['Yes', 'No'], help="Do you experience excessive itching?")
# itching = 1 if itching == 'Yes' else 0

# irritability = st.selectbox('Irritability', options=['Yes', 'No'], help="Do you feel unusually irritable?")
# irritability = 1 if irritability == 'Yes' else 0

# delayed_healing = st.selectbox('Delayed Healing', options=['Yes', 'No'], help="Have you noticed that your wounds take longer to heal?")
# delayed_healing = 1 if delayed_healing == 'Yes' else 0

# partial_paresis = st.selectbox('Partial Paresis (Partial Paralysis)', options=['Yes', 'No'], help="Do you experience partial paralysis or weakness?")
# partial_paresis = 1 if partial_paresis == 'Yes' else 0

# muscle_stiffness = st.selectbox('Muscle Stiffness', options=['Yes', 'No'], help="Do you experience muscle stiffness or joint pain?")
# muscle_stiffness = 1 if muscle_stiffness == 'Yes' else 0

# alopecia = st.selectbox('Alopecia (Hair Loss)', options=['Yes', 'No'], help="Have you experienced hair loss?")
# alopecia = 1 if alopecia == 'Yes' else 0

# obesity = st.selectbox('Obesity', options=['Yes', 'No'], help="Are you overweight or obese?")
# obesity = 1 if obesity == 'Yes' else 0

# # Collecting all input data into a numpy array
# input_data = np.array([age, gender, polyuria, polydipsia, sudden_weight_loss, weakness,
#                        polyphagia, genital_thrush, visual_blurring, itching, irritability,
#                        delayed_healing, partial_paresis, muscle_stiffness, alopecia, obesity])

# if st.button('Predict Diabetes'):
#     prediction = predict_diabetes(input_data)
#     if prediction == 1:
#         st.write("### Prediction: **Diabetes**")
#         st.write("You are predicted to have diabetes. Please consult a healthcare professional for further diagnosis.")
#     else:
#         st.write("### Prediction: **No Diabetes**")
#         st.write("You are not predicted to have diabetes, but it's always good to maintain a healthy lifestyle.")

#     # Show performance metrics and visualizations
#     st.subheader('Model Performance & Accuracy')

#     # Assuming the model was trained and tested elsewhere, we load some dummy test data for visualizations
#     # Here we use placeholders for the demonstration purpose

#     # Placeholder actual labels and predicted labels for performance metrics
#     y_true = [0, 1, 0, 1, 0]  # Actual labels (this should come from your test data)
#     y_pred = [0, 1, 0, 0, 0]  # Predicted labels (this should be from model prediction)
#     y_pred_proba = [0.2, 0.9, 0.4, 0.3, 0.7]  # Predicted probabilities (e.g., from logistic regression or similar models)

#     # Accuracy
#     accuracy = accuracy_score(y_true, y_pred)
#     st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

#     # Confusion Matrix
#     st.write("**Confusion Matrix (Interactive):**")
#     plot_confusion_matrix_plotly(y_true, y_pred)

#     # ROC Curve
#     st.write("**ROC Curve (Interactive):**")
#     plot_roc_curve_plotly(y_true, y_pred_proba)

#     # Show Feature Importance (Optional if available in the model)
#     if hasattr(model, 'feature_importances_'):
#         st.subheader('Feature Importance (Interactive)')
#         feature_names = ['Age', 'Gender', 'Polyuria', 'Polydipsia', 'Sudden Weight Loss', 'Weakness',
#                          'Polyphagia', 'Genital Thrush', 'Visual Blurring', 'Itching', 'Irritability',
#                          'Delayed Healing', 'Partial Paresis', 'Muscle Stiffness', 'Alopecia', 'Obesity']
#         feature_importance = model.feature_importances_
#         plot_feature_importance_plotly(feature_names, feature_importance)

#     st.write('---')
#     st.write("Note: The above performance metrics are based on a placeholder dataset. In production, these would be based on real test data.")
