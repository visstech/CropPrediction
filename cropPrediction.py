import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load Cleaned Dataset
data_path = "c:\\ML\\Cleaned_FAOSTAT_data.csv"
df = pd.read_csv(data_path)

# App Title
st.title("Crop Production Prediction App")
st.write("This app predicts crop production based on agricultural data.")

# Sidebar for User Input
st.sidebar.header("Input Features")

# Function to get user inputs
def user_input_features():
    area = st.sidebar.selectbox('Select Region (Area)', df['Area'].unique())
    item = st.sidebar.selectbox('Select Crop (Item)', df['Item'].unique())
    year = st.sidebar.slider('Year', int(df['Year'].min()), int(df['Year'].max()), step=1)
    area_harvested = st.sidebar.number_input('Area Harvested (scaled)', min_value=float(df['Area_Harvested'].min()), max_value=float(df['Area_Harvested'].max()), step=0.1)
    yield_val = st.sidebar.number_input('Yield (scaled)', min_value=float(df['Yield'].min()), max_value=float(df['Yield'].max()), step=0.1)

    features = {
        'Area': area,
        'Item': item,
        'Year': year,
        'Area_Harvested': area_harvested,
        'Yield': yield_val
    }
    return pd.DataFrame(features, index=[0])

input_df = user_input_features()

# Preprocess Data: Encode Categorical Variables
label_encoders = {}
for col in ['Area', 'Item']:
    label_encoders[col] = {v: k for k, v in enumerate(df[col].unique())}
    df[col] = df[col].map(label_encoders[col])
    if col in input_df.columns:
        input_df[col] = input_df[col].map(label_encoders[col])

# Prepare Data for Training
X = df[['Area', 'Item', 'Year', 'Area_Harvested', 'Yield']]
y = df['Production']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save the Model
with open('crop_production_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Prediction
if st.button("Predict Crop Production"):
    prediction = model.predict(input_df)
    st.write(f"Predicted Crop Production (scaled): {prediction[0]:.2f}")

# Model Evaluation
if st.checkbox("Show Model Evaluation"):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R² Score: {r2:.2f}")
