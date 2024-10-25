# app.py
import streamlit as st
import pandas as pd
import pickle

# Load the saved model and encoders
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('team_encoder.pkl', 'rb') as team_enc_file:
    team_encoder = pickle.load(team_enc_file)

with open('winner_encoder.pkl', 'rb') as winner_enc_file:
    winner_encoder = pickle.load(winner_enc_file)

# Streamlit UI for Prediction
st.title("Euro 2024 Match Outcome Predictor")

# Create input widgets for user input
home_team = st.selectbox("Select Home Team", team_encoder.classes_)
away_team = st.selectbox("Select Away Team", team_encoder.classes_)
home_goals = st.number_input("Home Team Goals", min_value=0, max_value=10, value=0)
away_goals = st.number_input("Away Team Goals", min_value=0, max_value=10, value=0)

# Process input for prediction
if st.button("Predict Winner"):
    # Encode the user input
    home_team_encoded = team_encoder.transform([home_team])[0]
    away_team_encoded = team_encoder.transform([away_team])[0]

    # Create the input dataframe
    input_data = pd.DataFrame({
        'home_team_encoded': [home_team_encoded],
        'away_team_encoded': [away_team_encoded],
        'home_goals': [home_goals],
        'away_goals': [away_goals]
    })

    # Predict the winner
    prediction_encoded = model.predict(input_data)[0]
    prediction = winner_encoder.inverse_transform([prediction_encoded])[0]

    # Display the result
    st.success(f"The predicted outcome is: {prediction}")