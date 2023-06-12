import math
import numpy as np
import pickle
import streamlit as st

from PIL import Image
# Load the image
image = Image.open('https://m.media-amazon.com/images/I/71A9RuXTRlL._AC_UF1000,1000_QL80_.jpg')

# Display the image
st.image(image, use_column_width=True)
# Your Streamlit app code here
# ...
filename = 'ml_model.pkl'
model = pickle.load(open(filename, 'rb'))

# Title of the page with CSS
st.markdown("<h1 style='text-align: center; color: black;'>NBA Winner Predictor</h1>", unsafe_allow_html=True)

# Rest of your code for form input, prediction, etc.
# ...


def score_predict(team, team_opp, pts, total, home, pts_opp, total_opp, home_opp, won, model):
    prediction_array = []
  #Team 
    if team == 'DET':
     prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
    elif team == 'PHI':
     prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
    elif team == 'BOS':
     prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
    elif team == 'MIL':
     prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
    elif team == 'ATL':
     prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
    elif team == 'CLE':
     prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
    elif team == 'CHI':
     prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
    elif team == 'LAL':
     prediction_array = prediction_array + [0,0,0,0,0,0,0,1]
  #Team_opp
    if team_opp == 'DET':
     prediction_array = prediction_array + [1,0,0,0,0,0,0,0]
    elif team_opp == 'PHI':
     prediction_array = prediction_array + [0,1,0,0,0,0,0,0]
    elif team_opp == 'BOS':
     prediction_array = prediction_array + [0,0,1,0,0,0,0,0]
    elif team_opp == 'MIL':
     prediction_array = prediction_array + [0,0,0,1,0,0,0,0]
    elif team_opp == 'ATL':
     prediction_array = prediction_array + [0,0,0,0,1,0,0,0]
    elif team_opp == 'CLE':
     prediction_array = prediction_array + [0,0,0,0,0,1,0,0]
    elif team_opp == 'CHI':
     prediction_array = prediction_array + [0,0,0,0,0,0,1,0]
    elif team_opp == 'LAL':
     pprediction_array = prediction_array + [0,0,0,0,0,0,0,1]
    prediction_array = prediction_array + [pts,total,home,pts_opp,total_opp,home_opp]
    prediction_array = np.array([prediction_array])
    pred = model.predict(prediction_array)
    return int(round(pred[0]))

# Select teams and input values
with st.form("Prediction Form"):
    st.write("Enter the details below to predict the winner of an NBA game.")
    st.write("Select the teams:")
    team = st.selectbox("Home Team", ['DET', 'PHI', 'BOS', 'MIL', 'ATL', 'CLE', 'CHI', 'LAL'])
    team_opp = st.selectbox("Away Team", ['DET', 'PHI', 'BOS', 'MIL', 'ATL', 'CLE', 'CHI', 'LAL'])
    
    st.write("Enter the following information:")
    pts = st.number_input("Home Team Current Points", min_value=0)
    total = st.number_input("Total Points", min_value=0)
    home = st.number_input("Home Game (1 for Yes, 0 for No)", min_value=0, max_value=1)
    pts_opp = st.number_input("Away Team Current Points", min_value=0)
    total_opp = st.number_input("Total Points(Current) by Opponent", min_value=0)
    home_opp = st.number_input("Away Game (1 for Yes, 0 for No)", min_value=0, max_value=1)
    won = st.number_input("Number of Games Won by Home Team", min_value=0)
    
    submit_button = st.form_submit_button(label='Predict')

# Perform prediction and display the result
if submit_button:
    prediction = score_predict(team, team_opp, pts, total, home, pts_opp, total_opp, home_opp, won,model)
    st.write("The predicted winner is:", team if prediction == 1 else team_opp)
