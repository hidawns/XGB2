import streamlit as st
import joblib
import numpy as np

# Load the saved model, encoder, and scaler
model = joblib.load('xgb_model (2).pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')
pipeline = joblib.load('model_pipeline.pkl')

# Title of the app
st.title('Doctor Consultation Fee Prediction')

# Function to get user inputs
def user_input():
    Experience = st.number_input('Years of Experience', min_value=0, max_value=66, step=1, value=0)
    Num_of_Qualifications = st.number_input('Number of Qualifications', min_value=1, max_value=10, step=1, value=1)
    Rating = st.number_input('Doctor Rating', min_value=1, max_value=100, step=1, value=50)
    
    Miscellaneous_Info = st.selectbox('Miscellaneous Info Existent', options=['Not Present', 'Present'])
    Miscellaneous_Info = 1 if Miscellaneous_Info == 'Present' else 0

    Profile = st.selectbox('Doctor Specialization', options=[
        'Ayurveda', 'Dentist', 'Dermatologist', 'ENT Specialist', 
        'General Medicine', 'Homeopath'
    ])
    Profile_mapping = {'Ayurveda': 0, 'Dentist': 1, 'Dermatologist': 2, 'ENT Specialist': 3, 
                       'General Medicine': 4, 'Homeopath': 5}
    Profile = Profile_mapping[Profile]

    Place = st.selectbox('Place', options=[
        'Bangalore', 'Mumbai', 'Delhi', 'Hyderabad', 
        'Chennai', 'Coimbatore', 'Ernakulam', 'Thiruvananthapuram', 'Other'
    ])
    Place_mapping = {'Bangalore': 0, 'Mumbai': 1, 'Delhi': 2, 'Hyderabad': 3, 
                     'Chennai': 4, 'Coimbatore': 5, 'Ernakulam': 6, 
                     'Thiruvananthapuram': 7, 'Other': 8}
    Place = Place_mapping[Place]

    Fee_category = 0.0  # Initialize Fee_category as 0.0 since it is not an input

    # Creating a dataframe of the user inputs
    data = {
        'Experience': Experience,
        'Rating': Rating,
        'Num_of_Qualifications': Num_of_Qualifications,
        'Miscellaneous_Info': Miscellaneous_Info,
        'Profile': Profile,
        'Place': Place,
        'Fee_category': Fee_category
    }
    
    features = np.array([[Experience, Rating, Num_of_Qualifications, Miscellaneous_Info, Profile, Place, Fee_category]])
    return features

# Get user input
input_data = user_input()

# Make predictions
if st.button('Predict Fee'):
    prediction = model.predict(input_data)
    st.write(f"Predicted Doctor Consultation Fee: ₹{np.round(prediction[0], 2)}")
