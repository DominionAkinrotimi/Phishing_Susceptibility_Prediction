import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MultiLabelBinarizer

# Load the trained model and encoders
model = joblib.load('Phishing_Susceptibility_Prediction_model.pkl')
ordinal_encoder = joblib.load('ordinal_encoder.pkl')
one_hot_encoder = joblib.load('one_hot_encoder.pkl')
mlb_phishing_types = joblib.load('mlb_phishing_types.pkl')
mlb_suspicious_actions = joblib.load('mlb_suspicious_actions.pkl')

# Define the options for PhishingTypes and SuspiciousActions
phishing_types_options = [
    'Emails containing suspicious links or attachments',
    'Emails asking for personal information such as passwords or Social Security numbers',
    'Messages claiming that you have won a prize or lottery',
    'Communications requesting urgent action such as clicking on a link or transferring money',
    'Messages from unfamiliar senders requesting sensitive information or login credentials'
]

suspicious_actions_options = [
    'Do not click on any links or download any attachments',
    'Delete the email or message immediately',
    'Report the email or message as spam',
    'Verify the sender\'s identity through a separate communication channel',
    'Notify your organization\'s IT department or security team'
]

# Define options for other categorical columns
gender_options = ['Prefer not to say', 'Male', 'Female']
education_options = ['Graduate', 'Undergraduate', 'High School']
job_category_options = ['Teacher', 'Administrator', 'Student', 'Parent', 'Customer service officer', 'Entertainer', 'Technician', 'Care Provider', 'Engineering', 'Self employed & undergraduate', 'Lecturer', 'Entrepreneur']
social_media_options = ['Daily', 'Weekly', 'Monthly']
shopping_options = ['Weekly', 'Monthly', 'Seasonal', 'Daily', 'Never', 'Yearly']
videos_options = ['Daily', 'Seasonal', 'Weekly', 'Never', 'Monthly', 'Yearly']
gaming_options = ['Daily', 'Seasonal', 'Monthly', 'Weekly', 'Never', 'Yearly']
learning_options = ['Daily', 'Seasonal', 'Weekly', 'Monthly', 'Never']
communicating_options = ['Daily', 'Monthly', 'Weekly', 'Seasonal', 'Yearly', 'Never']
phishing_experience_options = ['Yes', 'Maybe', 'No']
check_sender_options = ['Always', 'Very often', 'Sometimes', 'Rarely', 'Never', 'Often']
detection_system_options = ['Strongly agree', 'Agree', 'Neither agree nor disagree']

# Function to encode categorical data
def encode_categorical_data(df):
    ordinal_cols = ['Age', 'Income', 'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism', 'PhishingConfidence', 'GutFeelings', 'SimpleRules', 'CarefulAnalysis', 'ConsiderFactors']
    one_hot_cols = ['Gender', 'Education', 'JobCategory', 'SocialMedia', 'Shopping', 'Videos', 'Gaming', 'Learning', 'Communicating', 'PhishingExperience', 'CheckSender', 'DetectionSystem']
    
    # Transform the data using the loaded encoders
    ordinal_encoded = pd.DataFrame(ordinal_encoder.transform(df[ordinal_cols]), columns=ordinal_cols)
    one_hot_encoded = pd.DataFrame(one_hot_encoder.transform(df[one_hot_cols]), columns=one_hot_encoder.get_feature_names_out(one_hot_cols))
    
    df['PhishingTypes'] = df['PhishingTypes'].astype(str).str.split(';')
    df['SuspiciousActions'] = df['SuspiciousActions'].astype(str).str.split(';')
    
    phishing_types_encoded = pd.DataFrame(mlb_phishing_types.transform(df['PhishingTypes']), columns=mlb_phishing_types.classes_, index=df.index)
    suspicious_actions_encoded = pd.DataFrame(mlb_suspicious_actions.transform(df['SuspiciousActions']), columns=mlb_suspicious_actions.classes_, index=df.index)
    
    return ordinal_encoded, one_hot_encoded, phishing_types_encoded, suspicious_actions_encoded

# Function to predict phishing victim
def predict_phishing_victim(user_input):
    try:
        user_df = pd.DataFrame([user_input])
        ordinal_encoded, one_hot_encoded, phishing_types_encoded, suspicious_actions_encoded = encode_categorical_data(user_df)
        user_encoded_features = pd.concat([ordinal_encoded, one_hot_encoded, phishing_types_encoded, suspicious_actions_encoded], axis=1)
        
        # Ensure all expected columns are present
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in user_encoded_features.columns:
                user_encoded_features[col] = 0
        
        user_encoded_features = user_encoded_features[expected_columns]
        if user_encoded_features.shape[0] == 0:
            raise ValueError("No valid samples to predict.")
        
        prediction = model.predict(user_encoded_features)
        return 'Phishing Victim' if prediction[0] else 'Not a Phishing Victim'
    except Exception as e:
        st.error(f"Error in prediction: {e}")
        return "Error"

# Main function to create Streamlit app
def main():
    st.title("Phishing Susceptibility Prediction")
    st.write("Please fill in the details below to predict if you are a potential phishing victim.")

    user_input = {
        'Age': st.selectbox('Age', ['15-25', '25-35', '35-45', '45-60', '60 above'], help="Select your age range."),
        'Income': st.selectbox('Income', ['20k - 50k', '50k - 100k', '100k - 200k', '200k above'], help="Select your income range."),
        'Openness': st.selectbox('Openness', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Do you enjoy trying new things and exploring new ideas?"),
        'Conscientiousness': st.selectbox('Conscientiousness', ['Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Are you organized and responsible?"),
        'Extraversion': st.selectbox('Extraversion', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Are you outgoing and social?"),
        'Agreeableness': st.selectbox('Agreeableness', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Are you friendly and kind towards others?"),
        'Neuroticism': st.selectbox('Neuroticism', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Do you often feel negative emotions like worry or stress?"),
        'PhishingConfidence': st.selectbox('PhishingConfidence', ['Extremely not confident', 'Somewhat confident', 'Neutral','Very Confident', 'Extremely confident'], help="How confident are you in identifying phishing attempts?"),
        'GutFeelings': st.selectbox('GutFeelings', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Do you rely on gut feelings to detect phishing?"),
        'SimpleRules': st.selectbox('SimpleRules', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Do you follow simple rules to avoid phishing?"),
        'CarefulAnalysis': st.selectbox('CarefulAnalysis', ['Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Do you carefully analyze emails to detect phishing?"),
        'ConsiderFactors': st.selectbox('ConsiderFactors', ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], help="Do you consider multiple factors before acting on an email?"),
        'PhishingTypes': st.multiselect('PhishingTypes', phishing_types_options, help="Select all types of phishing messages you are aware of."),
        'SuspiciousActions': st.multiselect('SuspiciousActions', suspicious_actions_options, help="Select all actions you would take upon receiving a suspicious message."),
        'Gender': st.selectbox('Gender', gender_options, help="Select your gender."),
        'Education': st.selectbox('Education', education_options, help="Select your education level."),
        'JobCategory': st.selectbox('JobCategory', job_category_options, help="Select your job category."),
        'SocialMedia': st.selectbox('SocialMedia', social_media_options, help="How often do you use social media?"),
        'Shopping': st.selectbox('Shopping', shopping_options, help="How often do you shop online?"),
        'Videos': st.selectbox('Videos', videos_options, help="How often do you watch videos online?"),
        'Gaming': st.selectbox('Gaming', gaming_options, help="How often do you play online games?"),
        'Learning': st.selectbox('Learning', learning_options, help="How often do you engage in online learning activities?"),
        'Communicating': st.selectbox('Communicating', communicating_options, help="How often do you communicate online?"),
        'PhishingExperience': st.selectbox('PhishingExperience', phishing_experience_options, help="Have you experienced phishing before?"),
        'CheckSender': st.selectbox('CheckSender', check_sender_options, help="How often do you check the sender of an email?"),
        'DetectionSystem': st.selectbox('DetectionSystem', detection_system_options, help="Do you agree that having a phishing detection system is important?")
    }

    if st.button('Predict'):
        prediction = predict_phishing_victim(user_input)
        st.write(f"**Prediction:** {prediction}")

if __name__ == '__main__':
    main()

