import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("svc.pkl", "rb"))

# Symptom and Disease Mapping
symptom_list = [
    "Fever", "Cough", "Fatigue", "Headache", "Sore Throat", "Runny Nose", "Body Aches", "Shortness of Breath",
    "Chest Pain", "Chills", "Nausea", "Vomiting", "Diarrhea", "Dizziness", "Sweating", "Loss of Appetite",
    "Joint Pain", "Skin Rash", "Blurred Vision", "Frequent Urination", "Weight Loss", "Insomnia", "Chest Tightness",
    "Palpitations", "Weakness", "Irritability", "Anxiety", "Depression", "Hair Loss", "Dry Skin", "Hearing Loss",
    "Loss of Smell", "Loss of Taste", "Sensitivity to Light", "Numbness", "Tingling", "Difficulty Swallowing",
    "Hoarseness", "Swollen Lymph Nodes", "Chronic Pain", "Seizures", "Memory Loss", "Confusion", "Hallucinations",
    "Swollen Joints", "Cold Hands and Feet", "Excessive Sweating", "Difficulty Breathing", "Nasal Congestion",
    "Chronic Cough", "Wheezing", "Blood in Urine", "Dark Urine", "Pale Stools", "Jaundice", "Bloating",
    "Constipation", "Heartburn", "Difficulty Walking", "Muscle Weakness", "Balance Problems", "Coordination Issues",
    "Tremors", "Slurred Speech", "Chronic Fatigue", "Night Sweats", "Mouth Ulcers", "Bruising Easily",
    "Slow Healing Wounds", "Excessive Thirst", "Cold Intolerance", "Heat Intolerance", "Rapid Heartbeat",
    "Unexplained Weight Gain", "Frequent Infections", "Increased Hunger", "Difficulty Concentrating",
    "Loss of Consciousness", "Sudden Mood Changes", "Bleeding Gums", "Excessive Gas", "Acid Reflux",
    "Dry Eyes", "Ear Pain", "Frequent Headaches", "Vision Changes", "Neck Stiffness", "Leg Cramps",
    "Unexplained Swelling", "Abdominal Pain", "Blood in Stool", "Nail Changes", "Itchy Skin", "Hives",
    "Chronic Sinus Infections", "Frequent Sneezing", "Lightheadedness", "Cold Sensitivity"
]

disease_list = [
    "Common Cold", "Flu", "COVID-19", "Pneumonia", "Bronchitis", "Asthma", "Gastroenteritis", "Migraine", 
    "Hypertension", "Diabetes", "Tuberculosis", "Malaria", "Typhoid", "Sinusitis", "Dengue", "Anemia",
    "Hepatitis", "Chronic Kidney Disease", "Lupus", "Parkinson's", "Alzheimer's", "Epilepsy", "Multiple Sclerosis",
    "Stroke", "Heart Disease", "COPD", "Liver Cirrhosis", "Irritable Bowel Syndrome", "Celiac Disease",
    "Thyroid Disorders", "Anxiety Disorder", "Depression", "Schizophrenia", "Psoriasis", "Eczema", "Osteoarthritis",
    "Rheumatoid Arthritis", "Gout", "Pancreatitis", "Glaucoma", "Macular Degeneration", "Sleep Apnea",
    "HIV/AIDS", "Polycystic Ovary Syndrome", "Endometriosis", "Prostate Cancer", "Breast Cancer", "Leukemia",
    "Lymphoma", "Skin Cancer", "Colon Cancer", "Esophageal Cancer", "Parkinson's Disease", "Huntington's Disease"
]

# Ensure model compatibility
def predict_disease(selected_symptoms):
    symptoms_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    symptoms_vector += [0] * (132 - len(symptoms_vector))  # Ensuring vector length is 132
    prediction_index = model.predict([symptoms_vector])[0]
    return disease_list[prediction_index] if prediction_index < len(disease_list) else "Unknown Disease"

# Streamlit UI Setup
st.set_page_config(page_title="Smart Disease Prediction", page_icon="🩺", layout="wide")

# CSS for Styling
st.markdown(
    """
    <style>
    body { background-color: #C8E6C9; color: black; }
    .main { background-color: #C8E6C9; color: black; }
    .stButton>button { background-color: #1B5E20; color: white; border-radius: 10px; font-size: 16px; padding: 10px; margin: 5px; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
page = st.sidebar.radio("Navigation", ["Home", "About", "Symptoms Guide", "Contact", "FAQ", "Health Tips", "Emergency"])

# Home Page
if page == "Home":
    st.title("SMART MEDICL DIAGONOSIS AI")
    st.subheader("Enter your symptoms below to get a diagnosis.")

    # Symptom Selection and Input
    selected_symptoms = st.multiselect("Choose your symptoms", symptom_list)
    typed_symptoms = st.text_input("Or type your symptoms, separated by commas")
    
    if st.button("Predict Disease", key="predict"):
        all_symptoms = selected_symptoms + [sym.strip() for sym in typed_symptoms.split(",") if sym.strip()]
        if len(all_symptoms) >= 2:
            prediction = predict_disease(all_symptoms)
            st.success(f"🩺 Predicted Disease: **{prediction}**")
        else:
            st.error("Please enter at least 2 symptoms.")

# Other Pages Remain the Same
elif page == "About":
    st.title("🩺 About Smart Disease Prediction")
    st.write("""
    Our AI-powered system predicts potential diseases based on symptoms entered by users.
    It is designed to assist in **early diagnosis** and **health awareness**.
    """)

elif page == "Symptoms Guide":
    st.title("📌 Symptoms Guide")
    st.write("""
    - **Common Cold**: Runny nose, sneezing, mild fever  
    - **Flu**: High fever, severe body aches, dry cough  
    - **COVID-19**: Fever, dry cough, loss of taste/smell  
    - **Diabetes**: Frequent urination, thirst, fatigue  
    """)

elif page == "Contact":
    st.title("📞 Contact Us")
    st.write("""
    **Email**: support@smarthealth.com  
    **Phone**: +1 800-HEALTH  
    **Location**: Health Tech Center, NY  
    """)

elif page == "FAQ":
    st.title("❓ Frequently Asked Questions")
    st.write("""
    - **Is this tool a replacement for doctors?**  
      No, this is an AI-powered assistant to guide you, but professional medical consultation is recommended.
    
    - **How accurate is the prediction?**  
      The accuracy depends on the symptoms entered by the user.
    """)

elif page == "Health Tips":
    st.title("💡 Health & Wellness Tips")
    st.write("""
    - Stay **hydrated** and eat a **balanced diet**.  
    - Exercise **regularly** and get **enough sleep**.  
    - Avoid **processed foods** and **smoking**.  
    """)

elif page == "Emergency":
    st.title("@ Emergency contacts")
    st.write("""
    - INDIA - 108 
    - UK - 911  
    - SINGAPORE - 888
    """)