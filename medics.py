import streamlit as st
import numpy as np
import pickle 
import sklearn
print("scikit-learn version:", sklearn.__version__)

# Load models
model = pickle.load(open("svc.pkl", "rb"))
bp_model = pickle.load(open("bp_model.pkl", "rb"))
lung_model = pickle.load(open("lung_cancer_model.pkl", "rb"))
heart_model = pickle.load(open("heart_disease_model.pkl", "rb"))
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))
parkinsons_model = pickle.load(open("parkinsons_model.pkl", "rb"))

# Symptom and Disease Mapping (Extended with 100+ symptoms & diseases)
symptom_list = [
    "Fever", "Cough", "Fatigue", "Headache", "Sore Throat", "Runny Nose", "Nausea", "Vomiting", "Diarrhea", 
    "Shortness of Breath", "Chest Pain", "Dizziness", "Skin Rash", "Muscle Weakness", "Joint Pain", "Swelling", 
    "Loss of Appetite", "Unexplained Weight Loss", "Blurred Vision", "Memory Loss", "Seizures", "Difficulty Swallowing", 
    "Frequent Urination", "Night Sweats", "Tingling in Hands/Feet", "Excessive Thirst", "Hair Loss", "Abdominal Pain", 
    "Depression", "Anxiety", "Sleep Disturbances", "Nasal Congestion", "Sneezing", "Ear Pain", "Hearing Loss", 
    "Palpitations", "Cold Hands and Feet", "Excessive Sweating", "Yellowing of Skin (Jaundice)", "Difficulty Breathing", 
    "Chest Tightness", "Cyanosis (Blue Skin)", "Chronic Cough", "Loss of Smell", "Loss of Taste", "Hoarseness", 
    "Heartburn", "Difficulty Urinating", "Blood in Urine", "Dark Urine", "Lower Back Pain", "Muscle Cramps", 
    "Numbness", "Dry Skin", "Excessive Itching", "Changes in Nail Color", "Skin Peeling", "Sudden Weakness", 
    "Uncontrolled Movements", "Shakiness", "Difficulty Speaking", "Fainting", "Confusion", "Sudden Aggression", 
    "Mood Swings", "Hallucinations", "Difficulty Walking", "Bone Pain", "Chest Discomfort", "Leg Swelling", 
    "Difficulty Concentrating", "Lethargy", "Cold Intolerance", "Heat Intolerance", "Irregular Menstrual Cycles", 
    "Low Blood Pressure", "High Blood Pressure", "Dry Mouth", "Excessive Salivation", "Red Eyes", "Eye Pain", 
    "Sensitivity to Light", "Blood in Stool", "Constipation", "Persistent Hiccups", "Frequent Infections", 
    "Slow Healing Wounds", "Sudden Weight Gain", "Extreme Hunger", "Bruising Easily", "Tremors", "Slurred Speech", 
    "Drooling", "Unsteady Gait", "Reduced Handwriting Size", "Daytime Sleepiness", "Erectile Dysfunction"
]

disease_list = [
    "Common Cold", "Flu", "COVID-19", "Pneumonia", "Bronchitis", "Diabetes", "Hypertension", "Heart Disease", 
    "Stroke", "Asthma", "Lung Cancer", "Kidney Disease", "Liver Disease", "Dementia", "Parkinson's", "Tuberculosis", 
    "Anemia", "Migraine", "Hyperthyroidism", "Hypothyroidism", "Arthritis", "Osteoporosis", "Epilepsy", 
    "Multiple Sclerosis", "Alzheimer's Disease", "Depression", "Anxiety Disorder", "Bipolar Disorder", 
    "Schizophrenia", "Chronic Kidney Disease", "Cirrhosis", "Fatty Liver Disease", "Pancreatitis", 
    "Peptic Ulcer Disease", "Gastroesophageal Reflux Disease (GERD)", "Irritable Bowel Syndrome (IBS)", 
    "Ulcerative Colitis", "Crohn’s Disease", "Hepatitis", "HIV/AIDS", "Malaria", "Dengue Fever", "Zika Virus", 
    "Chikungunya", "Lyme Disease", "Psoriasis", "Eczema", "Lupus", "Gout", "Deep Vein Thrombosis", 
    "Varicose Veins", "Peripheral Artery Disease", "COPD", "Sleep Apnea", "Meningitis", "Encephalitis", 
    "Polio", "Measles", "Rubella", "Chickenpox", "Shingles", "Tetanus", "Whooping Cough", "Scarlet Fever", 
    "Leukemia", "Lymphoma", "Melanoma", "Breast Cancer", "Prostate Cancer", "Colorectal Cancer", "Pancreatic Cancer", 
    "Bladder Cancer", "Ovarian Cancer", "Cervical Cancer", "Endometriosis", "Polycystic Ovary Syndrome (PCOS)", 
    "Infertility", "Metabolic Syndrome", "Celiac Disease", "Hemophilia", "Sickle Cell Disease", "Thalassemia", 
    "Aplastic Anemia", "Huntington’s Disease", "ALS (Lou Gehrig’s Disease)", "Autoimmune Disorders", 
    "Raynaud’s Disease", "Fibromyalgia", "Chronic Fatigue Syndrome", "Interstitial Lung Disease", 
    "Sarcoidosis", "Tonsillitis", "Bronchiolitis", "Rhinitis", "Pharyngitis", "Sinusitis", "Endocarditis"
]


def predict_disease(selected_symptoms):
    symptoms_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
    symptoms_vector += [0] * (132 - len(symptoms_vector))
    prediction_index = model.predict([symptoms_vector])[0]
    return disease_list[prediction_index] if prediction_index < len(disease_list) else "Unknown Disease"

# Streamlit UI Setup
st.set_page_config(page_title="Smart Disease Prediction", page_icon="🩺", layout="wide")

# Custom CSS for Styling
st.markdown(
    """
    <style>
    body { background-color: white; color: black; }
    .main { background-color: white; color: black; }
    .stButton>button { 
        background-color: #007BFF; 
        color: white; 
        border-radius: 10px; 
        font-size: 16px; 
        padding: 10px; 
        margin: 5px; 
        border: none; 
        transition: 0.3s; 
    }
    .stButton>button:hover {
        background-color: #0056b3; /* Darker blue on hover */
    }
    [data-testid="stSidebar"] {
        background-color: #0077B6;
        color: white;
    }
    [data-testid="stSidebar"] .stButton>button {
        background-color: white; 
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("🎯 Check-Up Corner")
page = st.sidebar.radio("Select Your Agenda", ["About", "General Disease Prediction", "Smart Doctor", "Symptoms Guide", "Contact", "FAQ", "Health Tips", "Emergency"])

# General Disease Prediction Submenu
if page == "General Disease Prediction":
    st.sidebar.subheader("Choose a Disease Prediction:")
    prediction_type = st.sidebar.radio("Select", ["BP Diagnosis", "Lung Cancer", "Heart Disease", "Diabetes", "Parkinson's Disease"])
    
    if prediction_type == "BP Diagnosis":
        st.title("🩸 Blood Pressure Diagnosis")
        age = st.number_input("Age", min_value=1)
        bmi = st.number_input("BMI")
        smoking = st.selectbox("Do you smoke?", ["Yes", "No"])
        hemoglobin = st.number_input("Level of Hemoglobin")
        pedigree = st.number_input("Genetic Pedigree Coefficient")
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
        pregnancy = st.selectbox("Pregnancy (1=Yes, 0=No)", [1, 0])
        physical_activity = st.number_input("Physical Activity")
        salt_diet = st.number_input("Salt Content in Diet")
        alcohol = st.number_input("Alcohol Consumption per Day")
        stress = st.number_input("Level of Stress")
        kidney_disease = st.selectbox("Chronic Kidney Disease (1=Yes, 0=No)", [1, 0])
        adrenal_disorders = st.selectbox("Adrenal & Thyroid Disorders (1=Yes, 0=No)", [1, 0])
        if st.button("Predict BP Abnormality"):
            input_data = np.array([[age, bmi, 1 if smoking == "Yes" else 0, hemoglobin, pedigree, sex, pregnancy, physical_activity, salt_diet, alcohol, stress, kidney_disease, adrenal_disorders]])
            prediction = bp_model.predict(input_data)
            st.success(f"BP Diagnosis Result: {'Abnormal' if prediction[0] else 'Normal'}")

    

    elif prediction_type == "Lung Cancer":
        st.title("🌬️ Lung Cancer Prediction")

        age = st.number_input("Age", min_value=1)
        cigarettes_per_day = st.number_input("Cigarettes per Day", min_value=0)
        air_quality = st.slider("Air Quality (1-10)", min_value=1, max_value=10)
        alcohol_consumption = st.slider("Alcohol Consumption (1-10)", min_value=1, max_value=10)

        if st.button("Predict Lung Cancer Risk"):
            if lung_model is None:
                st.error("Lung Cancer model is not loaded. Please check your model file.")
            else:
                input_data = np.array([[age, cigarettes_per_day, air_quality, alcohol_consumption]])
                prediction = lung_model.predict(input_data)
                st.success(f"Prediction: {'Lung Cancer Detected' if prediction[0] else 'No Lung Cancer Detected'}")




    
    elif prediction_type == "Heart Disease":
        st.title("❤️ Heart Disease Prediction")
        age = st.number_input("Age")
        sex = st.selectbox("Sex (1=Male, 0=Female)", [1, 0])
        chest_pain = st.slider("Chest Pain Type (0-3)", 0, 3)
        bp = st.number_input("Resting Blood Pressure")
        chol = st.number_input("Cholesterol Level")
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
        rest_ecg = st.slider("Resting ECG (0-2)", 0, 2)
        max_hr = st.number_input("Max Heart Rate Achieved")
        angina = st.selectbox("Exercise-Induced Angina (1=Yes, 0=No)", [1, 0])
        st_depression = st.number_input("ST Depression")
        st_slope = st.slider("Slope of ST Segment (0-2)", 0, 2)
        major_vessels = st.slider("Number of Major Vessels (0-3)", 0, 3)
        thalassemia = st.slider("Thalassemia (0-3)", 0, 3)
        if st.button("Predict Heart Disease"):
            input_data = np.array([[age, sex, chest_pain, bp, chol, fbs, rest_ecg, max_hr, angina, st_depression, st_slope, major_vessels, thalassemia]])
            prediction = heart_model.predict(input_data)
            st.success(f"Heart Disease Risk: {'High' if prediction[0] else 'Low'}")

    elif prediction_type == "Diabetes":
        st.title("🍬 Diabetes Prediction")
        pregnancies = st.number_input("Pregnancies (0-17)", min_value=0, max_value=17)
        glucose = st.number_input("Glucose Level (0-200)", min_value=0, max_value=200)
        blood_pressure = st.number_input("Blood Pressure (0-122)", min_value=0, max_value=122)
        skin_thickness = st.number_input("Skin Thickness (0-100)", min_value=0, max_value=100)
        insulin = st.number_input("Insulin Level (0-846)", min_value=0, max_value=846)
        bmi = st.number_input("BMI (0.0-67.0)", min_value=0.0, max_value=67.0)
        diabetes_pedigree = st.number_input("Diabetes Pedigree Function (0.0-2.4)", min_value=0.0, max_value=2.4)
        age = st.number_input("Age (21-88)", min_value=21, max_value=88)
        
        if st.button("Predict Diabetes"):
            input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]])
            prediction = diabetes_model.predict(input_data)
            st.success(f"Diabetes Prediction: {'Positive' if prediction[0] else 'Negative'}")

    
    elif prediction_type == "Parkinson's Disease":
        st.title("🧠 Parkinson's Disease Prediction")
        mdvp_fo = st.number_input("Enter MDVP:Fo(Hz)")
        mdvp_fhi = st.number_input("Enter MDVP:Fhi(Hz)")
        mdvp_flo = st.number_input("Enter MDVP:Flo(Hz)")
        mdvp_jitter_perc = st.number_input("Enter MDVP:Jitter(%)")
        mdvp_jitter_abs = st.number_input("Enter MDVP:Jitter(Abs)")
        mdvp_rap = st.number_input("Enter MDVP:RAP")
        mdvp_ppq = st.number_input("Enter MDVP:PPQ")
        jitter_ddp = st.number_input("Enter Jitter:DDP")
        mdvp_shimmer = st.number_input("Enter MDVP:Shimmer")
        mdvp_shimmer_db = st.number_input("Enter MDVP:Shimmer(dB)")
        shimmer_apq3 = st.number_input("Enter Shimmer:APQ3")
        shimmer_apq5 = st.number_input("Enter Shimmer:APQ5")
        mdvp_apq = st.number_input("Enter MDVP:APQ")
        shimmer_dda = st.number_input("Enter Shimmer:DDA")
        nhr = st.number_input("Enter NHR")
        hnr = st.number_input("Enter HNR")
        rpde = st.number_input("Enter RPDE")
        dfa = st.number_input("Enter DFA")
        spread1 = st.number_input("Enter spread1")
        spread2 = st.number_input("Enter spread2")
        d2 = st.number_input("Enter D2")
        ppe = st.number_input("Enter PPE")

        if st.button("Predict Parkinson's Disease"):
            input_data = np.array([[mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter_perc, mdvp_jitter_abs, mdvp_rap, mdvp_ppq,
                                    jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, shimmer_apq3, shimmer_apq5, mdvp_apq,
                                    shimmer_dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])
            prediction = parkinsons_model.predict(input_data)
            st.success(f"Parkinson's Disease Prediction: {'Positive' if prediction[0] else 'Negative'}")

# Home Page
elif page == "Smart Doctor":
    st.title("SMART MEDICAL DIAGNOSIS AI")
    selected_symptoms = st.multiselect("Choose your symptoms", symptom_list)
    typed_symptoms = st.text_input("Or type symptoms, separated by commas")
    if st.button("Predict Disease"):
        all_symptoms = selected_symptoms + [sym.strip() for sym in typed_symptoms.split(",") if sym.strip()]
        if len(all_symptoms) >= 2:
            prediction = predict_disease(all_symptoms)
            st.success(f"🩺 Predicted Disease: **{prediction}**")
        else:
            st.error("Please enter at least 2 symptoms.")

# Other Pages (Same as before)
elif page == "About":
    st.title("🩺 About Smart Disease Prediction")
    st.write("🌟 Smart Disease Prediction – Your AI Health Companion! 🏥🤖.")
    st.write("")
    st.write("Welcome to Smart Disease Prediction, your intelligent AI-powered health assistant!")


    st.write("🚀 This cutting-edge platform uses advanced machine learning models to analyze your symptoms and provide instant predictions for various diseases. "
    "Whether it’s heart disease ❤️, diabetes 🍬, lung cancer 🌬️, or blood pressure abnormalities 🩸, our system delivers fast, reliable, and personalized health insights.")


    st.write("With a sleek, colorful UI 🎨, real-time analysis, and easy-to-use interface, health monitoring has never been this seamless! 🏆 Plus, it provides tailored recommendations on precautions, medications, diet 🥗, and workouts 🏋️ to help you stay healthy."
    "🔮 Your health, our priority! Try Smart Disease Prediction today and take control of your well-being with just a few clicks! 💙✨")

elif page == "Symptoms Guide":
    st.title("📌 Symptoms Guide")
    st.write(" Know Your Health Signs! 🏥🔍")


    st.write("Understanding your symptoms is the first step toward better health! 🚑 Our Smart Disease Prediction system analyzes your inputs and detects potential health risks in real-time. 🧠💡")


    st.write("🔎 Common Symptoms & Possible Conditions:")
    st.write("✔ Persistent Cough & Chest Pain – Could indicate Lung Cancer 🌬️ or Respiratory Issues 🫁")
    st.write("✔ High Fever & Fatigue – May signal Infections 🤒 or Immune Disorders 🦠")
    st.write("✔ Frequent Thirst & Unexplained Weight Loss – Signs of Diabetes 🍬")
    st.write("✔ Irregular Heartbeat & Dizziness – Could be linked to Heart Disease ❤️ or Blood Pressure Issues 🩸")


    st.write("💡 How It Works?")
    st.write("1️⃣ Enter your symptoms in our AI-powered system.")
    st.write("2️⃣ Get instant insights on possible health conditions.")
    st.write("3️⃣ Receive expert-backed recommendations on precautions, diet, and medications!")


    st.write("⚠️ Note: Our tool provides guidance but is not a replacement for professional medical advice. Always consult a doctor for accurate diagnosis! 🩺👨‍⚕️")


    st.write("Stay informed, stay healthy! 🌟💙")
elif page == "Contact":
    st.title("📞 Contact Us")
    st.write("📞 Phone: +91 XXXXX XXXXX")
    st.write("📧 Email: support@smartdiseaseprediction.com")
    st.write("🌐 Website: www.smartdiseaseprediction.com")
    st.write("📍 Address: Tamil Nadu, Coimbatore")
    st.write("💬 Social Media:Instagram: @smartdiseasepredict OR Twitter/X: @smartdisease_ai OR LinkedIn: Smart Disease Prediction")

 

elif page == "FAQ":
    st.title("❓ Frequently Asked Questions")

    faq_list = [
        ("What is Smart Disease Prediction?", 
         "Smart Disease Prediction is an AI-powered platform that analyzes symptoms to predict possible diseases and provide personalized health insights."),
        
        ("How does the disease prediction system work?", 
         "Our system uses Machine Learning to analyze your symptoms and compare them with medical data to predict potential health conditions."),
        
        ("What symptoms can I check on this platform?", 
         "You can enter various symptoms, such as fever, cough, fatigue, headache, chest pain, and more, to get an AI-powered diagnosis."),
        
        ("Is the prediction accurate?", 
         "While our AI model provides high accuracy, it is not a replacement for professional medical consultation. We recommend consulting a doctor for confirmation."),
        
        ("Can I check for multiple symptoms at once?", 
         "Yes! You can enter multiple symptoms, and our system will analyze them to provide the best possible predictions."),
        
        ("Is this service free to use?", 
         "Yes, our basic prediction services are free. However, we may introduce premium features in the future for advanced insights."),
        
        ("Do I need to create an account to use the service?", 
         "No, you can use the symptom checker without signing up. However, creating an account allows you to save your health history and get personalized recommendations."),
        
        ("How can I contact customer support?", 
         "You can reach us via email at **support@smartdiseaseprediction.com** or call us at **+91 98765 43210**."),
        
        ("Is my data safe and secure?", 
         "Yes! We prioritize user privacy and follow strict security protocols to ensure that your health data remains confidential."),
        
        ("Can I use this tool on my mobile?", 
         "Absolutely! Our platform is mobile-friendly, and we are also working on launching a dedicated app soon.")
    ]

    for question, answer in faq_list:
        with st.expander(f"❓ {question}"):
            st.write(answer)


elif page == "Health Tips":
    st.title("💡 Health & Wellness Tips")

    health_tips = [
        ("🥗 Eat a Balanced Diet", 
         "Include a variety of fruits, vegetables, whole grains, and lean proteins in your meals to stay healthy and energized."),
        
        ("💧 Stay Hydrated", 
         "Drink at least 8 glasses of water daily to keep your body hydrated and support overall well-being."),
        
        ("🏋️‍♂️ Exercise Regularly", 
         "Engage in physical activities like walking, jogging, or yoga for at least 30 minutes a day to maintain a healthy body and mind."),
        
        ("😴 Get Enough Sleep", 
         "Aim for 7-9 hours of quality sleep every night to improve concentration, mood, and overall health."),
        
        ("🧘‍♀️ Manage Stress", 
         "Practice meditation, deep breathing, or engage in hobbies to reduce stress and maintain emotional well-being."),
        
        ("🚭 Avoid Smoking & Alcohol", 
         "Reduce or eliminate tobacco and alcohol consumption to lower the risk of chronic diseases."),
        
        ("🦷 Maintain Good Hygiene", 
         "Brush and floss daily, wash your hands frequently, and maintain overall hygiene to prevent infections."),
        
        ("👨‍⚕️ Regular Health Check-ups", 
         "Visit your doctor for routine check-ups to detect any health issues early and take necessary precautions."),
        
        ("🌞 Get Sunlight & Fresh Air", 
         "Spend some time outdoors to absorb vitamin D and improve your mood naturally."),
        
        ("📱 Limit Screen Time", 
         "Take breaks from screens to reduce eye strain and improve focus and productivity."),
    ]

    for tip, description in health_tips:
        with st.expander(f"{tip}"):
            st.write(description)

elif page == "Emergency":
    st.title("🚨 Emergency Contacts")

    st.write("In case of an emergency, quickly dial the appropriate number based on your location. Here are the emergency contact numbers for different countries:")

    emergency_numbers = {
        "🇺🇸 United States": "911",
        "🇨🇦 Canada": "911",
        "🇬🇧 United Kingdom": "999 / 112",
        "🇦🇺 Australia": "000",
        "🇮🇳 India": "112",
        "🇫🇷 France": "112 / 15 (Medical)",
        "🇩🇪 Germany": "112",
        "🇪🇸 Spain": "112",
        "🇮🇹 Italy": "112 / 118 (Medical)",
        "🇧🇷 Brazil": "190 / 192 (Medical)",
        "🇯🇵 Japan": "110 (Police) / 119 (Fire & Ambulance)",
        "🇨🇳 China": "110 (Police) / 120 (Medical) / 119 (Fire)",
        "🇷🇺 Russia": "112",
        "🇿🇦 South Africa": "10111 (Police) / 10177 (Ambulance)",
        "🇲🇽 Mexico": "911",
        "🇦🇪 UAE": "999 (Police) / 998 (Ambulance)",
        "🇸🇦 Saudi Arabia": "999 (Police) / 997 (Medical)",
        "🇰🇷 South Korea": "112 (Police) / 119 (Fire & Ambulance)",
        "🇹🇷 Turkey": "112",
        "🇮🇩 Indonesia": "112",
        "🇦🇷 Argentina": "911 / 107 (Medical)",
        "🇳🇿 New Zealand": "111",
        "🇸🇬 Singapore": "999 (Police) / 995 (Medical & Fire)",
        "🇲🇾 Malaysia": "999 / 112",
        "🇹🇭 Thailand": "191 (Police) / 1669 (Medical)",
        "🇵🇭 Philippines": "911",
        "🇻🇳 Vietnam": "113 (Police) / 115 (Medical)",
        "🇳🇬 Nigeria": "112 / 767",
        "🇵🇰 Pakistan": "15 (Police) / 115 (Medical)",
        "🇧🇩 Bangladesh": "999",
        "🇳🇴 Norway": "112 / 113 (Medical)",
        "🇸🇪 Sweden": "112",
        "🇩🇰 Denmark": "112",
        "🇫🇮 Finland": "112",
        "🇮🇪 Ireland": "112 / 999",
        "🇵🇱 Poland": "112",
        "🇺🇦 Ukraine": "112",
        "🇳🇱 Netherlands": "112",
        "🇨🇭 Switzerland": "112 / 144 (Medical)",
        "🇵🇹 Portugal": "112",
        "🇨🇿 Czech Republic": "112 / 155 (Medical)",
        "🇬🇷 Greece": "112 / 166 (Medical)",
        "🇭🇺 Hungary": "112 / 104 (Medical)",
        "🇧🇪 Belgium": "112",
        "🇦🇹 Austria": "112",
        "🇮🇸 Iceland": "112",
        "🇨🇴 Colombia": "123",
        "🇨🇱 Chile": "133 (Police) / 131 (Medical)",
    }

    for country, number in emergency_numbers.items():
        with st.expander(f"{country}"):
            st.write(f"📞 **Emergency Number:** {number}")
