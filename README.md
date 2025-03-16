# 🏥 Smart Disease Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live--Demo-brightgreen)](https://medicspy-ffqnex9wk5zu6xyp68ltan.streamlit.app/)

## 📌 Overview
**Smart Disease Prediction** is an **AI-powered** web application designed to assist users in identifying potential diseases based on their symptoms. By leveraging **Machine Learning**, the app analyzes user inputs and provides **accurate disease predictions**, along with recommendations on **precautions, medications, diet, and workouts**.

This project is aimed at making **healthcare more accessible** by providing users with a **quick, intelligent, and interactive** diagnostic experience.

## ✨ Features
- 🌡️ **Symptom-based disease prediction** using a trained **ML model**.
- 🎨 **User-friendly & colorful UI** designed for an engaging experience.
- 🔍 **Flexible symptom input** (users can enter varying numbers of symptoms).
- 🏥 **Detailed disease insights**, including symptoms, causes, and recommendations.
- 📊 **Scalable real-time predictions** for users of all backgrounds.
- 🚀 **Fast & lightweight** deployment on **Streamlit**.
- 🔗 **Mobile & desktop-friendly** interface for easy accessibility.
- 📚 **Data-driven insights** powered by an **advanced ML model**.

## 🚀 Live Demo
You can access the **Smart Disease Prediction** app by clicking below:
👉 [Smart Disease Prediction](https://medicspy-ffqnex9wk5zu6xyp68ltan.streamlit.app/)

## 🛠️ Tech Stack
| Component  | Technology |
|------------|------------|
| **Frontend** | Streamlit (Python-based UI framework) |
| **Backend** | Flask (for API handling) |
| **Machine Learning Model** | Support Vector Classifier (SVC) |
| **Database** | CSV-based dataset (for training) |
| **Deployment** | Streamlit Cloud |
| **Model & Scaler** | `svc.pkl`, `scaler.pkl` |

## 📂 Project Structure
```
medicspy/
│── app.py                # Streamlit frontend (UI & user interactions)
│── main.py               # Flask backend (ML model API)
│── svc.pkl               # Trained Machine Learning model
│── scaler.pkl            # Data scaler for preprocessing
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
│── dataset.csv           # Dataset used for training (if applicable)
```

## 📥 Installation & Setup
To run the project locally, follow these simple steps:

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/medicspy.git
cd medicspy
```

### 2️⃣ Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate     # On Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

Now, open the **local URL** provided in the terminal to interact with the app.

## 🎯 How It Works
1️⃣ Users enter symptoms they are experiencing in the input field.


2️⃣ The **ML model** processes the symptoms and predicts the possible disease.

3️⃣ The app displays **disease details, precautions, medications, and lifestyle recommendations**.

4️⃣ Users can adjust their inputs to explore different predictions.

## 🏗️ Future Enhancements

🔹 Improve model accuracy with more diverse datasets.

🔹 Integrate **Natural Language Processing (NLP)** for symptom recognition.

🔹 Implement **voice-based symptom input** for better accessibility.

🔹 Add **user authentication** for a personalized experience.

🔹 Expand the **database** with more diseases and treatments.

##  Contributing
We welcome contributions to improve this project! To contribute:

1️⃣ Fork the repository.

2️⃣ Create a new branch (`git checkout -b feature-branch`).

3️⃣ Make your changes and commit (`git commit -m "Added a new feature"`).

4️⃣ Push your branch (`git push origin feature-branch`).

5️⃣ Open a Pull Request.

## 💡 Acknowledgments
This project was inspired by the idea of leveraging **AI & Machine Learning** to create **accessible healthcare solutions** for everyone. 

## 📜 License
This project is licensed under the **MIT License**.

---

⭐ **If you like this project, give it a star on GitHub!** ⭐

📩 **For queries, feature requests, or collaborations, feel free to reach out!** 🚀

