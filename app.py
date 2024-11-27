import streamlit as st
import pickle
import fitz  # PyMuPDF for PDF parsing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK data if not already available
nltk.download("stopwords")
nltk.download("wordnet")

# Load models and TFIDF vectorizer into memory
model_paths = {
    "Model 1 (Classification): Logistic Regression": "models/logistic_regression_model.pkl",
    # Add other models if needed
}
# Path to the TFIDF vectorizer
tfidf_vectorizer_path = "models/tfidf_vectorizer.pkl"

# Load all models
models = {}
for name, path in model_paths.items():
    with open(path, "rb") as file:
        models[name] = pickle.load(file)

# Load the TFIDF vectorizer
with open(tfidf_vectorizer_path, "rb") as file:
    tfidf_vectorizer = pickle.load(file)

# Personality Mapping and Descriptions
personality_mapping = {
    0: "ISTJ", 1: "ISFJ", 2: "INFJ", 3: "INTJ", 4: "ISTP", 5: "ISFP",
    6: "INFP", 7: "INTP", 8: "ESTP", 9: "ESFP", 10: "ENFP", 11: "ENTP",
    12: "ESTJ", 13: "ESFJ", 14: "ENFJ", 15: "ENTJ"
}

personality_descriptions = {
    "ISTJ": "Practical, detail-oriented, reliable, and organized. Prefers structure and values tradition.",
    "ISFJ": "Compassionate, loyal, and considerate. Often focused on helping others and maintaining harmony.",
    "INFJ": "Introspective, empathetic, and idealistic. Known for a deep sense of purpose and a desire to improve the world.",
    "INTJ": "Strategic, independent, and visionary. Enjoys analyzing systems and solving complex problems.",
    "ISTP": "Practical, observant, and adaptable. Prefers hands-on problem-solving and thrives in dynamic environments.",
    "ISFP": "Gentle, artistic, and flexible. Values personal expression and seeks beauty in the world around them.",
    "INFP": "Thoughtful, idealistic, and introspective. Guided by strong personal values and a desire to help others.",
    "INTP": "Analytical, curious, and innovative. Enjoys exploring ideas and solving intellectual challenges.",
    "ESTP": "Energetic, pragmatic, and action-oriented. Excels in fast-paced environments and enjoys taking risks.",
    "ESFP": "Spontaneous, outgoing, and fun-loving. Thrives on social interaction and bringing joy to others.",
    "ENFP": "Enthusiastic, creative, and open-minded. Values individuality and inspires others with their optimism.",
    "ENTP": "Quick-witted, curious, and innovative. Enjoys debating ideas and exploring new possibilities.",
    "ESTJ": "Organized, practical, and decisive. Skilled at managing teams and ensuring efficiency.",
    "ESFJ": "Warm, sociable, and community-focused. Strives to create harmony and nurture relationships.",
    "ENFJ": "Charismatic, empathetic, and inspiring. Motivates others and leads with a focus on shared goals.",
    "ENTJ": "Strategic, assertive, and results-driven. Excels in leadership roles and values efficiency.",
}

# Function to parse PDF and extract text


def parse_pdf(file):
    text = ""
    try:
        with fitz.open(stream=file.read(), filetype="pdf") as pdf:
            for page in pdf:
                text += page.get_text()
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

# Function to preprocess text


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    # Tokenization
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# Streamlit App
st.title("AI Driven MBTI Personality Traits Predictor from CV/Resume")
st.write("Upload a resume (PDF), and this app will classify it using a pre-trained model.")

# Model selection
selected_model_name = st.selectbox("Select a Model", list(models.keys()))
selected_model = models[selected_model_name]

# File upload
uploaded_file = st.file_uploader(
    "Upload your resume (PDF format only)", type=["pdf"])

# Predict button
if st.button("Predict"):
    if uploaded_file is not None:
        # Parse the PDF
        raw_text = parse_pdf(uploaded_file)
        if raw_text:
            # Preprocess the text
            processed_text = preprocess_text(raw_text)
            # Make prediction
            try:
                vectorized_text = tfidf_vectorizer.transform([processed_text])
                predicted_label = selected_model.predict(
                    vectorized_text)[0]  # Get the predicted label (integer)
                # Get the personality type
                personality_type = personality_mapping[predicted_label]
                # Get its description
                personality_description = personality_descriptions[personality_type]

                st.success(
                    f"Prediction: {predicted_label} - {personality_type}")
                st.write(f"**Personality Type:** {personality_type}")
                st.write(f"**Description:** {personality_description}")
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
    else:
        st.error("Please upload a valid PDF file.")
