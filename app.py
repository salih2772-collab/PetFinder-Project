import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import cv2
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

# --- Page Configuration ---
st.set_page_config(page_title="PetFinder AI Pro", page_icon="🐾", layout="wide")

st.title("🐾 PetFinder Adoption Smart Predictor")
st.write("Full-Feature AI Model: Metadata + NLP + CNN + Image Analytics")

# --- Path Handling ---
current_dir = os.path.dirname(os.path.abspath(__file__))

def get_file_path(filename):
    return os.path.join(current_dir, filename)

# --- Load Models ---
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load(get_file_path('stacking_model.pkl'))
        scaler = joblib.load(get_file_path('scaler.pkl'))
        encoders = joblib.load(get_file_path('label_encoders.pkl'))
        tfidf = joblib.load(get_file_path('tfidf_vectorizer.pkl'))
        pca = joblib.load(get_file_path('pca_adapter.pkl'))
        feature_names = joblib.load(get_file_path('feature_names_final.pkl'))
        cnn_base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
        return model, scaler, encoders, tfidf, pca, feature_names, cnn_base
    except FileNotFoundError as e:
        st.error(f"❌ Error: Files not found in {current_dir}")
        return None, None, None, None, None, None, None

model, scaler, encoders, tfidf, pca, feature_names, cnn_base = load_artifacts()

if model:
    st.success("✅ All AI Modules Loaded Successfully!")

st.write("---")

# --- User Input Section ---
with st.sidebar:
    st.header("📝 Pet Details")
    age = st.slider("Age (Months)", 1, 200, 3)
    photo_amt = st.number_input("Total Photos Available", 1, 30, 5)
    fee = st.number_input("Adoption Fee ($)", 0, 1000, 0)
    
    type_animal = st.radio("Animal Type", ["Dog", "Cat"], horizontal=True)
    gender = st.selectbox("Gender", ["Male", "Female", "Mixed"])
    
    col_breed, col_color = st.columns(2)
    with col_breed:
        breed1 = st.text_input("Primary Breed", "Mixed Breed")
    with col_color:
        color1 = st.selectbox("Primary Color", ["Black", "Brown", "Golden", "White", "Cream", "Gray", "Yellow"])
    
    size = st.selectbox("Size", ["Small", "Medium", "Large", "Extra Large"])
    
    col_vac, col_ster = st.columns(2)
    with col_vac:
        vaccinated = st.selectbox("Vaccinated", ["Yes", "No", "Not Sure"])
    with col_ster:
        sterilized = st.selectbox("Sterilized", ["Yes", "No", "Not Sure"])
        
    health = st.selectbox("Health Condition", ["Healthy", "Minor Injury", "Serious Injury"])
    desc = st.text_area("Description", "Cute, playful, and healthy. Loves to play with ball.")
    uploaded_file = st.file_uploader("Upload Image (Required)", type=["jpg", "png", "jpeg"])

# --- Function: Extract Image Features (For Display & Model) ---
def analyze_image(pil_image):
    # Convert to OpenCV format
    img_array = np.array(pil_image.convert('RGB')) 
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Brightness
    brightness = np.mean(img_gray)
    
    # 2. Edge Density (Texture)
    edges = cv2.Canny(img_gray, 100, 200)
    edge_density = np.sum(edges) / edges.size
    
    # 3. Saturation (Vibrance)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    
    # 4. Deep Learning (CNN + PCA)
    img_resized = pil_image.resize((128, 128)).convert('RGB')
    x = np.array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = cnn_base.predict(x, verbose=0)
    features_pca = pca.transform(features)
    
    return brightness, edge_density, saturation, features_pca

# --- Main Processing Function ---
def process_input(bright, edge, sat, pca_features):
    # Helper for encoding
    def safe_transform(col_name, val):
        try:
            return encoders[col_name].transform([val])[0]
        except:
            return 0 

    # 1. Metadata
    data = {
        'Type': 1 if type_animal == 'Dog' else 2,
        'Age': age,
        'Gender': 1 if gender == 'Male' else (2 if gender == 'Female' else 3),
        'MaturitySize': ["Small", "Medium", "Large", "Extra Large"].index(size) + 1,
        'FurLength': 1, 
        'Vaccinated': 1 if vaccinated == 'Yes' else (2 if vaccinated == 'No' else 3),
        'Dewormed': 1, 
        'Sterilized': 1 if sterilized == 'Yes' else (2 if sterilized == 'No' else 3),
        'Health': 1 if health == 'Healthy' else (2 if health == 'Minor Injury' else 3),
        'Quantity': 1,
        'Fee': fee,
        'VideoAmt': 0,
        'PhotoAmt': photo_amt,
        'RescuerID': '0', 'PetID': '0'
    }

    # Encode Categoricals
    data['Breed1_Name'] = safe_transform('Breed1_Name', breed1)
    data['Breed2_Name'] = 0 
    data['Color1_Name'] = safe_transform('Color1_Name', color1)
    data['Color2_Name'] = 0 
    data['Color3_Name'] = 0
    data['State_Name'] = safe_transform('State_Name', "Selangor")

    df = pd.DataFrame([data])
    
    # 2. Text Features
    text_vectors = tfidf.transform([desc]).toarray()
    tfidf_cols = [f'tfidf_{word}' for word in tfidf.get_feature_names_out()]
    df_tfidf = pd.DataFrame(text_vectors, columns=tfidf_cols)
    
    # 3. Image PCA Features
    pca_cols = [f'img_pca_{i}' for i in range(32)]
    df_pca = pd.DataFrame(pca_features, columns=pca_cols)
    
    # 4. Add Manual Image Stats (Try using names that match your training code)
    # Even if model drops them, we add them here first
    df['img_brightness'] = bright
    df['img_edge_density'] = edge
    df['img_saturation'] = sat
    # Also try alternative names just in case your notebook used these:
    df['Brightness'] = bright
    df['Vibrance'] = sat
    df['Texture'] = edge
    
    # 5. Combine All
    df_final = pd.concat([df, df_tfidf, df_pca], axis=1)
    
    # 6. Final Alignment (The Safety Filter)
    # This keeps ONLY the columns the model actually learned
    # If 'img_brightness' wasn't in training, it gets dropped here (preventing errors)
    for col in feature_names:
        if col not in df_final.columns:
            df_final[col] = 0
            
    return df_final[feature_names]

# --- Main Logic ---
if uploaded_file:
    st.image(uploaded_file, caption="Analyzed Pet Image", width=300)

if st.button("🔮 Predict Adoption Speed", type="primary"):
    if model:
        with st.spinner('Analyzing Brightness, Texture, and Details...'):
            try:
                # 1. Analyze Image First (Get stats separately)
                if uploaded_file:
                    img = Image.open(uploaded_file)
                    # Get values directly into variables
                    b_val, e_val, s_val, pca_val = analyze_image(img)
                else:
                    # Default if no image
                    b_val, e_val, s_val = 0, 0, 0
                    pca_val = np.zeros((1, 32))
                
                # 2. Create Model Input (Pass stats in)
                input_data = process_input(b_val, e_val, s_val, pca_val)
                input_scaled = scaler.transform(input_data)
                
                # 3. Predict
                probs = model.predict_proba(input_scaled)[0]
                prediction = np.argmax(probs)
                
                # 4. Display Results
                speeds = {
                    0: "Same Day (0)",
                    1: "1-7 Days (1)",
                    2: "8-30 Days (2)",
                    3: "1-3 Months (3)",
                    4: "Slow +100 Days (4)"
                }
                
                result_text = speeds.get(prediction)
                
                col_res, col_chart = st.columns([1, 2])
                
                with col_res:
                    st.subheader("Prediction:")
                    if prediction <= 2:
                        st.success(f"🚀 {result_text}")
                    elif prediction == 3:
                        st.warning(f"⚖️ {result_text}")
                    else:
                        st.error(f"🐢 {result_text}")
                    
                    st.metric("Confidence", f"{probs[prediction]*100:.1f}%")
                    
                    # 5. Show Stats (Using variables, NOT dataframe)
                    # This prevents the KeyError because we don't rely on the dataframe
                    if uploaded_file:
                        st.markdown("---")
                        st.write("**Image Stats:**")
                        st.write(f"☀️ Brightness: {b_val:.1f}")
                        st.write(f"🎨 Vibrance: {s_val:.1f}")
                        st.write(f"🔍 Detail: {e_val:.3f}")

                with col_chart:
                    st.write("📊 Probability Distribution:")
                    chart_data = pd.DataFrame(probs, index=speeds.values(), columns=["Probability"])
                    st.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Processing Error: {e}")