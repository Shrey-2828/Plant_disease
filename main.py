import streamlit as st
import tensorflow as tf
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'plan_disease.h5',
        compile=False
    )
model = load_model()

def model_pred(image):
    img=tf.keras.preprocessing.image.load_img(image,target_size=(256,256))
    img_arr=tf.keras.preprocessing.image.img_to_array(img)
    img_arr=np.array([img_arr])
    predict=model.predict(img_arr)
    result_index=np.argmax(predict)
    return result_index


st.set_page_config(
    page_title="PlantGuard | Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("PlantGuard - Plant Disease Protector",icon="🍃",text_alignment="center",width="content")
st.sidebar.title("Navigation",icon="🧭",width="content")
app_mode=st.sidebar.radio("PAGES",["Home","Disease Prediction"])


#home
if app_mode=='Home':

    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently.
    Upload an image of a plant, and our system will analyze it to detect
    any signs of diseases. Together, let's protect our crops and ensure
    a healthier harvest!
    """)
    
    #st.image('home.jpg')
    

#diese cover    
    st.markdown("---")

    st.markdown(
        """
        <h2 style="
            text-align:center;
            margin-bottom:5px;
        ">
            🌿 Diseases Covered by PlantGuard
        </h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p style="
            text-align:center;
            color:grey;
            margin-bottom:30px;
        ">
            Plants and diseases supported by our disease detection model
        </p>
        """,
        unsafe_allow_html=True
    )

#data card
    diseases = {
        "🍎 Apple": [
            "Apple Scab",
            "Black Rot",
            "Cedar Apple Rust",
            "Healthy"
        ],

        "🫐 Blueberry": [
            "Healthy"
        ],

        "🍒 Cherry": [
            "Powdery Mildew",
            "Healthy"
        ],

        "🌽 Corn (Maize)": [
            "Cercospora Leaf Spot / Gray Leaf Spot",
            "Common Rust",
            "Northern Leaf Blight",
            "Healthy"
        ],

        "🍇 Grape": [
            "Black Rot",
            "Esca (Black Measles)",
            "Leaf Blight (Isariopsis Leaf Spot)",
            "Healthy"
        ],

        "🍊 Orange": [
            "Haunglongbing (Citrus Greening)"
        ],

        "🍑 Peach": [
            "Bacterial Spot",
            "Healthy"
        ],

        "🫑 Pepper Bell": [
            "Bacterial Spot",
            "Healthy"
        ],

        "🥔 Potato": [
            "Early Blight",
            "Late Blight",
            "Healthy"
        ],

        "🫐 Raspberry": [
            "Healthy"
        ],

        "🌱 Soybean": [
            "Healthy"
        ],

        "🎃 Squash": [
            "Powdery Mildew"
        ],

        "🍓 Strawberry": [
            "Leaf Scorch",
            "Healthy"
        ],

        "🍅 Tomato": [
            "Bacterial Spot",
            "Early Blight",
            "Late Blight",
            "Leaf Mold",
            "Septoria Leaf Spot",
            "Spider Mites (Two-spotted Spider Mite)",
            "Target Spot",
            "Tomato Yellow Leaf Curl Virus",
            "Tomato Mosaic Virus",
            "Healthy"
        ]
    }



    st.markdown(
        """
        <style>

        .disease-card {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 14px;
            padding: 20px;
            margin-bottom: 20px;
            background: rgba(128, 128, 128, 0.05);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            min-height: 245px;
        }

        .disease-title {
            font-size: 21px;
            font-weight: 700;
            margin-bottom: 14px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(128, 128, 128, 0.25);
        }

        .disease-item {
            font-size: 15px;
            padding: 5px 0;
            line-height: 1.45;
        }

        /* Tomato has more diseases, so show only its list in two columns */
        .tomato-card {
            min-height: 245px;
            width: calc(200% + 20px);
            box-sizing: border-box;
        }

        .tomato-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            column-gap: 18px;
        }

        .tomato-grid .disease-item {
            padding: 4px 0;
            font-size: 14px;
        }

        @media (max-width: 900px) {
            .tomato-grid {
                grid-template-columns: 1fr;
            }
        }

        </style>
        """,
        unsafe_allow_html=True
    )

    

    disease_items = list(diseases.items())

    for i in range(0, len(disease_items), 3):

        cols = st.columns(3)

        for j in range(3):

            if i + j < len(disease_items):

                plant_name, disease_list = disease_items[i + j]

                if plant_name == "🍅 Tomato":
                    disease_html = (
                        '<div class="disease-card tomato-card">'
                        '<div class="disease-title">🍅 Tomato</div>'
                        '<div class="tomato-grid">'
                    )

                    for disease in disease_list:
                        disease_html += (
                            f'<div class="disease-item">• {disease}</div>'
                        )

                    disease_html += '</div></div>'

                else:
                    disease_html = (
                        '<div class="disease-card">'
                        f'<div class="disease-title">{plant_name}</div>'
                    )

                    for disease in disease_list:
                        disease_html += (
                            f'<div class="disease-item">• {disease}</div>'
                        )

                    disease_html += '</div>'

                with cols[j]:
                    st.markdown(
                        disease_html,
                        unsafe_allow_html=True
                    )
    
    
    st.markdown("""
 
     ### How It Works
 
     1. **Upload Image:** Go to the **Disease Recognition** page and upload
        an image of a plant's leaf with suspected diseases.
     2. **Analysis:** Our system will process the image using deep learning model to identify potential diseases.
     3. **Results:** View the results and recommendations for further action.
 
 
     ### Get Started
 
     Click on the **Disease Prediction** page in the left sidebar to upload an
     image and experience the power of our Plant Disease Recognition System!
     """)
 
#prediction
if app_mode=='Disease Prediction':
    st.subheader("Upload image of your plant's leaf to check if it has any disease or not..")

    test_img=st.file_uploader("Choose image:",type=['jpg','png','jpeg'])
    if test_img is not None:

        if st.button("Predict"):
              st.image(test_img)
              result=model_pred(test_img)
              class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']

              if class_name[result].endswith("healthy"):
                  st.success("Model is Predicting it's a {}".format(class_name[result]))
                  
              else:
                  st.error("Model is Predicting it's a {}".format(class_name[result]))
         