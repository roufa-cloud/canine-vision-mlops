import sys
from pathlib import Path
import streamlit as st
import base64
from PIL import Image
import pandas as pd
import time
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
from src.inference.predictor import create_predictor

st.set_page_config(
    page_title="Canine Breed Classifier",
    page_icon="assets/logo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .confidence-bar {
        background-color: #e0e0e0;
        border-radius: 5px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    .confidence-fill {
        background-color: #4CAF50;
        height: 25px;
        text-align: right;
        padding-right: 10px;
        color: white;
        font-weight: bold;
        line-height: 25px;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #ddd;
        line-height: 1;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """ load best predictor model, cached for performance """
    with st.spinner("Loading model..."):
        try:
            predictor = create_predictor(use_best_model=True)
            return predictor
        except FileNotFoundError as e:
            st.error(f"[ALERT] Model not found: {e}")
            st.info("""
                    **Setup Required:**
                    1. Train models (notebooks 02-04)
                    2. Run evaluation (notebook 05)
                    3. Run: `python scripts/prepare_deployment_artifacts.py`
                    """)
            st.stop()


def display_prediction_results(result, image):
    """ display prediction results  """

    st.markdown(f"""<div class="prediction-box">
                <h3><strong>{result['top_prediction']}</strong></h3>
                <p><strong>Confidence:</strong> {result['top_confidence']:.1%}</p>
                </div>""", unsafe_allow_html=True)

    st.markdown("### Top 5 Predictions")
    for i, (breed, confidence) in enumerate(result['top_k_predictions'], 1):
        col1, col2, col3 = st.columns([0.5, 3, 2])
        with col1:
            st.markdown(f"**{i}.**")
        with col2:
            st.markdown(f"**{breed}**")
        with col3:
            bar_width = int(confidence*100)
            st.markdown(f"""<div class="confidence-bar">
                        <div class="confidence-fill" style="width: {bar_width}%;">
                        {confidence:.1%}</div> </div>""", unsafe_allow_html=True)

def main():
    """ main application function """

    # header
    with open("assets/logo.png", "rb") as f:
        logo64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
    <div style='display:flex; align-items:center; justify-content:center; margin-bottom:1rem;'>
        <img src='data:image/png;base64,{logo64}' style='height:50px; margin-right:1rem;'>
        <div style='font-size:2.5rem; font-weight:bold; color:#1f77b4;'>
            Canine Breed Classifier
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-header">Upload a photo of a dog to identify its breed using deep learning</div>', unsafe_allow_html=True)
    predictor = load_model()

    # side bar
    with st.sidebar:
        st.markdown("## Model Information")
        model_info = predictor.get_model_info()
        card_html = f"""<div class="metric-card">
        <p><strong>Model:</strong> {model_info['model_name']}</p>
        <p><strong>Source:</strong> {model_info.get('model_source', 'unknown').upper()}</p>
        <p><strong>Classes:</strong> {model_info['num_classes']}</p>"""

        val_acc = model_info.get("val_accuracy")
        if val_acc is not None:
            card_html += f"<p><strong>Validation Accuracy:</strong> {val_acc:.2%}</p>"

        top5_acc = model_info.get("val_top_5_accuracy")
        if top5_acc is not None:
            card_html += f"<p><strong>Top-5 Accuracy:</strong> {top5_acc:.2%}</p>"

        card_html += "</div>"
        st.markdown(card_html, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("## About")
        st.markdown("""
        This classifier can identify **120 dog breeds** from the Stanford Dogs dataset.
        
        **How it works:**
        1. Upload an image
        2. Model processes the image
        3. Get breed predictions with confidence scores
        
        **Model Details:**
        - Architecture: Transfer learning with pretrained CNN
        - Training: Stanford Dogs dataset (12,000 images)
        - Accuracy: 85%+ on validation set
        """)
        
        st.markdown("---")
        st.markdown("### Links")
        st.markdown("""
        - [GitHub Repository](https://github.com/roufa-cloud/canine-vision-mlops.git)
        - [Dataset Info](http://vision.stanford.edu/aditya86/ImageNetDogs/)
        """)
    
    # main content
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("## Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a dog image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of a dog (JPG, JPEG, or PNG)"
        )
        
        st.markdown("### Or try a sample image:")
        sample_dir = project_root/"assets"/"sample_images"
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
            
            if sample_images:
                sample_names = [img.stem.replace("_", " ").title() for img in sample_images]
                selected_sample = st.selectbox(
                    "Select a sample:",
                    ["None"] + sample_names
                )
                
                if selected_sample != "None":
                    sample_idx = sample_names.index(selected_sample)
                    uploaded_file = str(sample_images[sample_idx])
        else:
            st.info("Add sample images to `assets/sample_images/` folder for quick testing")
        
        if uploaded_file is not None:
            if isinstance(uploaded_file, str):
                image = Image.open(uploaded_file)
            else:
                image = Image.open(uploaded_file)
   
            # st.image(image, caption="Image uploaded", width="content")
            st.image(image, caption="Image uploaded", width=500) 
            if st.button("Classify Breed", type="primary", use_container_width=True):
            # if st.button("Classify Breed", type="primary", width="stretch"):
                with st.spinner("Analyzing image..."):
                    time.sleep(0.1)
                    try:
                        result = predictor.predict(image, top_k=5)
                        st.session_state['prediction_result'] = result
                        st.session_state['prediction_image'] = image
                        st.success("Classification complete!")
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
                        st.exception(e)
    
    with col2:
        st.markdown("## Prediction Results")
        if 'prediction_result' in st.session_state:
            display_prediction_results(
                st.session_state['prediction_result'],
                st.session_state['prediction_image']
            )
            
            with st.expander("View All Probabilities"):
                result = st.session_state['prediction_result']
                all_probs = result['all_probabilities']
                class_names = predictor.class_names
                prob_df = pd.DataFrame({
                    'Breed': [name.replace('_', ' ').title() for name in class_names],
                    'Probability': all_probs
                })
                
                prob_df = prob_df.sort_values('Probability', ascending=False).reset_index(drop=True)
                prob_df['Rank'] = prob_df.index + 1
                prob_df = prob_df[['Rank', 'Breed', 'Probability']]
                prob_df['Probability'] = prob_df['Probability'].apply(lambda x: f"{x:.2%}")
                st.dataframe(
                    prob_df.head(20),
                    hide_index=True,
                    use_container_width=True
                    # width='stretch'
                )
        else:
            st.info("Upload an image and click 'Classify Breed' to see predictions")
    
    # footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with Streamlit  |  Powered by TensorFlow</p>
        <p>Stanford Dogs Dataset  |  120 Breeds  |  Transfer Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


