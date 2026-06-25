import pickle
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Fish Species Predictor",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for header color and styling
st.markdown("""
<style>
    /* Change header gradient color */
    .stApp header {
        background: linear-gradient(90deg, #0d7377 0%, #14a3a8 50%, #0d7377 100%) !important;
    }
    .stApp header a, .stApp header button {
        color: #e0f7fa !important;
    }
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem 0;
        color: #666;
        font-size: 0.9rem;
    }
    .footer a {
        color: #0d7377;
        text-decoration: none;
        font-weight: 500;
    }
    .footer a:hover {
        color: #14a3a8;
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📖 About This App")
    st.markdown("""
    This app uses a **Decision Tree model** to predict fish species based on:
    - **Length**: Body length in centimeters
    - **Weight**: Body weight in grams
    - **Ratio**: Length-to-weight ratio
    """)

    st.markdown("---")
    st.markdown("### 💡 Quick Tips")
    st.markdown("""
    - Use the **Presets** tab to try example fish
    - The model works best with realistic fish measurements
    - Different species have different length/weight ratios
    """)

# Main title
col1, col2 = st.columns([0.7, 0.3])
with col1:
    st.markdown("# 🐟 Fish Species Predictor")
with col2:
    st.markdown("")
    st.markdown("")
    st.markdown("*Powered by Machine Learning*")

# Tabs for different modes
tab1, tab2 = st.tabs(["🎯 Predictor", "📊 Presets"])

with tab2:
    st.markdown("### Load Example Fish Data")

    fish_presets = {
        "Common Carp": {"length": 45.0, "weight": 1250.0},
        "Pike": {"length": 50.0, "weight": 900.0},
        "Roach": {"length": 25.0, "weight": 200.0},
        "Perch": {"length": 30.0, "weight": 350.0},
        "Trout": {"length": 32.0, "weight": 400.0},
        "Salmon": {"length": 60.0, "weight": 2000.0},
        "Bream": {"length": 28.0, "weight": 500.0},
    }

    selected_preset = st.selectbox(
        "Select a fish preset:",
        list(fish_presets.keys()),
        help="Choose a typical fish to see a prediction"
    )

    if selected_preset:
        preset_data = fish_presets[selected_preset]
        st.info(
            f"✨ {selected_preset}: {preset_data['length']}cm, {preset_data['weight']}g")

with tab1:
    st.markdown("### Enter Fish Measurements")

    # Check if we have preset selected
    preset_length = None
    preset_weight = None
    if 'selected_preset' in dir() and selected_preset:
        preset_length = fish_presets[selected_preset]["length"]
        preset_weight = fish_presets[selected_preset]["weight"]

    col1, col2, col3 = st.columns(3)

    with col1:
        length = st.number_input(
            "Length (cm)",
            min_value=0.0,
            max_value=1000.0,
            value=preset_length or 20.0,
            help="Measure the fish from head to tail in centimeters"
        )

    with col2:
        weight = st.number_input(
            "Weight (g)",
            min_value=0.0,
            max_value=100000.0,
            value=preset_weight or 200.0,
            help="Weigh the fish in grams"
        )

    with col3:
        st.markdown("**Auto Ratio**")
        st.markdown("")
        auto_ratio = (length / weight) if weight != 0 else 0.0
        st.metric("Computed", f"{auto_ratio:.4f}")

    ratio = st.number_input(
        "Length/Weight Ratio",
        min_value=0.0,
        value=auto_ratio,
        help="Length ÷ Weight. Auto-calculated but you can adjust if needed.",
        step=0.0001
    )

    # Visualization of inputs
    with st.expander("📈 View Input Statistics", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Length", f"{length:.1f} cm")
        with col2:
            st.metric("Weight", f"{weight:.1f} g")
        with col3:
            st.metric("Ratio", f"{ratio:.4f}")

    input_data = [[length, weight, ratio]]

MODEL_PATH = "decision_fish.pkl"
ENCODER_PATH = "decision_fish_encoder.pkl"

model = None
encoder = None


def load_artifacts():
    global model, encoder
    if model is None:
        try:
            with open(MODEL_PATH, "rb") as mf:
                model = pickle.load(mf)
        except Exception as e:
            st.error(f"❌ Could not load model '{MODEL_PATH}': {e}")
    if encoder is None:
        try:
            with open(ENCODER_PATH, "rb") as ef:
                encoder = pickle.load(ef)
        except Exception as e:
            st.warning(f"⚠️ Could not load encoder '{ENCODER_PATH}': {e}")


# Prediction button with better styling
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🎯 Predict Fish Species", use_container_width=True):
        load_artifacts()
        if model is None:
            st.error(
                "❌ Model not available. Place the model file 'decision_fish.pkl' in this folder.")
        else:
            with st.spinner("🔄 Analyzing fish measurements..."):
                try:
                    pred = model.predict(input_data)

                    # Display results in a nice container
                    st.markdown("---")
                    result_col1, result_col2 = st.columns([1, 1])

                    with result_col1:
                        if encoder is not None:
                            try:
                                species = encoder.inverse_transform(pred)
                                st.success(
                                    f"### ✅ Prediction Result\n\n**Species:** {species[0]}")
                            except Exception:
                                st.success(
                                    f"### ✅ Prediction Result\n\n**Label:** {pred[0]}")
                        else:
                            st.success(
                                f"### ✅ Prediction Result\n\n**Label:** {pred[0]}")

                    with result_col2:
                        st.info(f"""
                        ### 📋 Input Summary
                        - **Length:** {length:.1f} cm
                        - **Weight:** {weight:.1f} g
                        - **Ratio:** {ratio:.4f}
                        """)

                    st.markdown("---")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

# Footer
st.markdown("---")
st.markdown(
    '<div class="footer">Built by <a href="https://github.com/junaidnawaz1" target="_blank">Junaid Nawaz</a> 🚀</div>',
    unsafe_allow_html=True,
)
st.markdown("---")
with st.expander("ℹ️ How This Works"):
    st.markdown("""
    ### Model Information
    This app uses a **Decision Tree classifier** trained on fish measurement data.
    
    **Features used:**
    - **Length**: The body length impacts species identification
    - **Weight**: Related to body mass and species
    - **Ratio**: Length-to-weight ratio is a key distinguishing feature
    
    **How to get best results:**
    1. Measure the fish accurately from head to tail
    2. Weigh the fish in grams
    3. The ratio will be automatically calculated
    4. Click "Predict" to see the predicted species
    """)
