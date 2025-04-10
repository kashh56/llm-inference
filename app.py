import streamlit as st
from providers import get_available_providers, get_models_for_provider, validate_api_keys
from inference import get_inference_func
import time
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="LLM Inference Speed Comparison",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css(css_file):
    with open(css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Load the CSS file
load_css('static/styles.css')

# Main title with icon in header
st.markdown("""
    <div class='app-header'>
        <h1 class='gradient-text'>⚡ LLM Inference Speed Comparison</h1>
    </div>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    # Remove any potential empty space at the top
    st.markdown('<style>div[data-testid="stSidebarNav"] {display: none;}</style>', unsafe_allow_html=True)
    
    # Configuration header
    st.markdown("""
        <div class='sidebar-header'>
            <div class='config-header'>⚙️ Configuration</div>
            <div class='api-key-section'>
                <h3>🔑 API Key Status</h3>
                <div class='provider-list'>
    """, unsafe_allow_html=True)

    # Validate API keys and show status
    validation_status = validate_api_keys()
    available_providers = get_available_providers()

    # Show provider status
    for provider, is_valid in validation_status.items():
        status_color = "#4CAF50" if is_valid else "#FF5252"
        status_icon = "✅" if is_valid else "❌"
        st.markdown(
            f"<div class='provider-status' style='color: {status_color};'>{status_icon} {provider}</div>",
            unsafe_allow_html=True
        )

    # Close the API key section
    st.markdown("""
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    if not available_providers:
        logger.error("No providers configured")
        st.error("⚠️ No providers configured. Please check your .env file and API keys.")
        st.stop()

    # Provider selection
    selected_provider = st.selectbox(
        "🔌 Select Provider",
        available_providers
    )
    logger.info(f"Selected provider: {selected_provider}")

    # Model selection
    available_models = get_models_for_provider(selected_provider)
    selected_model = st.selectbox(
        "🤖 Select Model",
        available_models,
        format_func=lambda x: x.split('/')[-1]
    )
    logger.info(f"Selected model: {selected_model}")

# Main content area
st.markdown("""
    <div class='try-it-out'>
        🚀 Try it out!
    </div>
""", unsafe_allow_html=True)

# Input area
user_input = st.text_area(
    "💬 Enter your prompt:",
    height=150,
    placeholder="Type your message here...",
    help="Enter the text you want to process with the selected model"
)

# Create a container for the response section
response_container = st.container()

# Generate button
if st.button("🔮 Generate Response", use_container_width=True):
    if not user_input.strip():
        logger.warning("Empty prompt submitted")
        st.warning("🤔 Please enter a prompt first.")
    else:
        try:
            logger.info(f"Starting inference with {selected_provider}/{selected_model}")
            with st.spinner(f"🔄 Generating response using {selected_model.split('/')[-1]}..."):
                inference_func = get_inference_func(selected_provider)
                response, inference_time = inference_func(user_input, selected_model)
                
                logger.info(f"Inference completed in {inference_time:.2f}s")
                
                # Only show results section if we have a response
                if response:
                    with response_container:
                        # Results section
                        st.markdown("<div class='response-container'>", unsafe_allow_html=True)
                        
                        # Metrics in columns
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("⚡ Inference Time", f"{inference_time:.2f}s")
                        with col2:
                            st.metric("🔌 Provider", selected_provider)
                        with col3:
                            st.metric("🤖 Model", selected_model.split('/')[-1])
                        
                        # Response display
                        st.markdown("### 📝 Generated Response")
                        st.code(response, language="text")
                        
                        st.markdown("</div>", unsafe_allow_html=True)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Inference error: {error_msg}")
            with response_container:
                st.error(f"❌ Error: {error_msg}")

# Footer
st.markdown("""
    <div class='footer'>
        <p>🔍 Compare inference speeds across different LLM providers</p>
        <p>⚙️ Configure your API keys in the <code>.env</code> file to get started</p>
        <p>🌟 Built with Streamlit</p>
    </div>
""", unsafe_allow_html=True) 