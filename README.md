# LLM Inference Speed Comparison

A Streamlit application to compare inference speeds across different LLM providers.

link : https://llm-inference-akash.streamlit.app/

## Features

- Compare inference speeds across multiple providers:
  - Google (Gemini)
  - Cohere
  - Groq
  - Together AI
  - HuggingFace
  - Cerebras
- Real-time inference timing
- Easy-to-use interface
- Support for multiple models per provider

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.template` to `.env`:
```bash
cp .env.template .env
```

4. Add your API keys to `.env` file:
- Get API keys from each provider's platform:
  - Google AI Studio: https://makersuite.google.com/
  - Cohere: https://dashboard.cohere.com/
  - Groq: https://console.groq.com/
  - Together AI: https://www.together.ai/
  - HuggingFace: https://huggingface.co/settings/tokens
  - Replicate: https://replicate.com/account
  - Cerebras: https://cloud.cerebras.ai/platform/org_2eek5cnmk9jd95kkdk9e2xe5/playground

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Select a provider and model from the sidebar
3. Enter your prompt
4. Click "Generate Response" to see the results

## Notes

- Inference times may vary based on:
  - Network conditions
  - Provider load
  - Model size
  - Input length
- Some providers may have rate limits on their free tier
- Keep your API keys secure and never commit them to version control 
