# Image OCR with Gemini API

A Streamlit web application that extracts text from images using Google's Gemini AI API.

## Features

- 📤 Upload images in multiple formats (PNG, JPG, JPEG, GIF, BMP, WebP)
- 🔍 Extract text using Google's Gemini 1.5 Flash model
- 📝 Display extracted text with copy functionality
- 💾 Download extracted text as a file
- 🎨 Clean, responsive UI

## Setup

### 1. Install Dependencies

```bash
uv sync
```

### 2. Set up Gemini API Key

Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey) and set it as:

**Option A: Environment Variable**
```bash
export GEMINI_API_KEY="your-api-key-here"
```

**Option B: Streamlit Secrets**
Create `.streamlit/secrets.toml`:
```toml
GEMINI_API_KEY = "your-api-key-here"
```

### 3. Run the Application

```bash
uv run streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`

## Usage

1. Upload an image containing text
2. Click "Extract Text" to process with Gemini AI
3. View and copy the extracted text
4. Optionally download the text as a file

## Requirements

- Python >=3.12
- Streamlit
- Google Generative AI
- Pillow

## Project Structure

```
price_update/
├── main.py              # Streamlit application
├── pyproject.toml       # Project configuration
├── README.md           # This file
├── .python-version     # Python version
└── .gitignore         # Git ignore rules
```