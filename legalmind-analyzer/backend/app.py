from flask import Flask, request, jsonify
from flask_cors import CORS
import pdfplumber
import pytesseract
from PIL import Image
import spacy
import google.generativeai as genai
import os
import tempfile
import io
import json

app = Flask(__name__)
CORS(app)  # Allow all origins in production

# Configure Gemini API (use environment variable in production)
genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyDzzYbKPHfWTByl9SefFXu7Sw-GHfKQwdI"))

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


def extract_text_from_pdf(file_path):
    """Extract text from PDF file (no OCR fallback)"""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text



def extract_text_from_image(file):
    """Extract text from image using OCR"""
    file.seek(0)  # Reset file pointer
    image = Image.open(io.BytesIO(file.read()))
    text = pytesseract.image_to_string(image)
    return text


def preprocess_text(text):
    """Preprocess text using spaCy"""
    doc = nlp(text)
    # Basic preprocessing - remove stopwords, punctuation, and lemmatize
    processed_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(processed_tokens)


def parse_gemini_response(response_text, analysis_type):
    """Parse Gemini response and extract JSON data"""
    try:
        # Try to find JSON in the response
        if "```json" in response_text:
            json_str = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            json_str = response_text.split("```")[1].split("```")[0].strip()
        else:
            json_str = response_text.strip()

        # Parse the JSON
        data = json.loads(json_str)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from Gemini response: {e}")
        print(f"Response text: {response_text}")

        # Fallback for different analysis types
        if analysis_type == "classification":
            return {"type": "Unknown", "confidence": "N/A"}
        elif analysis_type == "risks":
            return [{"level": "Unknown", "description": "Could not parse risks from response"}]
        elif analysis_type == "terms":
            return ["Could not extract terms"]
        elif analysis_type == "summary":
            return {"summary": response_text[:500] + "..."}
        return None


def analyze_with_gemini(text, analysis_type):
    """Analyze text using Google Gemini API"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        if analysis_type == "classification":
            prompt = f"""Classify this legal document and provide the result in JSON format with 'type' and 'confidence' fields. 
            The type should be a specific legal document category like 'Rental Agreement', 'Employment Contract', 'NDA', etc.
            Example response: {{"type": "Rental Agreement", "confidence": "92%"}}

            Document text: {text[:3000]}"""

        elif analysis_type == "risks":
            prompt = f"""Identify risks in this legal document and provide the result as a JSON array with objects containing 'level' (High, Medium, Low) and 'description' fields.
            Example response: [{{"level": "High", "description": "Clause 6 imposes excessive penalties for early termination"}}]

            Document text: {text[:3000]}"""

        elif analysis_type == "terms":
            prompt = f"""Extract key terms from this legal document and provide as a JSON array of strings.
            Example response: ["Lease Term: 12 months", "Monthly Rent: $2,500"]

            Document text: {text[:3000]}"""

        elif analysis_type == "summary":
            prompt = f"""Provide a comprehensive summary of this legal document as plain text only.
            Example response: "This is a standard residential lease agreement with mostly reasonable terms."

            Document text: {text[:3000]}"""

        else:
            return None

        response = model.generate_content(prompt)
        parsed = parse_gemini_response(response.text, analysis_type)

        # Unwrap summary for frontend
        if analysis_type == "summary":
            if isinstance(parsed, dict) and "summary" in parsed:
                return parsed["summary"]
            elif isinstance(parsed, str):
                return parsed
            else:
                return str(parsed)

        return parsed
    except Exception as e:
        print(f"Error in Gemini API call: {e}")
        # safe defaults
        if analysis_type == "classification":
            return {"type": "Unknown", "confidence": "N/A"}
        elif analysis_type == "risks":
            return [{"level": "Unknown", "description": "Could not parse risks"}]
        elif analysis_type == "terms":
            return ["Could not extract terms"]
        elif analysis_type == "summary":
            return "Could not generate summary"
        return None

@app.route('/api/analyze/text', methods=['POST'])
def analyze_text():
    """Endpoint for analyzing text input"""
    data = request.json
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    processed_text = preprocess_text(text)

    # Call Gemini for real analysis
    classification = analyze_with_gemini(processed_text, "classification")
    risks = analyze_with_gemini(processed_text, "risks")
    terms = analyze_with_gemini(processed_text, "terms")
    summary = analyze_with_gemini(processed_text, "summary")

    result = {
        'classification': classification,
        'risks': risks,
        'terms': terms,
        'summary': summary
    }

    return jsonify(result)


@app.route('/api/analyze/file', methods=['POST'])
def analyze_file():
    """Endpoint for analyzing uploaded file"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
        file.save(tmp_file.name)
        tmp_file_path = tmp_file.name

    try:
        # Extract text
        if file.filename.lower().endswith('.pdf'):
            text = extract_text_from_pdf(tmp_file_path)
        elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            text = extract_text_from_image(file)
        else:
            text = file.read().decode('utf-8')

        os.unlink(tmp_file_path)

        if not text:
            return jsonify({'error': 'Could not extract text from file'}), 400

        processed_text = preprocess_text(text)

        # Call Gemini for real analysis
        classification = analyze_with_gemini(processed_text, "classification")
        risks = analyze_with_gemini(processed_text, "risks")
        terms = analyze_with_gemini(processed_text, "terms")
        summary = analyze_with_gemini(processed_text, "summary")

        result = {
            'classification': classification,
            'risks': risks,
            'terms': terms,
            'summary': summary
        }

        return jsonify(result)

    except Exception as e:
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Endpoint for chat functionality"""
    data = request.json
    message = data.get('message', '')
    document_context = data.get('context', '')

    if not message:
        return jsonify({'error': 'No message provided'}), 400

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')

        if document_context:
            prompt = f"Based on this legal document: {document_context[:2000]}\n\nAnswer this question: {message}"
        else:
            prompt = f"Answer this question about legal documents: {message}"

        response = model.generate_content(prompt)
        return jsonify({'response': response.text})
    except Exception as e:
        return jsonify({'error': f'Error generating response: {str(e)}'}), 500

def chunk_text(text, max_chars=2500):
    """
    Split long text into chunks <= max_chars (Gemini context limit).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end
    return chunks


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
