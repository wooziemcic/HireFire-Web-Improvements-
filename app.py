from flask import Flask, render_template, request
from flask_sqlalchemy import SQLAlchemy
from transformers import pipeline
import speech_recognition as sr
from pydub import AudioSegment
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import fitz 
import spacy
import cv2
import nltk
import base64
import io
import string
import numpy as np
from deepface import DeepFace

# Initialize Flask app and database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///interviews.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
nlp = spacy.load('en_core_web_sm')

# Initialize the FLAN-T5 model and speech recognizer
generator = pipeline('text2text-generation', model='google/flan-t5-large')
recognizer = sr.Recognizer()
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
nltk.download('stopwords')

# Database model for storing interview data
class Interview(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    job_description = db.Column(db.Text, nullable=False)
    questions = db.Column(db.Text, nullable=False)
    transcription = db.Column(db.Text, nullable=True)
    score = db.Column(db.Float, nullable=True)
    status = db.Column(db.String(10), nullable=True)

# Create the database tables
with app.app_context():
    db.create_all()

def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file using PyMuPDF."""
    pdf_data = pdf_file.read()
    pdf = fitz.open(stream=pdf_data, filetype="pdf")
    text = ""
    for page_num in range(pdf.page_count):
        page = pdf.load_page(page_num)
        text += page.get_text()
    pdf.close()
    return text

def check_resume_fit(job_description, resume_text):
    job_doc = nlp(job_description)
    resume_doc = nlp(resume_text)

    # Calculate semantic similarity
    similarity_score = job_doc.similarity(resume_doc)

    print(f"Semantic Similarity Score: {similarity_score}")

    # Adjust threshold based on testing
    return similarity_score >= 0.8

def generate_questions(description):
    questions = []
    for question_type in ['technical', 'non-technical']:
        prompt = f"Based on the following job description:\n\n{description}\n\nGenerate one {question_type} interview question."
        result = generator(
            prompt, max_new_tokens=50, num_return_sequences=1,
            temperature=0.7, repetition_penalty=1.2, num_beams=5
        )[0]['generated_text']
        questions.append(result.strip())
    return questions

@app.route('/record_answer/<int:interview_id>', methods=['POST'])
def record_answer(interview_id):
    interview = Interview.query.get_or_404(interview_id)

    # Get the video data from the form
    video_data = request.form['video_data']

    # Analyze the video for confidence score
    confidence_score = analyze_video(video_data)

    # Transcribe the video into text
    transcription = transcribe_from_video(video_data)

    # Initialize NLP score to 0 by default
    nlp_score = 0.0

    # Check if transcription contains any meaningful data
    if transcription and transcription.strip() not in ["No speech detected or audio was unclear.", "API error"]:
        nlp_score = score_transcription(transcription, interview.job_description)

    # Calculate the final score with weighted average
    final_score = (0.4 * confidence_score) + (0.6 * nlp_score)
    status = "Hired" if final_score >= 0.5 else "Not Hired"

    # Update the interview record in the database
    interview.transcription = transcription
    interview.score = final_score
    interview.status = status
    db.session.commit()

    # Render the result page with detailed scores
    return render_template(
        'result.html',
        questions=interview.questions.split("\n"),
        transcription=transcription,
        confidence_score=confidence_score,
        nlp_score=nlp_score,
        final_score=final_score,
        status=status
    )


def analyze_video(video_data):
    """Analyze the video to compute a confidence score using facial expressions."""
    video_bytes = base64.b64decode(video_data)
    video_buffer = io.BytesIO(video_bytes)

    # Save video temporarily
    with open("temp_video.webm", "wb") as f:
        f.write(video_buffer.read())

    # Open video using OpenCV
    cap = cv2.VideoCapture("temp_video.webm")

    frame_count = 0
    emotion_scores = {"happy": 0, "neutral": 0, "angry": 0, "surprise": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # DeepFace.analyze() returns a list; access the first element
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)[0]
            dominant_emotion = result['dominant_emotion']

            if dominant_emotion in emotion_scores:
                emotion_scores[dominant_emotion] += 1  # Increment count for detected emotion

        except Exception as e:
            print(f"Error analyzing frame: {e}")

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    # Normalize emotion scores based on the number of frames
    for emotion in emotion_scores:
        emotion_scores[emotion] /= max(frame_count, 1)  # Avoid division by zero

    # Compute a confidence score (more weight to positive emotions)
    confidence_score = (0.6 * emotion_scores["happy"]) + (0.4 * emotion_scores["neutral"])

    return confidence_score

def transcribe_from_video(video_data):
    """Extract audio from video and transcribe it to text."""
    try:
        # Decode the video data from Base64
        video_bytes = base64.b64decode(video_data)
        video_buffer = io.BytesIO(video_bytes)

        # Extract audio from the video using pydub
        audio = AudioSegment.from_file(video_buffer, format="webm")

        # Export audio to WAV format
        wav_buffer = io.BytesIO()
        audio.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Use SpeechRecognition to transcribe the audio
        with sr.AudioFile(wav_buffer) as source:
            audio_content = recognizer.record(source)
            transcription = recognizer.recognize_google(audio_content)
            return transcription

    except sr.UnknownValueError:
        # Handle cases where speech is not detected
        return "No speech detected or audio was unclear."

    except sr.RequestError as e:
        # Handle API request errors
        return f"API error: {e}"

    except Exception as e:
        # Handle other unexpected errors
        print(f"Error during transcription: {e}")
        return "An unexpected error occurred during transcription."

def score_transcription(transcription, job_description):
    """Calculate the similarity score between the transcription and job description."""
    # Preprocess both the transcription and job description text
    transcription_clean = preprocess_text(transcription)
    job_description_clean = preprocess_text(job_description)

    # Use TF-IDF Vectorizer to transform the text data
    vectorizer = TfidfVectorizer().fit_transform([job_description_clean, transcription_clean])
    vectors = vectorizer.toarray()

    # Calculate the cosine similarity between the job description and transcription
    similarity_score = cosine_similarity([vectors[0]], [vectors[1]])[0][0]

    return similarity_score

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        job_description = request.form['job_description']
        resume = request.files['resume']

        if resume and resume.filename.endswith('.pdf'):
            resume_text = extract_text_from_pdf(resume)
            if check_resume_fit(job_description, resume_text):
                # Generate questions if resume fits the job description
                questions = generate_questions(job_description)
                questions_str = "\n".join(questions)

                new_interview = Interview(job_description=job_description, questions=questions_str)
                db.session.add(new_interview)
                db.session.commit()

                return render_template(
                    'questions.html',
                    questions=questions,
                    job_description=job_description,
                    interview_id=new_interview.id
                )
            else:
                return "<h3>You are not eligible for this job based on your resume.</h3>"
        else:
            return "<h3>Please upload a valid PDF resume.</h3>"

    return render_template('home.html')


if __name__ == "__main__":
    app.run(debug=True)
