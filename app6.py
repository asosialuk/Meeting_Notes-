#Libraries import
import streamlit as st
import whisper
import openai
import tempfile
import os
from fpdf import FPDF
from pyannote.audio import Pipeline
from huggingface_hub import login
from pydub import AudioSegment 

# Load pyannote model 
def load_diarization_pipeline():
    HUGGINGFACE_TOKEN = st.session_state.get("huggingface_token")  
    if not HUGGINGFACE_TOKEN:
        st.warning("Please enter your Hugging Face token.")
        return None

    try:
        login(token=HUGGINGFACE_TOKEN)
        return Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=True)
    except Exception as e:
        st.error(f"Failed to load speaker diarization model: {e}")
        return None

# Load Whisper model
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

def convert_to_wav(uploaded_file):
    """Convert uploaded audio file to WAV format."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file_name = temp_file.name
    
    # Convert to WAV using pydub
    audio = AudioSegment.from_file(uploaded_file)
    audio.export(temp_file_name, format="wav")
    return temp_file_name

def transcribe_audio(uploaded_file):
    """Audio transcribe using Whisper with speaker diarization."""
    # Convert the uploaded file to WAV format using ffmpeg
    temp_file_name = convert_to_wav(uploaded_file)
    if temp_file_name is None:
        return "Failed to convert audio file.", "Unknown"
    
    whisper_model = load_whisper_model()
    transcription = whisper_model.transcribe(temp_file_name, fp16=False)

    # Run speaker diarization 
    diarization_pipeline = load_diarization_pipeline()
    if diarization_pipeline is None:
        st.error("Speaker diarization failed to load.")
        os.remove(temp_file_name)
        return transcription["text"], transcription["language"]
    
    diarization = diarization_pipeline(temp_file_name)
    
    # Assign speaker labels
    transcript_with_speakers = ""
    speaker_counter = 1  # Counter for assigning Speaker 1, Speaker 2, etc.
    speaker_mapping = {}  # Maps pyannote speaker labels to readable labels (Speaker 1, Speaker 2, etc.)

    # Group Whisper segments by speaker
    grouped_segments = []
    current_speaker = None
    current_start = None
    current_end = None
    current_text = ""

    for segment in transcription["segments"]:
        segment_start = segment["start"]
        segment_end = segment["end"]
        segment_text = segment["text"].strip()
        
        # Find overlapping diarization segments
        speaker_label = "Unknown"
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if turn.start <= segment_start and turn.end >= segment_end:
                # Map the pyannote speaker label to a readable label
                if speaker not in speaker_mapping:
                    speaker_mapping[speaker] = f"Speaker {speaker_counter}"
                    speaker_counter += 1
                speaker_label = speaker_mapping[speaker]
                break
        
        # If no speaker is found, assign the segment to the previous speaker (if any)
        if speaker_label == "Unknown" and current_speaker is not None:
            speaker_label = current_speaker
        
        # If the speaker is the same as the previous segment, merge them
        if speaker_label == current_speaker:
            current_text += " " + segment_text
            current_end = segment_end
        else:
            # Save the previous segment if it exists
            if current_speaker is not None:
                grouped_segments.append({
                    "speaker": current_speaker,
                    "start": current_start,
                    "end": current_end,
                    "text": current_text.strip()
                })
            # Start a new segment
            current_speaker = speaker_label
            current_start = segment_start
            current_end = segment_end
            current_text = segment_text
    
    # Add the last segment
    if current_speaker is not None:
        grouped_segments.append({
            "speaker": current_speaker,
            "start": current_start,
            "end": current_end,
            "text": current_text.strip()
        })
    
    # Build the final transcript with grouped speaker blocks
    for segment in grouped_segments:
        transcript_with_speakers += f"{segment['speaker']}: {segment['text']} {segment['start']:.2f} - {segment['end']:.2f}\n"
    
    os.remove(temp_file_name)
    return transcript_with_speakers, transcription["language"]
    
def process_transcript(transcript_text, language):

    openai_api_key = st.session_state.get("openai_token")  
    if not openai_api_key:
        st.warning("Please enter your OpenAI token.")
        return "OpenAI token is missing."
    openai.api_key = openai_api_key

    #Generate structured meeting notes using OpenAI API
    prompt = f"""
    Generate structured meeting notes from the following transcript exclusively in the {language} language.  Do not include any text in other languages.
    Transcript:
    {transcript_text}
    
    1. **Main Objective**:
    2. **Highlighted Points**:
    3. **Followed Actions from Last Time **:
    4. **Discussion Details**:
    5. **Problems**:
    6. **Actions Assigned to Each Person**:
    7. **Date of the Next Meeting**:
    """
    try: 
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", 
            messages=[{"role": "system", "content": "You are a helpful assistant for extracting structured notes."},
                      {"role": "user", "content": prompt}]
        )
        return response["choices"][0]["message"]["content"]
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"

# Generate pdf
def generate_pdf(notes): 
    """Generate a PDF file from the generated meeting notes"""
    pdf = FPDF()   
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", style="B", size=12)
    pdf.cell(200, 10, "Meeting Summary", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, notes)

    pdf_path = "meeting_notes.pdf"
    pdf.output(pdf_path)
    return pdf_path

def main():
    st.title("Meeting Notes Generator")
    # Token input boxes 
    st.sidebar.title("API Tokens")
    st.sidebar.markdown("To use this application, you need to provide API tokens for Hugging Face and OpenAI.  Here's how to generate them:")

    st.sidebar.subheader("HuggingFace Token")
    st.sidebar.markdown("1. Go to your [Hugging Face settings](https://huggingface.co/settings/tokens).")
    st.sidebar.markdown("2. Create a new access token with 'write' permissions.")
    st.sidebar.markdown("3. Copy the generated token and paste it below:")
    st.session_state["huggingface_token"] = st.sidebar.text_input("Hugging Face Token", type="password")


    st.sidebar.subheader("OpenAI Token")
    st.sidebar.markdown("1. Go to your [OpenAI API keys](https://platform.openai.com/account/api-keys).")
    st.sidebar.markdown("2. Create a new secret key.")
    st.sidebar.markdown("3. Copy the generated key and paste it below:")
    st.session_state["openai_token"] = st.sidebar.text_input("OpenAI Token", type="password")

    #Upload the audio file 
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "mp4"])
    
    if "transcript_text" not in st.session_state:
        st.session_state["transcript_text"] = ""

    if "meeting_notes" not in st.session_state:
        st.session_state["meeting_notes"] = ""

    if "language" not in st.session_state:
        st.session_state["language"] = ""
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")
        # Audio transcription
        if st.button("Transcribe Audio"):
            with st.spinner("Transcribing..."):
                transcript, language = transcribe_audio(uploaded_file)
                st.session_state["transcript_text"] = transcript
                st.session_state["language"] = language 
            st.success(f"Transcription complete! Detected language: {language}")
            st.text_area("Transcript", transcript, height=200)

    # Generate the notes 
    if st.session_state["transcript_text"]:
        if st.button("Generate Meeting Notes"):
            with st.spinner("Generating notes..."):
                st.session_state["meeting_notes"] = process_transcript(st.session_state["transcript_text"], st.session_state["language"])
            
            st.subheader("Meeting Summary")
            st.write(st.session_state["meeting_notes"])
    # Download options: txt or pdf  
    if st.session_state['meeting_notes']:
        st.download_button(
            label="Download Notes", 
            data=st.session_state["meeting_notes"], 
            file_name="meeting_notes.txt",
            mime="text/plain"
        )
        pdf_path = generate_pdf(st.session_state["meeting_notes"])
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="Download Notes as PDF",
                data=pdf_file,
                file_name="meeting_notes.pdf",
                mime="application/pdf")

if __name__ == "__main__":
    main()