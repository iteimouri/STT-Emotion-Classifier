import streamlit as st
import librosa
import sounddevice as sd
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, pipeline
import pandas as pd

@st.cache(allow_output_mutation=True, suppress_st_warning=True, show_spinner=True)

#load model and tokenizer
def load_model():
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
    return model

def load_tokenizer():
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    return tokenizer

def load_classifier():
    classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
    return classifier

model = load_model()
tokenizer = load_tokenizer()
classifier = load_classifier()

st.header("Speech to Text Emotion Classifier 🚀")
st.caption("Implemented by Ilia Teimouri")
st.write("This app records your voice and then pass it to a ROBERTA based sentiment classifier.\n In return you will get a table of your probable emotions!")

st.subheader("Ready to try it on your voice?")
duration = st.slider("Recording duration", 3, 20, 5)

#st.sidebar.title("Parameters")
#duration = st.sidebar.slider("Recording duration", 3, 20, 5)
#st.sidebar.text("Implemented by Ilia Teimouri")


def record(sr=16000, channels=1, duration=3, filename='record.wav'):
    recording = sd.rec(int(sr * duration), samplerate=sr, channels=channels, blocking=False)
    sd.wait()
    sf.write(filename, recording, sr)
    samples, sample_rate = librosa.load(filename , sr = sr)
    return samples

def transcription(test_record):
    input_values = tokenizer(test_record, return_tensors = 'pt').input_values
    #Store logits (non-normalized predictions)
    logits = model(input_values).logits
    #Store predicted id's
    predicted_ids = torch.argmax(logits, dim =-1)
    #decode the audio to generate text
    transcriptions = tokenizer.decode(predicted_ids[0])
    return transcriptions

def analyse(transcriptions):
    classed = classifier(transcriptions)
    df = pd.DataFrame(classed[0]).sort_values(by=['score'], ascending=False)
    return df



if st.button("Start Recording"):
    with st.spinner("Recording..."):
        test_record = record(duration=duration)
        st.write("Transcription:")
        st.success(transcription(test_record))
        st.dataframe(analyse(transcription(test_record)).style.highlight_max(axis=0))