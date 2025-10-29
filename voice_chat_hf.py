import sounddevice as sd
import numpy as np
import wave
import pyttsx3
from faster_whisper import WhisperModel
from transformers import pipeline


# --- AUDIO RECORDING ---
def record_audio(filename="input.wav", duration=6, samplerate=16000):
    print("🎙️ Speak now...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(samplerate)
        wf.writeframes(audio.tobytes())
    return filename


# --- SPEECH TO TEXT ---
def transcribe_audio(filename):
    print("🔍 Transcribing...")
    model = WhisperModel("medium", device="cuda")
    segments, _ = model.transcribe(filename)
    text = " ".join([seg.text for seg in segments])
    print(f"🗣️ You said: {text}")
    return text.strip()


# --- AI RESPONSE (Hugging Face Local Model) ---
print("🧩 Loading Mistral model (first run may take time)...")
chatbot = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype="auto",
    device_map="auto"
)

def get_ai_response(prompt):
    print("🤖 Thinking...")
    result = chatbot(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
    reply = result[0]["generated_text"].split(prompt)[-1].strip()
    print(f"🤖 AI: {reply}")
    return reply


# --- TEXT TO SPEECH ---
def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 180)
    engine.say(text)
    engine.runAndWait()


# --- MAIN LOOP ---
def voice_chat_loop():
    print("🧠 Voice Chat (say 'exit' to quit)\n")
    while True:
        audio = record_audio()
        user_text = transcribe_audio(audio)
        if not user_text:
            continue
        if "exit" in user_text.lower():
            print("👋 Goodbye!")
            break
        ai_text = get_ai_response(user_text)
        speak_text(ai_text)


if __name__ == "__main__":
    voice_chat_loop()
