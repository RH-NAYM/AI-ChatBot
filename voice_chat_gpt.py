import os
import sounddevice as sd
import wave
import torch
from faster_whisper import WhisperModel
from TTS.api import TTS

# -----------------------------
# 🧠 Suppress LLaMA Logs
# -----------------------------
os.environ["LLAMA_CPP_LOG_LEVEL"] = "error"  # hide perf logs

try:
    from llama_cpp import Llama
except ImportError:
    print("Installing llama-cpp-python...")
    os.system("pip install llama-cpp-python")
    from llama_cpp import Llama

# -----------------------------
# LLaMA Model Setup
# -----------------------------
MODEL_NAME = "llama-2-7b-chat.Q4_K_M.gguf"
MODEL_PATH = f"models/{MODEL_NAME}"

if not os.path.exists(MODEL_PATH):
    print(f"⬇ Downloading LLaMA 2 7B model ({MODEL_NAME})...")
    os.makedirs("models", exist_ok=True)
    hf_token = os.getenv("HF_TOKEN", "my hf_ token")
    download_cmd = f"curl -L -H 'Authorization: Bearer {hf_token}' -o {MODEL_PATH} " \
                   f"https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF-f16/resolve/main/{MODEL_NAME}"
    os.system(download_cmd)

print("🔧 Initializing LLaMA model...")
try:
    llm = Llama(model_path=MODEL_PATH, n_threads=8, n_gpu_layers=32, gpu_index=0)
except Exception as e:
    print(f"⚠ GPU load failed, falling back to CPU: {e}")
    llm = Llama(model_path=MODEL_PATH, n_threads=8)

# -----------------------------
# Audio Input Configuration
# -----------------------------
def get_input_device():
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_input_channels'] > 0:
            return i
    raise RuntimeError("No input device found")

try:
    MIC_DEVICE_INDEX = 10
    SAMPLE_RATE = int(sd.query_devices(MIC_DEVICE_INDEX, 'input')['default_samplerate'])
except ValueError:
    MIC_DEVICE_INDEX = get_input_device()
    SAMPLE_RATE = int(sd.query_devices(MIC_DEVICE_INDEX, 'input')['default_samplerate'])

CHANNELS = 1
FILENAME = "input.wav"
print(f"🎤 Using mic: {sd.query_devices(MIC_DEVICE_INDEX)['name']} ({MIC_DEVICE_INDEX}) at {SAMPLE_RATE} Hz")

# -----------------------------
# Text-to-Speech (TTS) Setup
# -----------------------------
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=torch.cuda.is_available())

def speak(text):
    tts.tts_to_file(text=text, file_path="ai_speech.wav")
    # play audio using sounddevice
    import soundfile as sf
    data, fs = sf.read("ai_speech.wav", dtype='float32')
    sd.play(data, fs)
    sd.wait()

# -----------------------------
# Whisper Setup
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🧠 Loading Whisper model on {device}...")
whisper_model = WhisperModel("large-v3", device=device, compute_type="float16" if device=="cuda" else "default")

# -----------------------------
# Record Audio
# -----------------------------
def record_audio(filename=FILENAME, duration=3):
    print(f"🎙 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS,
                   dtype='int16', device=MIC_DEVICE_INDEX)
    sd.wait()
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return filename

# -----------------------------
# Speech-to-Text
# -----------------------------
def transcribe_audio(filename):
    segments, _ = whisper_model.transcribe(filename, language="en")
    return " ".join([seg.text for seg in segments]).strip()

# -----------------------------
# Get AI Response
# -----------------------------
def get_ai_response(user_text):
    prompt = f"You are a helpful AI assistant. The user said: {user_text}"
    response = llm(prompt, max_tokens=256)
    return response["choices"][0]["text"].strip()

# -----------------------------
# Voice Assistant Loop
# -----------------------------
def voice_assistant():
    print("🧠 Voice Assistant Ready! (say 'exit' to quit)\n")
    while True:
        try:
            audio_file = record_audio(duration=5)
            user_text = transcribe_audio(audio_file)

            if not user_text:
                continue

            print(f"🗣 You: {user_text}")

            if "exit" in user_text.lower():
                speak("Goodbye!")
                break

            ai_reply = get_ai_response(user_text)
            print(f"🤖 AI: {ai_reply}")
            speak(ai_reply)

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user.")
            break
        except Exception as e:
            print(f"⚠ Error: {e}")

# -----------------------------
if __name__ == "__main__":
    voice_assistant()
