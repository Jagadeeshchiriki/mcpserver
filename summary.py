

from mcp.server.fastmcp import FastMCP
import os
from faster_whisper import WhisperModel
from pydub import AudioSegment
import speech_recognition as sr
import tempfile
import requests
import json
from dotenv import load_dotenv

load_dotenv()


# --- Create MCP server ---
mcp = FastMCP("summary")




@mcp.tool()
async def get_summary(file_path: str) -> str:
    """Transcribe and summarize audio/video file using Whisper + DeepSeek"""

    def save_uploaded_file(input_path):
        try:
            with open(input_path, "rb") as src:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.' + input_path.split('.')[-1]) as tmp_file:
                    tmp_file.write(src.read())
                    return tmp_file.name
        except Exception as e:
            print(f"[ERROR] Saving file: {e}")
            return None

    def convert_to_wav(input_file):
        try:
            output_file = os.path.join(tempfile.gettempdir(), "converted_audio.wav")
            if input_file.lower().endswith('.mp3'):
                audio = AudioSegment.from_mp3(input_file)
            elif input_file.lower().endswith(('.mp4', '.avi', '.mov')):
                audio = AudioSegment.from_file(input_file)
            elif input_file.lower().endswith('.wav'):
                audio = AudioSegment.from_wav(input_file)
            else:
                audio = AudioSegment.from_file(input_file)
            audio = audio.set_channels(1).set_frame_rate(16000).normalize()
            audio.export(output_file, format="wav")
            return output_file
        except Exception as e:
            print(f"[ERROR] Converting file: {e}")
            return None

    def transcribe_audio(audio_file):
        model = WhisperModel("base", compute_type="int8")  # Use "int8" for CPU efficiency

        segments, info = model.transcribe(audio_file)

        print("Detected language:", info.language)

        transcription = ""
        for segment in segments:
            transcription += segment.text + " "

        # print("Transcription:", transcription)
        return transcription
    
    # Main logic
    if not os.path.exists(file_path):
        return "❌ File not found."

    temp_file = save_uploaded_file(file_path)
    if not temp_file:
        return "❌ Failed to save uploaded file."

    wav_file = convert_to_wav(temp_file)
    if not wav_file:
        return "❌ Failed to convert file to WAV format."

    transcript = transcribe_audio(wav_file)
    if not transcript:
        return "❌ Transcription failed."

    # Cleanup
    for f in [temp_file, wav_file]:
        if f and os.path.exists(f):
            os.unlink(f)
    print(transcript)

    return transcript

if __name__=="__main__":
    mcp.run(transport="streamable-http")
