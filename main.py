from video_retalking.voice_clone import VoiceCloner
from video_retalking.inference_module import run_inference
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import sys

voice_cloner = VoiceCloner(speaker_dir="")


if __name__ == "__main__":
    audio = voice_cloner.generate_voice(voice_dir = "speakers\obama")
    outfile = run_inference(
        face='speakers\obama\2.mp4',
        audio_path=audio
    )
    print("Output ready at: ",outfile )