from video_retalking.voice_clone import VoiceCloner
from video_retalking.inference_module import run_inference
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import sys

voice_cloner = VoiceCloner(speaker_dir="")


if __name__ == "__main__":
    audio = voice_cloner.generate_voice(voice_dir = "speakers/obama", output_path= "output.wav")
    run_inference(
        face='video_retalking/examples/face/7.mp4',
        audio_path='output.wav',
        outfile='video_retalking/results/obama_7_2.mp4',
    )