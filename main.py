from video_retalking.voice_clone import VoiceCloner
from video_retalking.inference_module import run_inference
import sys


if __name__ == "__main__":
    run_inference(
        face='video_retalking/examples/face/7.mp4',
        audio_path='video_retalking/examples/audio/output_waved_class.wav',
        outfile='video_retalking/results/obama_7_2.mp4',
    )