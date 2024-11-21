from video_retalking.voice_clone import VoiceCloner
from video_retalking.inference_module import LipSyncer
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
import sys

class AiVideoGenerator:
    def __init__(self, text =  "Hello, I have joined Aftr and I also invite you to join the platform") -> None:
        self.voice_cloner = VoiceCloner(speaker_dir="speakers/", text = text)
        self.lip_syncer = LipSyncer()
    
    def generate_template_vide(self, face_video, audio_directory):
        audio = self.voice_cloner.generate_voice(voice_dir = audio_directory)
        print("################################################################")
        print(audio)
        outfile = self.lip_syncer.inference(
        face_path=face_video,
        audio_path=audio
    )
        return outfile






if __name__ == "__main__":
    
    ai_generator = AiVideoGenerator()
    outfile = ai_generator.generate_template_vide(face_video=r"speakers/obama/2.mp4", audio_directory="obama")
    print("Output ready at: ", outfile )