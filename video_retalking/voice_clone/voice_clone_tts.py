
from tortoise import api_fast, utils
import numpy as np
from scipy.io.wavfile import write
import os

class VoiceCloner:
    def __init__(self, speaker_dir = "speakers") -> None:
        self.tts = api_fast.TextToSpeech(kv_cache=True, half=True)
        self.aftr_template = "Hello, I have joined Aftr and I also invite you to join the platform"
        self.speaker_dir = speaker_dir

    def save_wav_file(self, tensor, file_path, sample_rate = 22050):
        """
        Save a tensor as a WAV file using scipy.

        Args:
            tensor (Tensor): The audio tensor (shape: [num_samples] or [channels, num_samples]).
            sample_rate (int): The sample rate of the audio.
            file_path (str): Path to save the WAV file.
        """
        # Convert the tensor to a NumPy array
        audio_np = tensor.detach().cpu().numpy()

        # Scale to int16 range for WAV files
        audio_np = (audio_np * 32767).astype(np.int16)

        # Save as a WAV file
        write(file_path, sample_rate, audio_np)
        return file_path
    def generate_voice(self, voice_dir, output_path):
        voice_dir = os.path.join(self.speaker_dir, voice_dir)
        clips_paths = [os.path.join(voice_dir, i) for i in os.listdir(voice_dir) if i.endswith(".mp3") or i.endswith(".wav")]
        reference_clips = [utils.audio.load_audio(p, 22050) for p in clips_paths]

       
        pcm_audio = self.tts.tts(self.aftr_template, voice_samples=reference_clips, verbose=False)

        return self.save_wav_file(pcm_audio, output_path)


if __name__ == "__main__":

    vc = VoiceCloner()
    vc.generate_voice("obama", "output_waved_class.wav")