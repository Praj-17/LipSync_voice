from video_retalking.voice_clone import VoiceCloner
from video_retalking.inference_module import LipSyncer
import warnings
import os
# Suppress all warnings
warnings.filterwarnings("ignore")
import sys
import requests
import mimetypes
import re

class AiVideoGenerator:
    def __init__(self, text =  "Hello, I have joined Aftr and I also invite you to join the platform", speaker_dir = "speakers/") -> None:
        self.voice_cloner = VoiceCloner(speaker_dir=speaker_dir, text = text)
        self.lip_syncer = LipSyncer()
        self.speaker_dir = speaker_dir
    def create_default_folder(self, user_id):
            folder_path = os.path.join(self.speaker_dir, user_id)
            os.makedirs(folder_path,  exist_ok=True)
            return folder_path
    
    def process_video_url(self, url, base_folder, user_id):
        video_name = os.path.join(base_folder, f"{user_id}.mp4")
        face_video = self.download_video(url,video_name)
        return face_video

    def generate_video_local(self, face_video, audio_directory):

        audio = self.voice_cloner.generate_voice(voice_dir = audio_directory)

        outfile = self.lip_syncer.inference(
            face_path=face_video,
            audio_path=audio
        )
        return outfile
        

    
    def generate_template_video(self,user_id,  face_video, audio_url_list):
            base_folder = self.create_default_folder(user_id)
            if self.is_valid_url(face_video):
                face_video = self.process_video_url(url= face_video, base_folder=base_folder, user_id=user_id)
            
            self.download_audio_files(audio_url_list, base_folder)


            audio = self.voice_cloner.generate_voice(voice_dir = base_folder)
            outfile = self.lip_syncer.inference(
            face_path=face_video,
            audio_path=audio
        )
            return outfile
    def is_valid_url(self, url):
        regex = re.compile(
            r'^(?:http|ftp)s?://'        # http://, https://, ftp://, or ftps://
            r'(?:\S+(?::\S*)?@)?'        # Optional username:password@
            r'(?:'                       # Start of group for domain name or IP
            r'(?P<ip>(?:\d{1,3}\.){3}\d{1,3})'  # IP address
            r'|'                         # or
            r'(?P<hostname>'             # Hostname
            r'(?:(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,})'
            r')'
            r')'
            r'(?::\d{2,5})?'             # Optional port
            r'(?:[/?#]\S*)?'             # Optional path, query, or fragment
            r'$', re.IGNORECASE)

        return re.match(regex, url) is not None
    
    def download_video(self, url, filename):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Check for HTTP errors

            with open(filename, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # Filter out keep-alive chunks
                        f.write(chunk)

            print(f"Video downloaded successfully and saved as '{filename}'.")
        except requests.exceptions.HTTPError as err:
            print(f"HTTP error occurred: {err}")
        except requests.exceptions.RequestException as err:
            print(f"Error occurred during requests to {url}: {err}")

    def download_audio_files(self, url_list, save_folder):
        # Ensure the save folder exists
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        for index, url in enumerate(url_list, start=1):
            try:
                # Send a HEAD request to get content type and disposition
                head = requests.head(url, allow_redirects=True)
                content_type = head.headers.get('content-type')
                content_disposition = head.headers.get('content-disposition')

                # Determine the file extension
                if content_disposition:
                    # Try to get filename from content-disposition
                    filename = content_disposition.split('filename=')[-1].strip('"\'')
                    extension = os.path.splitext(filename)[1]
                elif content_type:
                    # Guess extension from content type
                    extension = mimetypes.guess_extension(content_type.split(';')[0])
                else:
                    # Fallback to .mp3 if content type is unknown
                    extension = '.mp3'

                # Handle cases where extension is None
                if not extension:
                    extension = '.mp3'

                # Construct the filename as a number with extension
                filename = f"{index}{extension}"

                # Full path to save the file
                file_path = os.path.join(save_folder, filename)

                # Download the file
                response = requests.get(url, stream=True)
                response.raise_for_status()  # Check for HTTP errors

                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print(f"Downloaded '{filename}' successfully.")


            except requests.exceptions.HTTPError as err:
                print(f"HTTP error occurred while downloading '{url}': {err}")
            except requests.exceptions.RequestException as err:
                print(f"Error occurred while downloading '{url}': {err}")
            except Exception as err:
                print(f"An unexpected error occurred with '{url}': {err}")

if __name__ == "__main__":

    
    ai_generator = AiVideoGenerator()
    outfile = ai_generator.generate_video_local(face_video=r"speakers\obama\obama.mp4", audio_directory=r"obama")
    print("Output ready at: ", outfile )