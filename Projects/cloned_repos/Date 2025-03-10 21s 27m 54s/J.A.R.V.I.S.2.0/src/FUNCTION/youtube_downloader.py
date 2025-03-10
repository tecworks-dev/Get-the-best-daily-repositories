
import re 
from pytube import YouTube
from os import getcwd , mkdir , system
from os.path import join , exists
from src.FUNCTION.get_env import load_variable
from src.CONVERSATION.text_to_speech import speak

def extract_id(url:str) -> str:
    """ Extracts the video ID from the YouTube URL. """
    # Regular expression for extracting the video ID from a YouTube URL
    regex = r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/" \
        r"|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"

    match = re.search(regex, url)
    if not match:
        raise ValueError("Invalid YouTube URL")
    return match.group(1)



def yt_download() -> str:
    """Download a YouTube video."""
    speak("please Enter the download url.")
    link = input("Paste Link:")
    save_path = load_variable("Yt_path")
    # Extract video ID from URL
    video_id = extract_id(link)

    # Create filename
    file_name = f"{video_id}.mp4"

    # Download video
    try:
        # Create save path if it doesn't exist
        if not exists(save_path):
            mkdir(save_path)
        # Get the highest resolution video stream
        video = YouTube(link)
        highest_resolution_stream = video.streams.order_by('resolution').desc().first()
        # Download the video
        highest_resolution_stream.download(filename= join(save_path, file_name))

        print(f"Downloaded video: {file_name}")
    except Exception as e:
        print(f"Error downloading video: {e}")

    return file_name




#incog_mode("https://www.google.com")