import os
import re
import requests
import openai
from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def extract_youtube_video_id(url):
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        return None


def get_youtube_video_title(video_id):
    url = f"https://www.youtube.com/watch?v={video_id}"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    title_element = soup.find("meta", itemprop="name")
    if title_element:
        return title_element["content"]
    else:
        return None


def get_youtube_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return transcript
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def summarize_youtube_transcript(text):
    conversation = [
        {'role': 'system', 'content': 'Summarize the YouTube transcript in bullet points, highlighting key insights:'}]
    max_chunk_size = 2048 - len(conversation[0]['content']) - 100
    chunks = split_text_into_chunks(text, max_chunk_size)
    summarized_chunks = []

    total_tokens_used = 0

    for chunk in chunks:
        conversation.append({'role': 'user', 'content': chunk})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=conversation,
            max_tokens=100
        )
        api_usage = response['usage']
        print('Total token consumed: {0}'.format(api_usage['total_tokens']))

        summary = response.choices[0].message.content.strip()
        summarized_chunks.append(summary)

    return " ".join(summarized_chunks)


def split_text_into_chunks(text, max_chunk_size):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(" ".join(current_chunk)) > max_chunk_size:
            current_chunk.pop()
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

