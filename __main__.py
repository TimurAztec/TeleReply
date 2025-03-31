import base64
import io
import os
import re
import random
import subprocess
from pprint import pprint

import ffmpeg
import tiktoken
from PIL import Image
from datetime import datetime, timedelta
import openai
import asyncio
from openai import AsyncOpenAI
from telethon import TelegramClient, events
from collections import defaultdict
from dotenv import load_dotenv
from telethon.tl.functions.account import GetAuthorizationsRequest
from telethon.tl.functions.messages import SetTypingRequest, GetStickerSetRequest
from telethon.tl.types import SendMessageTypingAction, SendMessageRecordAudioAction, DocumentAttributeAudio, \
    InputStickerSetShortName

from youtube import extract_youtube_video_id, get_youtube_video_title, summarize_youtube_transcript, \
    get_youtube_transcript
whisper = None
# try:
#     import whisper
#     whisper = whisper
# except ImportError as e:
#     print("Could not import whisper module!")

EMOJI_REGEX = re.compile(
    r'^[\U0001F3FB-\U0001F3FF]?[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+$'
)

load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_WHITE_LIST = os.getenv("CHAT_WHITE_LIST").split(',')
SYS_PROMPT = os.getenv("SYS_PROMPT")
client = TelegramClient("session", int(API_ID), API_HASH)
me = None
openai_client = AsyncOpenAI(
  api_key=OPENAI_API_KEY
)

reply_enabled = True
busy_replying = defaultdict(lambda: False)
chats_history = defaultdict(list)

NUM_PREVIOUS_MESSAGES = 10
TYPING_SPEED = 10
SPEECH_SPEED = 15
temperature=1.011
presence_penalty=0.33
frequency_penalty=1
top_p=0.5
model_id="ft:gpt-4o-mini-2024-07-18:personal:timur:B6C081Io:ckpt-step-946"

@client.on(events.NewMessage(incoming=False))
async def process_out_message(event):
    global reply_enabled, me, temperature, presence_penalty, frequency_penalty, top_p
    sender_id = event.chat_id if event.is_group else event.sender_id
    print(f"Out | Chat id: {event.chat_id} | Text: {event.text}")
    if event.chat_id == me.id:
        if event.text.lower() == "reply-on" and not reply_enabled:
            reply_enabled = True
            await event.reply("✅ Auto-reply is ON.")
        elif event.text.lower() == "reply-off" and reply_enabled:
            reply_enabled = False
            await event.reply("❌ Auto-reply is OFF.")

        param_match = re.match(r'set-(temperature|top_p|presence_penalty|frequency_penalty):\s*([0-9]*\.?[0-9]+)',
                               event.text, re.IGNORECASE)
        if param_match:
            param_name = param_match.group(1)
            param_value = float(param_match.group(2))

            if param_name == "temperature":
                temperature = param_value
            elif param_name == "top_p":
                top_p = param_value
            elif param_name == "presence_penalty":
                presence_penalty = param_value
            elif param_name == "frequency_penalty":
                frequency_penalty = param_value

            await event.reply(f"✅ {param_name} set to {param_value}")
            return
    elif event.chat_id != me.id:
        if "@TimurWasHere" in event.text:
            if str(event.chat_id) in CHAT_WHITE_LIST:
                await handle_message(event)
                return

        if event.text.lower() == "reply-add" and not str(event.chat_id) in CHAT_WHITE_LIST:
            CHAT_WHITE_LIST.append(str(event.chat_id))
            await event.reply(f"Chat added: {event.chat_id}")
        elif event.text.lower() == "reply-remove" and str(event.chat_id) in CHAT_WHITE_LIST:
            CHAT_WHITE_LIST.remove(str(event.chat_id))
            await event.reply(f"Chat removed: {event.chat_id}")
        else:
            if event.text:
                chats_history[sender_id].append({"role": "assistant", "content": event.text})

async def respond_voice(event, text):
    user_id = event.chat_id

    await simulate_voice_recording(event, text)

    response = openai.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )

    raw_audio = io.BytesIO(response.content)
    raw_audio.seek(0)

    converted_audio = io.BytesIO()

    process = subprocess.run(
        [
            "ffmpeg", "-y", "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", "32k", "-vn",
            "-f", "ogg", "pipe:1"
        ],
        input=raw_audio.read(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if process.returncode != 0:
        print("FFmpeg error:", process.stderr.decode())
        return

    converted_audio.write(process.stdout)
    converted_audio.seek(0)
    converted_audio.name = "voice.ogg"

    if converted_audio.getbuffer().nbytes == 0:
        print("Converted audio file is empty!")
        return

    duration = round(len(text) / SPEECH_SPEED)

    await client.send_file(
        user_id,
        converted_audio,
        voice_note=True,
        reply_to=event.message,
        attributes=[DocumentAttributeAudio(duration=int(duration), voice=True)]
    )

@client.on(events.NewMessage(incoming=True))
async def process_in_message(event):
    print(f"Incoming | Chat id: {event.chat_id} | Text: {event.text}")
    
    # if str(event.chat_id) not in CHAT_WHITE_LIST or event.chat_id == me.id or (event.text == '' and not (event.photo or event.document or event.voice)):
    if str(event.chat_id) not in CHAT_WHITE_LIST or (event.text == '' and not (event.photo or event.document or event.voice)):
        return

    await handle_message(event)


async def handle_message(event):
    global me, reply_enabled, busy_replying, temperature, frequency_penalty, presence_penalty, top_p
    sender = await event.get_sender()

    sender_id = event.chat_id if event.is_group else event.sender_id
    if not chats_history[sender_id]:
        previous_messages = await client.get_messages(event.chat_id, limit=round(NUM_PREVIOUS_MESSAGES))
        for msg in previous_messages:
            if msg.from_id and msg.from_id.user_id != me.id:
                chats_history[sender_id].append({"role": "user", "content": msg.text})
            if msg.from_id and msg.from_id.user_id == me.id:
                chats_history[sender_id].append({"role": "assistant", "content": msg.text})


    if busy_replying[sender_id]:
        print(f"Busy!")
        return

    active = await check_active_sessions()
    if not reply_enabled or active:
        return

    await event.mark_read()
    busy_replying[sender_id] = True
    try:
        system_message = {
            "role": "system",
            "content": SYS_PROMPT
        }
        history = chats_history[sender_id][-NUM_PREVIOUS_MESSAGES:]
        await summarize_history(sender_id)
        history.insert(0, system_message)
        content_list = []

        username = await get_display_name(sender)
        if event.text:
            text = event.text
            youtube_id = extract_youtube_video_id(event.text)
            if youtube_id:
                youtube_title = get_youtube_video_title(youtube_id)
                youtube_summary = get_youtube_transcript(youtube_id)
                text += f"\n User attached video titled {youtube_title}: {youtube_summary}"
            content_list.append({"type": "text", "text": f'{sender.username} says: {text}' if username else text})

        image_base64 = None

        if event.photo or event.document:
            mime_type = getattr(event.document, "mime_type", None)
            print(f"Document: {mime_type}")

            if mime_type in ["image/gif", "image/webp", "application/x-tgsticker"]:
                blob = await event.download_media(bytes)
                image_base64 = await convert_to_jpeg(blob)

            elif mime_type in ["video/webm", "video/mp4"]:
                blob = await event.download_media(bytes)
                out, err = (
                    ffmpeg.input("pipe:0", format="mp4")  # Specify format
                    .output("pipe:1", vframes=1, format="image2", vcodec="mjpeg")
                    .run(input=blob, capture_stdout=True, capture_stderr=True)
                )
                if err:
                    print("FFmpeg error:", err)

                image_base64 = base64.b64encode(out).decode("utf-8")

            elif mime_type == "audio/ogg":
                if whisper:
                    audio_file = await event.download_media()
                    model = whisper.load_model("base")
                    transcribed_text = model.transcribe(audio_file)
                    print(f"Audio transcription: {transcribed_text.get('text')}")
                    history.append({"role": "user", "content": transcribed_text.get('text')})
                    os.remove(audio_file)
                else:
                    content_list.append(
                        {"type": "text", "text": "*User attached voice message, but you cant listen to it at the moment*"})

            else:
                blob = await event.download_media(bytes)
                image_base64 = base64.b64encode(blob).decode("utf-8")

        if image_base64:
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}
            })

        if content_list:
            history.append({"role": "user", "content": content_list})

        mention = await check_mention(me, sender_id, event)
        print(f"Mentioned: {mention}")
        if event.is_group and not mention:
            print(f"Not mentioned")
            return

        await respond(first_msg=True, event=event, history=history)

    except openai._exceptions.RateLimitError:
        print("Quota limit exceeded or rate limit error")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    finally:
        busy_replying[sender_id] = False

async def summarize_history(sender_id):
    if len(chats_history[sender_id]) < NUM_PREVIOUS_MESSAGES:
        return

    summary_prompt = {
        "role": "system",
        "content": "Summarize this conversation while keeping key details relevant to the discussion."
    }

    history_text = "\n".join([
        f'{msg["role"].capitalize()}: {msg["content"]}' if isinstance(msg["content"], str)
        else f'{msg["role"].capitalize()}: {str(msg["content"])}'
        for msg in chats_history[sender_id][-NUM_PREVIOUS_MESSAGES:]
    ])

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[summary_prompt, {"role": "user", "content": history_text}],
            max_tokens=200,
            temperature=0.33
        )
        summary = response.choices[0].message.content.strip()
        print(f"History summary: {summary}")
        chats_history[sender_id] = [{"role": "system", "content": "Previous conversation Summary: " + summary}]

    except openai._exceptions.RateLimitError:
        print("Rate limit exceeded, skipping history summarization.")
    except Exception as e:
        print(f"Error summarizing history: {e}")

async def convert_to_jpeg(blob):
    image = Image.open(io.BytesIO(blob))
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

async def generate_response(history):
    global model_id
    last_message = history[-1] if history else None

    if last_message and isinstance(last_message.get("content"), list):
        has_image = any(
            isinstance(msg, dict) and msg.get("type") == "image_url"
            for msg in last_message["content"]
        )

        text_content = next(
            (msg["text"] for msg in last_message["content"]
             if isinstance(msg, dict) and msg.get("type") == "text"),
            None
        )
    else:
        has_image = False
        text_content = None

    if has_image:
        image_description = await describe_image(last_message["content"])
        print(f"Image description: {image_description}")
        history.pop()
        history.append({"role": "user", "content": f'{text_content} | {image_description}' if text_content else image_description})

    response = await openai_client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=history,
        max_tokens=222,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        top_p=top_p
    )

    response_text = response.choices[0].message.content.strip()

    response_text = re.sub(r"https?://\S+|www\.\S+", "", response_text)
    response_text = response_text.replace("@TimurWasHere", "")
    response_text = response_text.rstrip("😂😏")

    return response_text


async def get_sticker_by_emoji(emoji):

    sticker_sets = [
        "monkeysbynorufx_by_fStikBot",
        "Monkiz3_by_fStikBot",
        "Angrykoreanartists"
    ]

    random.shuffle(sticker_sets)

    for sticker_set_name in sticker_sets:
        sticker_set = await client(GetStickerSetRequest(
            stickerset=InputStickerSetShortName(sticker_set_name),
            hash=0
        ))

        for pack, document in zip(sticker_set.packs, sticker_set.documents):
            if pack.emoticon == emoji:
                return document

    return None

async def describe_image(message_content):
    description_response = await openai_client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "Describe the image(s) provided in a way that another language model can understand and respond appropriately."},
            {"role": "user", "content": message_content}
        ],
        max_tokens=100
    )
    return description_response.choices[0].message.content.strip()

async def get_display_name(sender):
    if sender.first_name:
        return sender.first_name + (" " + sender.last_name if sender.last_name else "")
    elif sender.last_name:
        return sender.last_name
    elif sender.username:
        return sender.username
    else:
        return ""

def is_single_emoji(text):
    return bool(EMOJI_REGEX.fullmatch(text))

async def respond(first_msg: bool, event, history):
    pprint(history)
    response_text = await generate_response(history)
    tokens_count = count_tokens(response_text)

    print(f"Raw response: {response_text}")

    if "/stop-conversation" in response_text:
        raise ValueError("Conversation is over.")

    next_msg = False
    if "/next-msg" in response_text:
        next_msg = True
        response_text = response_text.replace("/next-msg", "").strip()

    if next_msg or len(response_text) == 0:
        await respond(True, event, history)
        return

    await asyncio.sleep(random.uniform(0, 5))
    if is_single_emoji(response_text):
        file = await get_sticker_by_emoji(response_text)
        if file:
            await client.send_file(event.chat_id, file)
        else:
            await event.respond(response_text)
        return

    last_symbol_emoji = None
    if is_single_emoji(response_text[-1]):
        last_symbol_emoji = response_text[-1]
        response_text = response_text[:-1]

    if whisper and random.choice([False, False, True, False, False]):
        await respond_voice(event, response_text)
    else:
        await simulate_typing(event, response_text or '')
        await (event.reply(response_text) if event.is_group and first_msg else event.respond(response_text))

    if last_symbol_emoji:
        file = await get_sticker_by_emoji(last_symbol_emoji)
        if file:
            await client.send_file(event.chat_id, file)

    if next_msg:
        history.append({"role": "assistant", "content": event.text})
        await respond(False, event, history)

async def simulate_typing(event, text):
    chat_id = event.chat_id
    try:
        await client(SetTypingRequest(chat_id, SendMessageTypingAction()))
        typing_time = round(len(text) / TYPING_SPEED)
        print(f"Typing for: {typing_time}")
        await asyncio.sleep(typing_time)
    except Exception as e:
        print(f"Error while sending typing action: {e}")

async def simulate_voice_recording(event, text):
    chat_id = event.chat_id
    try:
        await client(SetTypingRequest(chat_id, SendMessageRecordAudioAction()))
        speech_time = round(len(text) / SPEECH_SPEED)
        print(f"Voice recording for: {speech_time}")
        await asyncio.sleep(speech_time)
    except Exception as e:
        print(f"Error while sending typing action: {e}")


async def check_mention(me, sender_id, event):
    if event.is_reply:
        msg = await event.get_reply_message()
        if msg.from_id and msg.from_id.user_id == me.id:
            return True
        if msg.from_id and msg.from_id.user_id == event.sender_id:
            if msg.is_reply:
                r_msg = await msg.get_reply_message()
                if r_msg.from_id and r_msg.from_id.user_id == me.id:
                    return True

    if bool(re.search(r"@[\w]+", event.text)) and not f"@{me.username}" in event.text:
        return False

    if f"@{me.username}" in event.text:
        return True

    if chats_history.get(sender_id) and len(chats_history[sender_id]) > 2:
        last_msg = chats_history[sender_id][-2]

        if last_msg.get("role") == "assistant" and not event.is_reply:
            return True

    return False

async def check_active_sessions():
    global client
    return False
    auths = await client(GetAuthorizationsRequest())

    now = datetime.utcnow().replace(tzinfo=None)
    print(f"Session time now: {now}")
    active_threshold = timedelta(minutes=3)

    for session in auths.authorizations:
        last_active = session.date_active.replace(tzinfo=None)
        print(f"Session time last active: {last_active}")

        if now - last_active < active_threshold:
            print("User is online - no response")
            return True

    return False

def count_tokens(text, model="gpt-4o-mini-2024-07-18"):
    encoding = tiktoken.encoding_for_model(model)
    tokens = encoding.encode(text)
    return len(tokens)

async def init():
    global me
    me = await client.get_me()
    print(f"🤖 Bot init as: {me.username} | {me.id}")

async def main():
    await init()
    print("🤖 Bot is running...")
    await client.run_until_disconnected()

if __name__ == "__main__":
    # id = extract_youtube_video_id("https://www.youtube.com/watch?v=T23g5f6XmS8")
    # print(id)
    with client:
        client.loop.run_until_complete(main())