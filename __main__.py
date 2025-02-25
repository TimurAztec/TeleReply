import base64
import io
import os
import random
import ffmpeg
import whisper
from PIL import Image
from datetime import datetime, timedelta
import openai
import asyncio
from openai import AsyncOpenAI
from telethon import TelegramClient, events
from collections import defaultdict
from dotenv import load_dotenv
from telethon.tl.functions.account import GetAuthorizationsRequest
from telethon.tl.functions.messages import SetTypingRequest
from telethon.tl.types import SendMessageTypingAction

load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_WHITE_LIST = os.getenv("CHAT_WHITE_LIST").split(',')
SYS_PROMPT = os.getenv("SYS_PROMPT")

client = TelegramClient("session", int(API_ID), API_HASH)
openai_client = AsyncOpenAI(
  api_key=OPENAI_API_KEY
)

reply_enabled = True
busy_replying = defaultdict(lambda: False)
chats_history = defaultdict(list)

NUM_PREVIOUS_MESSAGES = 11
TYPING_SPEED = 11

@client.on(events.NewMessage(incoming=False))
async def toggle_reply(event):
    global reply_enabled
    me = await client.get_me()
    print(f"Out | Chat id: {event.chat_id} | Text: {event.text}")
    if event.chat_id == me.id:
        if event.text.lower() == "reply-on" and not reply_enabled:
            reply_enabled = True
            await event.respond("✅ Auto-reply is ON.")
        elif event.text.lower() == "reply-off" and reply_enabled:
            reply_enabled = False
            await event.respond("❌ Auto-reply is OFF.")
    elif event.chat_id != me.id:
        if event.text.lower() == "reply-add" and not str(event.chat_id) in CHAT_WHITE_LIST:
            CHAT_WHITE_LIST.append(str(event.chat_id))
            print(f"Chat added: {event.chat_id}")
        elif event.text.lower() == "reply-remove" and str(event.chat_id) in CHAT_WHITE_LIST:
            CHAT_WHITE_LIST.remove(str(event.chat_id))
            print(f"Chat removed: {event.chat_id}")

@client.on(events.NewMessage(incoming=True))
async def handle_private_message(event):
    global reply_enabled, busy_replying
    me = await client.get_me()
    print(f"Incoming | Chat id: {event.chat_id} | Text: {event.text}")
    
    if str(event.chat_id) not in CHAT_WHITE_LIST or event.chat_id == me.id or (event.text == '' and not (event.photo or event.document or event.voice)):
        return
    
    sender_id = event.chat_id if event.is_group else event.sender_id

    if not chats_history[sender_id]:
        previous_messages = await client.get_messages(event.chat_id, limit=NUM_PREVIOUS_MESSAGES)
        
        for msg in previous_messages:
            if msg.from_id != me.id:
                chats_history[sender_id].append({"role": "user", "content": msg.text})
            if msg.from_id == me.id:
                chats_history[sender_id].append({"role": "assistant", "content": msg.text})
    
    if (event.is_group and not check_mention(me, sender_id, event)) or busy_replying[sender_id]:
        return

    active = await check_active_sessions()
    if not reply_enabled or active:
        return

    busy_replying[sender_id] = True
    try:
        system_message = {
            "role": "system",
            "content": SYS_PROMPT
        }
        history = chats_history[sender_id][-NUM_PREVIOUS_MESSAGES:]
        history.insert(0, system_message)

        if event.text:
            history.append({"role": "user", "content": event.text})

        image_base64 = None
        if event.photo or event.document:
            print(f"Document type: {event.document.mime_type}")
            if event.document and event.document.mime_type == "image/gif":
                blob = await event.download_media(bytes)
                image = Image.open(io.BytesIO(blob))
                image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            elif event.document and event.document.mime_type == "video/webm":
                blob = await event.download_media(bytes)
                out, _ = (
                    ffmpeg.input("pipe:0")
                    .output("pipe:1", vframes=1, format="image2", vcodec="mjpeg")
                    .run(input=blob, capture_stdout=True, capture_stderr=True)
                )
                image = Image.open(io.BytesIO(out))
                image = image.convert("RGB")
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            elif event.document and event.document.mime_type == "video/mp4":
                blob = await event.download_media(bytes)
                out, _ = (
                    ffmpeg.input("pipe:0")
                    .output("pipe:1", vframes=1, format="image2", vcodec="mjpeg")
                    .run(input=blob, capture_stdout=True, capture_stderr=True)
                )
                image = Image.open(io.BytesIO(out))
                image = image.convert("RGB")

                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
            elif event.document and event.document.mime_type == "audio/ogg":
                audio_file = await event.download_media()

                model = whisper.load_model("base")
                transcribed_text = model.transcribe(audio_file)
                print(f"Audio transcription: {transcribed_text.get('text')}")
                history.append({"role": "user", "content": transcribed_text.get('text')})
                os.remove(audio_file)
            else:
                blob = await event.download_media(bytes)
                image_base64 = base64.b64encode(blob).decode("utf-8")

            if image_base64:
                history.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}}
                    ]
                })

        response = await openai_client.chat.completions.create(
            model="gpt-4",
            messages=history,
            max_tokens=round(random.uniform(100, 333)),
            temperature=random.uniform(0.5, 1),
            presence_penalty=-1.5,
            frequency_penalty=0.0
        )
        
        explanation = response.choices[0].message.content.strip()
        # print(response.choices[0])
        # stop_flag = response.choices[0].get("stop_conversation", False)

        # if stop_flag:
        #     raise ValueError(f"Conversation with {sender_id} is over.")

        await simulate_typing(event, explanation)
        await event.reply(explanation) if event.is_group else await event.respond(explanation)

        chats_history[sender_id].append({"role": "assistant", "content": explanation})

    except openai._exceptions.RateLimitError:
        print("Quota limit exceeded or rate limit error")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    finally:
        busy_replying[sender_id] = False

async def simulate_typing(event, text):
    chat_id = event.chat_id
    try:
        await asyncio.sleep(round(random.uniform(0.5, 5)))
        await client(SetTypingRequest(chat_id, SendMessageTypingAction()))
        typing_time = round(len(text) / TYPING_SPEED)
        print(f"Typing for: {typing_time}")
        await asyncio.sleep(typing_time)
    except Exception as e:
        print(f"Error while sending typing action: {e}")
    
def check_mention(me, sender_id, event):
    mention = event.is_reply or (f"@{me.username}" in event.text) or chats_history[sender_id][-2].get("role") == "assistant"
    return mention

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
    
def main():
    print("🤖 Bot is running...")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    main()