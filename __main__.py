import base64
import io
import os
import random
import ffmpeg
from PIL import Image
from datetime import datetime, timedelta
import openai
from openai import AsyncOpenAI
from telethon import TelegramClient, events
from collections import defaultdict
from dotenv import load_dotenv
from telethon.tl.functions.account import GetAuthorizationsRequest

load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_GROUP_IDS = os.getenv("GROUP_WHITE_LIST").split(',')
SYS_PROMPT = os.getenv("SYS_PROMPT")

client = TelegramClient("session", int(API_ID), API_HASH)
openai_client = AsyncOpenAI(
  api_key=OPENAI_API_KEY
)

reply_enabled = True
chats_history = defaultdict(list)

NUM_PREVIOUS_MESSAGES = 11

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

@client.on(events.NewMessage(incoming=True))
async def handle_private_message(event):
    global reply_enabled
    me = await client.get_me()
    print(f"Incoming | Chat id: {event.chat_id} | Text: {event.text}")
    
    if (event.is_group and str(event.chat_id) not in ALLOWED_GROUP_IDS) or event.chat_id == me.id or (event.text == '' and not (event.photo or event.document)):
        return
    
    sender_id = event.sender_id if not event.is_group else event.chat_id

    if not chats_history[sender_id]:
        previous_messages = await client.get_messages(event.chat_id, limit=NUM_PREVIOUS_MESSAGES)
        
        for msg in previous_messages:
            if msg.from_id != me.id:
                chats_history[sender_id].append({"role": "user", "content": msg.text})
            if msg.from_id == me.id:
                chats_history[sender_id].append({"role": "assistant", "content": msg.text})

    chats_history[sender_id].append({"role": "user", "content": event.text})
    
    if event.is_group and not check_mention(me, sender_id, event):
        return

    if not reply_enabled or not await check_active_sessions():
        return

    try:
        system_message = {
            "role": "system",
            "content": SYS_PROMPT
        }
        history = chats_history[sender_id][-NUM_PREVIOUS_MESSAGES:]
        history.insert(0, system_message)

        if event.photo or event.document:
            print(f"Image type: {event.document.mime_type}")
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
            else:
                blob = await event.download_media(bytes)
                image_base64 = base64.b64encode(blob).decode("utf-8")

            history.append({
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}}
                ]
            })

        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=history,
            max_tokens=round(random.uniform(333, 1000)),
            temperature=random.uniform(0.25, 0.666)
        )
        
        explanation = response.choices[0].message.content.strip()

        if not explanation or explanation[-1] not in ".!?":
            explanation += "."

        await event.respond(explanation)

        chats_history[sender_id].append({"role": "assistant", "content": explanation})

    except openai._exceptions.RateLimitError:
        print("Quota limit exceeded or rate limit error")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
def check_mention(me, sender_id, event):
    mention = event.is_reply or (f"@{me.username}" in event.text) or chats_history[sender_id][-2].get("role") == "assistant"
    return mention

async def check_active_sessions():
    global client
    auths = await client(GetAuthorizationsRequest())

    now = datetime.utcnow()
    active_threshold = timedelta(minutes=1)

    for session in auths.authorizations:
        last_active = datetime.utcfromtimestamp(session.date_active)
        if now - last_active < active_threshold:
            return True

    return False
    
def main():
    print("🤖 Bot is running...")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    main()