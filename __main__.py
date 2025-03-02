import base64
import io
import os
import re
import random
import ffmpeg
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

try:
    import whisper
except ImportError as e:
    print("Could not import whisper module!")

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

NUM_PREVIOUS_MESSAGES = 5
TYPING_SPEED = 11
temperature=1
presence_penalty=0.11
frequency_penalty=0.99
top_p=0.55

@client.on(events.NewMessage(incoming=False))
async def toggle_reply(event):
    global reply_enabled
    me = await client.get_me()
    sender_id = event.chat_id if event.is_group else event.sender_id
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
        else:
            if event.text:
                chats_history[sender_id].append({"role": "assistant", "content": event.text})

@client.on(events.NewMessage(incoming=True))
async def handle_private_message(event):
    global reply_enabled, busy_replying, temperature, frequency_penalty, presence_penalty, top_p
    me = await client.get_me()
    sender = await event.get_sender()
    print(f"Incoming | Chat id: {event.chat_id} | Text: {event.text}")
    
    if str(event.chat_id) not in CHAT_WHITE_LIST or event.chat_id == me.id or (event.text == '' and not (event.photo or event.document or event.voice)):
    # if str(event.chat_id) not in CHAT_WHITE_LIST or (event.text == '' and not (event.photo or event.document or event.voice)):
        return
    
    sender_id = event.chat_id if event.is_group else event.sender_id
    if not chats_history[sender_id]:
        previous_messages = await client.get_messages(event.chat_id, limit=round(NUM_PREVIOUS_MESSAGES))
        
        for msg in previous_messages:
            if msg.from_id != me.id:
                chats_history[sender_id].append({"role": "user", "content": msg.text})
            if msg.from_id == me.id:
                chats_history[sender_id].append({"role": "assistant", "content": msg.text})

    mention = await check_mention(me, sender_id, event)
    print(f"Mentioned: {mention}")
    if (event.is_group and not mention) or busy_replying[sender_id]:
        print(f"Not mentioned or busy")
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
        content_list = []

        if event.text:
            content_list.append({"type": "text", "text": f'{sender.username}: {event.text}' if sender.username else event.text})

        image_base64 = None

        if event.document:
            mime_type = event.document.mime_type
            print(f"Document: {mime_type}")

            if mime_type in ["image/gif", "image/webp", "application/x-tgsticker"]:
                blob = await event.download_media(bytes)
                image_base64 = await convert_to_jpeg(blob)

            elif mime_type in ["video/webm", "video/mp4"]:
                blob = await event.download_media(bytes)
                out, _ = (
                    ffmpeg.input("pipe:0")
                    .output("pipe:1", vframes=1, format="image2", vcodec="mjpeg")
                    .run(input=blob, capture_stdout=True, capture_stderr=True)
                )
                print(_)
                image_base64 = await convert_to_jpeg(out)

            elif mime_type == "audio/ogg":
                if whisper:
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
            content_list.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}", "detail": "auto"}
            })

        if content_list:
            history.append({"role": "user", "content": content_list})

        await respond(first_msg=True, event=event, history=history)

    except openai._exceptions.RateLimitError:
        print("Quota limit exceeded or rate limit error")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    finally:
        busy_replying[sender_id] = False

async def convert_to_jpeg(blob):
    image = Image.open(io.BytesIO(blob))
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

async def generate_response(history):
    last_message = history[-1] if history else None

    if last_message and isinstance(last_message.get("content"), list):
        has_non_text = any(
            not isinstance(msg, str) or (isinstance(msg, dict) and msg.get("type") == "image_url")
            for msg in last_message["content"]
        )
    else:
        has_non_text = False

    model_name = "gpt-4o-mini-2024-07-18" if has_non_text else "ft:gpt-4o-mini-2024-07-18:personal:timur:B6RCOYAO"

    response = await openai_client.chat.completions.create(
        model=model_name,
        messages=history,
        max_tokens=222,
        temperature=temperature,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        top_p=top_p
    )

    response_text = response.choices[0].message.content.strip()

    if re.search(r"https?://\S+|www\.\S+", response_text) or "@TimurWasHere" in response_text:
        print(f"Link detected in response, regenerating: {response_text}")
        await asyncio.sleep(0.5)
        return await generate_response(history)
    else:
        return response_text

async def respond(first_msg: bool, event, history):
    response_text = await generate_response(history)

    if "/stop-conversation" in response_text:
        raise ValueError("Conversation is over.")

    next_msg = False
    if "/next-msg" in response_text:
        next_msg = True
        response_text = response_text.replace("/next-msg", "").strip()

    await simulate_typing(event, response_text or '')
    await (event.reply(response_text) if event.is_group and first_msg else event.respond(response_text))

    if next_msg:
        history.append({"role": "assistant", "content": event.text})
        await respond(False, event, history)

async def simulate_typing(event, text):
    chat_id = event.chat_id
    try:
        await asyncio.sleep(round(random.uniform(0.1, 5)))
        await client(SetTypingRequest(chat_id, SendMessageTypingAction()))
        typing_time = round(len(text) / TYPING_SPEED)
        print(f"Typing for: {typing_time}")
        await asyncio.sleep(typing_time)
    except Exception as e:
        print(f"Error while sending typing action: {e}")


async def check_mention(me, sender_id, event):
    if event.is_reply:
        msg = await event.get_reply_message()
        if msg.from_id == me.id:
            return True

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
    
def main():
    print("🤖 Bot is running...")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    main()