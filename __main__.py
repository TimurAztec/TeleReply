import base64
import io
import os
import time
from PIL import Image
import openai
from openai import AsyncOpenAI
from telethon import TelegramClient, events
from collections import defaultdict
from dotenv import load_dotenv

load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALLOWED_GROUP_IDS = os.getenv("GROUP_WHITE_LIST").split(',')
SYS_PROMPT = os.getenv("SYS_PROMPT")

client = TelegramClient("session", API_ID, API_HASH)
openai_client = AsyncOpenAI(
  api_key=OPENAI_API_KEY
)

reply_enabled = True
chats_history = defaultdict(list)

NUM_PREVIOUS_MESSAGES = 10

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

    if not chats_history[event.sender_id]:
        previous_messages = await client.get_messages(event.chat_id, limit=NUM_PREVIOUS_MESSAGES)
        
        for msg in previous_messages:
            if msg.from_id != me.id:
                chats_history[event.sender_id].append({"role": "user", "content": msg.text})
            if msg.from_id == me.id:
                chats_history[event.sender_id].append({"role": "assistant", "content": msg.text})

    chats_history[event.sender_id].append({"role": "user", "content": event.text})
    
    if event.is_group and not check_mention(me, event):
        return

    if not reply_enabled:
        return
    
    try:
        system_message = {
            "role": "system",
            "content": SYS_PROMPT
        }
        history = chats_history[event.sender_id][-10:]
        history.insert(0, system_message)
        
        file_path = None

        if event.photo or event.document:
            file_path = await event.download_media()
            image = process_image(file_path)

            history.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}", "detail": "auto"}}
                ]
            })

        response = await openai_client.chat.completions.create(
            model="gpt-4-turbo",
            messages=history,
            max_tokens=666,
            temperature=0.33
        )
        
        explanation = response.choices[0].message.content.strip()

        if not explanation or explanation[-1] not in ".!?":
            explanation += "."

        await event.respond(explanation)

        chats_history[event.sender_id].append({"role": "assistant", "content": explanation})
        
        if file_path:
            try:
                os.remove(file_path)
                print(f"Deleted image: {file_path}")
            except Exception as e:
                print(f"Failed to delete image: {e}")

    except openai._exceptions.RateLimitError:
        print("Quota limit exceeded or rate limit error")
        return ""
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""
    
def resize_image(image, max_dimension):
    width, height = image.size

    if image.mode == "P":
        if "transparency" in image.info:
            image = image.convert("RGBA")
        else:
            image = image.convert("RGB")

    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
        
        timestamp = time.time()

    return image

def convert_to_png(image):
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()
    
def process_image(path, max_size = 1024):
    with Image.open(path) as image:
        width, height = image.size
        mimetype = image.get_format_mimetype()
        if mimetype == "image/png" and width <= max_size and height <= max_size:
            with open(path, "rb") as f:
                encoded_image = base64.b64encode(f.read()).decode('utf-8')
                return (encoded_image, max(width, height))
        else:
            resized_image = resize_image(image, max_size)
            png_image = convert_to_png(resized_image)
            return (base64.b64encode(png_image).decode('utf-8'),
                    max(width, height)
                   )  
    
def check_mention(me, event):
    mention = event.is_reply or (f"@{me.username}" in event.text) or chats_history[event.sender_id][-2].get("role") == "assistant"
    return mention
    
def main():
    print("🤖 Bot is running...")
    client.start()
    client.run_until_disconnected()

if __name__ == "__main__":
    main()