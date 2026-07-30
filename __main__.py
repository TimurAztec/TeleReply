import base64
import io
import os
import re
import random
from pprint import pprint
import tempfile
import threading

import ffmpeg
from PIL import Image
from datetime import datetime, timedelta
import anthropic
import asyncio
from anthropic import AsyncAnthropic
from telethon import TelegramClient, events
from collections import defaultdict
from dotenv import load_dotenv
from telethon.tl.functions.account import GetAuthorizationsRequest
from telethon.tl.functions.messages import SetTypingRequest, GetStickerSetRequest
from telethon.tl.types import SendMessageTypingAction, InputStickerSetShortName
from youtube import extract_youtube_video_id, get_youtube_video_title, summarize_youtube_transcript, \
    get_youtube_transcript

EMOJI_REGEX = re.compile(
    r'^[\U0001F3FB-\U0001F3FF]?[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF'
    r'\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF'
    r'\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0\U000024C2-\U0001F251]+$'
)

load_dotenv()
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CHAT_WHITE_LIST = os.getenv("CHAT_WHITE_LIST").split(',')
SYS_PROMPT = os.getenv("SYS_PROMPT")
HISTORY_MODEL = "claude-haiku-4-5"
PROBABILITY_MODEL = "claude-haiku-4-5"
HISTORY_TOKENS = 777

# Python 3.10+ deprecated (and 3.14 removed) auto-creating a loop when none
# exists for the main thread, so asyncio.get_event_loop() now raises here.
# Create one explicitly and register it before anything else touches asyncio.
try:
    main_loop = asyncio.get_event_loop()
except RuntimeError:
    main_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(main_loop)

client = TelegramClient("session", int(API_ID), API_HASH)
me = None
anthropic_client = AsyncAnthropic(
  api_key=ANTHROPIC_API_KEY
)

reply_enabled = True
busy_replying = defaultdict(lambda: False)
wait_to_end_typing_timers = defaultdict(list)
chats_history = defaultdict(list)
chats_summaries = defaultdict(str)

NUM_PREVIOUS_MESSAGES = 10
TYPING_SPEED = 10
WAIT_TIMER = 5.0
temperature = 0.999
model_id = "claude-haiku-4-5"

@client.on(events.NewMessage(incoming=False))
async def process_out_message(event):
    global reply_enabled, me, temperature
    sender_id = event.chat_id if event.is_group else event.sender_id
    print(f"Out | Chat id: {event.chat_id} | Text: {event.text}")

    if event.chat_id == me.id:
        if event.text.lower() == "reply-on" and not reply_enabled:
            reply_enabled = True
            await event.reply("✅ Auto-reply is ON.")
        elif event.text.lower() == "reply-off" and reply_enabled:
            reply_enabled = False
            await event.reply("❌ Auto-reply is OFF.")

        param_match = re.match(r'set-temperature:\s*([0-9]*\.?[0-9]+)', event.text, re.IGNORECASE)
        if param_match:
            temperature = float(param_match.group(1))
            await event.reply(f"✅ temperature set to {temperature}")
            return
    elif event.chat_id != me.id:
        if f"@{me.username}" in event.text:
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
                chats_history[sender_id].append({"role": "assistant", "content": [{"type": "text", "text": event.text}]})

@client.on(events.NewMessage(incoming=True))
async def process_in_message(event):
    print(f"Incoming | Chat id: {event.chat_id} | Text: {event.text}")

    if (event.is_group and (str(event.chat_id) not in CHAT_WHITE_LIST)) or (event.text == '' and not (event.photo or event.document or event.voice)):
        return

    sender_id = event.chat_id if event.is_group else event.sender_id
    if wait_to_end_typing_timers[sender_id]:
        wait_to_end_typing_timers[sender_id]["timer"].cancel()
    wait_to_end_typing_timers[sender_id] = {
        "timer": threading.Timer(WAIT_TIMER, end_wait_timer, args=(sender_id, event)),
        "active": True
    }
    wait_to_end_typing_timers[sender_id]["timer"].start()

    await handle_message(event)


def build_system_prompt(sender_id):
    if chats_summaries[sender_id]:
        return SYS_PROMPT + "\n\nPrevious conversation summary: " + chats_summaries[sender_id]
    return SYS_PROMPT


async def handle_message(event):
    global me, reply_enabled, busy_replying, temperature
    sender = await event.get_sender()
    sender_id = event.chat_id if event.is_group else event.sender_id

    active = await check_active_sessions()
    if not reply_enabled or active:
        return

    try:
        if not chats_history[sender_id]:
            previous_messages = await client.get_messages(event.chat_id, limit=round(NUM_PREVIOUS_MESSAGES))
            for msg in previous_messages:
                if msg.sender_id and msg.sender_id != me.id:
                    chats_history[sender_id].append({"role": "user", "content": await get_event_content(msg)})
                if msg.sender_id and msg.sender_id == me.id:
                    chats_history[sender_id].append({"role": "assistant", "content": [{"type": "text", "text": msg.text or ""}]})
            chats_history[sender_id].reverse()
            # Claude requires the conversation to start with a "user" turn.
            while chats_history[sender_id] and chats_history[sender_id][0]["role"] != "user":
                chats_history[sender_id].pop(0)
        else:
            msg_content = await get_event_content(event)
            if await count_tokens(get_plain_chat_history(chats_history[sender_id]), HISTORY_MODEL) >= (HISTORY_TOKENS):
                await summarize_history(sender_id)

            chats_history[sender_id].append({"role": "user", "content": msg_content})
        await event.mark_read()

        if wait_to_end_typing_timers[sender_id]["active"]:
            return

        mention = await check_mention(me, sender_id, event)
        if event.is_group and not mention:
            print(f"Not mentioned")
            return

        if busy_replying[sender_id]:
            print(f"Busy!")
            return

        busy_replying[sender_id] = True
        await respond(first_msg=True, event=event, history=chats_history[sender_id])

    except anthropic.RateLimitError:
        print("Quota limit exceeded or rate limit error")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    finally:
        busy_replying[sender_id] = False

async def get_event_content(event):
    content_list = []
    sender = await event.get_sender()
    sender_id = event.chat_id if event.is_group else event.sender_id
    pprint(sender)
    username = await get_display_name(sender)

    if event.is_reply:
        reply_msg = await event.get_reply_message()
        if reply_msg not in chats_history[sender_id]:
            content = await get_event_content(reply_msg)
            summary = content[0]["text"] if content and content[0]["type"] == "text" else "Document or photo"
            content_list.append({"type": "text", "text": f"User responded to message containing: {summary}"})

    if event.text:
        text = event.text
        youtube_id = extract_youtube_video_id(event.text)
        if youtube_id:
            youtube_title = get_youtube_video_title(youtube_id)
            youtube_summary = get_youtube_transcript(youtube_id)
            text += f"\n User attached video titled {youtube_title}: {youtube_summary}"
        content_list.append({"type": "text", "text": f"{username} says: {text}"})

    image_base64 = None
    media_type = "image/jpeg"

    if event.photo or event.document:
        mime_type = getattr(event.document, "mime_type", None)
        print(f"Document: {mime_type}")

        if mime_type in ["image/gif", "image/webp", "application/x-tgsticker"]:
            blob = await event.download_media(bytes)
            image_base64 = await convert_to_jpeg(blob)

        elif mime_type in ["video/webm", "video/mp4"]:
            blob = await event.download_media(bytes)
            fmt = "webm" if "webm" in mime_type else "mp4"
            out = None

            try:
                with tempfile.NamedTemporaryFile(suffix=f".{fmt}") as tmp:
                    tmp.write(blob)
                    tmp.flush()

                    out, err = (
                        ffmpeg
                        .input(tmp.name)
                        .output("pipe:1", vframes=1, format="image2", vcodec="mjpeg", update=1)
                        .run(capture_stdout=True, capture_stderr=True)
                    )
            except ffmpeg.Error as e:
                print("FFmpeg error:", e.stderr.decode(errors="ignore"))

            if out:
                image_base64 = base64.b64encode(out).decode("utf-8")

        elif mime_type == "audio/ogg":
            content_list.append(
                {"type": "text", "text": "User attached voice message, but you cant listen to it at the moment"})

        else:
            blob = await event.download_media(bytes)
            image_base64 = base64.b64encode(blob).decode("utf-8")
            if mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]:
                media_type = mime_type

    if image_base64:
        img_description = await describe_image(image_base64, media_type)
        content_list.append(
                {"type": "text", "text": f"User attached an image that looks like: {img_description}"})

    if not content_list:
        content_list.append({"type": "text", "text": "*empty message*"})

    return content_list

async def summarize_history(sender_id):
    history_text = get_plain_chat_history(chats_history[sender_id])

    try:
        response = await anthropic_client.messages.create(
            model=HISTORY_MODEL,
            max_tokens=200,
            system="Summarize this conversation while keeping key details relevant to the discussion.",
            messages=[{"role": "user", "content": history_text}]
        )
        summary = next(b.text for b in response.content if b.type == "text").strip()
        print(f"History summary: {summary}")
        chats_summaries[sender_id] = summary
        chats_history[sender_id] = []

    except anthropic.RateLimitError:
        print("Rate limit exceeded, skipping history summarization.")
    except Exception as e:
        print(f"Error summarizing history: {e}")

async def convert_to_jpeg(blob):
    image = Image.open(io.BytesIO(blob))
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

async def estimate_response_probability(history, sender_id):
    system_prompt = (
        "You are a classifier. "
        "Your name is Штучний Хохол (aka @TimurIsHere). "
        "Pay attention to context. "
        "Do not get involved into dialog between other users. "
        "Decide if the assistant should respond. "
        "Output only one number between 0 and 1, no text, no punctuation."
    )

    messages = sanitize_history(history)
    if not messages:
        return False

    for attempt in range(2):
        try:
            response = await anthropic_client.messages.create(
                model=PROBABILITY_MODEL,
                max_tokens=10,
                system=system_prompt,
                messages=messages
            )

            text = next((b.text for b in response.content if b.type == "text"), "").strip()

            match = re.search(r"\b\d*\.?\d+\b", text)
            if match:
                prob = float(match.group())
                prob = max(0.0, min(1.0, prob))
                return prob > 0.5

        except Exception as e:
            print(f"[estimate_response_probability] attempt {attempt+1} failed:", e)

        system_prompt = (
            "Output only a number between 0 and 1. "
            "If unsure, output 0. No text, no explanation."
        )
        await asyncio.sleep(0.2)

    return False

async def generate_response(history, sender_id, search=False):
    global model_id

    kwargs = dict(
        model=model_id,
        max_tokens=200,
        temperature=temperature,
        system=build_system_prompt(sender_id),
        messages=sanitize_history(history),
    )

    if search:
        kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

    response = await anthropic_client.messages.create(**kwargs)

    response_text = "".join(b.text for b in response.content if b.type == "text").strip()

    response_text = re.sub(r"https?://\S+|www\.\S+", "", response_text)
    response_text = response_text.replace("@TimurWasHere", "")
    response_text = response_text.rstrip("😂😏")

    return response_text


async def get_sticker_by_emoji(emoji):

    sticker_sets = [
        "EdgyCatboy",
        "Mewglestickerpack",
        "Angrykoreanartists",
        "Almarts27hamsters_by_fStikBot"
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

async def describe_image(image_base64, media_type="image/jpeg"):
    description_response = await anthropic_client.messages.create(
        model=model_id,
        max_tokens=200,
        system="Describe the image(s) provided in a way that another language model can understand and respond appropriately.",
        messages=[
            {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_base64}}]}
        ]
    )
    return next(b.text for b in description_response.content if b.type == "text").strip()

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

async def respond(first_msg: bool, event, history, search=False):
    sender_id = event.chat_id if event.is_group else event.sender_id
    pprint(history)
    response_text = await generate_response(history, sender_id, search)

    print(f"Raw response: {response_text}")

    response_text = re.sub(r'^[\w@]+ says:\s*', '', response_text, flags=re.IGNORECASE).strip()

    if "/stop-conversation" in response_text:
        raise ValueError("Conversation is over.")

    if "/search" in response_text:
        await respond(True, event, history, True)
        return

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

    await simulate_typing(event, response_text or '')
    await (event.reply(response_text) if event.is_group and first_msg else event.respond(response_text))

    if last_symbol_emoji:
        file = await get_sticker_by_emoji(last_symbol_emoji)
        if file:
            await client.send_file(event.chat_id, file)

    if next_msg:
        history.append({"role": "assistant", "content": [{"type": "text", "text": event.text}]})
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


async def check_mention(me, sender_id, event):
    if event.forward:
        fwd = event.fwd_from

        if fwd.from_id and getattr(fwd.from_id, 'channel_id', None):
            return True

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

    if chats_history.get(sender_id) and (len(chats_history[sender_id]) > 2):
        last_msg = chats_history[sender_id][-2]
        print(f"last_mgs| {last_msg}")
        if last_msg.get("role") == "assistant":
            return True

    return await estimate_response_probability(chats_history[sender_id], sender_id)

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

async def count_tokens(text, model=HISTORY_MODEL):
    try:
        response = await anthropic_client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text or "(empty)"}]
        )
        return response.input_tokens
    except Exception as e:
        print(f"Error counting tokens: {e}")
        return 0

def get_plain_chat_history(msgs):
    lines = []

    for msg in msgs or []:
        try:
            role = msg.get("role", "").capitalize()
            if not role or role.lower() == "system":
                continue

            content = msg.get("content")
            text = ""

            if isinstance(content, str):
                text = content.strip()

            elif isinstance(content, list) and content:
                text_item = None
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text" and "text" in item:
                            text_item = item["text"]
                            break

                if text_item is not None:
                    if isinstance(text_item, list):
                        parts = []
                        for sub in text_item:
                            if isinstance(sub, dict) and "text" in sub:
                                parts.append(str(sub["text"]))
                            elif isinstance(sub, str):
                                parts.append(sub)
                            else:
                                parts.append(str(sub))
                        text = " ".join(parts).strip()
                    else:
                        text = str(text_item).strip()
                else:
                    parts = []
                    for item in content:
                        if isinstance(item, dict):
                            item_text = " ".join(str(v) for k, v in item.items() if k != "type")
                            if item_text:
                                parts.append(item_text)
                        elif isinstance(item, str):
                            parts.append(item)
                    text = " ".join(parts).strip() if parts else "*sent document or photo*"

            else:
                text = "*unknown message format*"

            lines.append(f"{role}: {text}")

        except Exception as e:
            lines.append(f"Error parsing message: {e}")

    return "\n".join(lines)

def end_wait_timer(sender_id, event):
    wait_to_end_typing_timers[sender_id]["active"] = False
    asyncio.run_coroutine_threadsafe(handle_message(event), main_loop)

def sanitize_history(msgs):
    cleaned = []

    for msg in msgs or []:
        try:
            role = msg.get("role")
            if not role or role == "system":
                continue

            content = msg.get("content")

            if isinstance(content, str):
                cleaned.append({"role": role, "content": content})
                continue

            if isinstance(content, list):
                valid_items = []

                for item in content:
                    if not isinstance(item, dict):
                        continue

                    item_type = item.get("type", "text")

                    item_text = item.get("text")
                    if item_text is None:
                        item_text = ""
                        for k, v in item.items():
                            if k != "type":
                                item_text += f"{v} "
                        item_text = item_text.strip()

                    if isinstance(item_text, list):
                        parts = []
                        for sub in item_text:
                            if isinstance(sub, dict) and "text" in sub:
                                parts.append(str(sub["text"]))
                            elif isinstance(sub, str):
                                parts.append(sub)
                            else:
                                parts.append(str(sub))
                        item_text = " ".join(parts).strip()

                    if not isinstance(item_text, str):
                        item_text = str(item_text)

                    valid_items.append({"type": item_type, "text": item_text})

                if valid_items:
                    cleaned.append({"role": role, "content": valid_items})
                continue

            cleaned.append({"role": role, "content": str(content)})

        except Exception as e:
            print(f"*error cleaning message: {e}*")

    return cleaned


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
        main_loop.run_until_complete(main())
