import json
import os
import re
from dotenv import load_dotenv
from telethon import TelegramClient

load_dotenv()
API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
SYS_PROMPT = "Your name is Timur, and your username is @TimurWasHere. You're a 23-year-old programmer, cyclist, and gamer from Poltava, now living in Lviv—with a girlfriend named Julia. You have a dark, sarcastic sense of humor steeped in internet culture. You love offensive jokes, swearing, memes, and inside jokes, and you're never afraid to roast someone when they're wrong. Respond like a real person chatting with friends: keep your tone casual, natural, and punchy. Stick to the topic and avoid introducing unrelated information. If a reply can be a single emoji, do that; If a joke is particularly inappropriate but funny, go for it. You don’t ask questions—just share your thoughts. Recognize when the conversation has reached its natural end or text not worth answering; when that happens, include '/stop-conversation' in your response. Also, you ride a fixed-gear Colossi bike and live off caffeine, bad decisions, silly jokes, old movies, indie video games, political debates, and cycling."

client = TelegramClient("session", API_ID, API_HASH)


def calculate_weight(assistant_text):
    assistant_length = len(assistant_text.split())

    weight = 0.5 + (1.5 * assistant_length / 50)

    return min(weight, 2.0)


async def extract_chat_data():
    fine_tuning_filename = "fine_tuning_data.jsonl"
    me = await client.get_me()

    try:
        with open(fine_tuning_filename, "w", encoding="utf-8") as outfile:
            async for dialog in client.iter_dialogs():
                print(f"Processing dialog: {dialog.title}")

                async for msg in client.iter_messages(dialog.id, reverse=True, from_user=me):
                    if msg.text and msg.is_reply and not re.search(r"https?://\S+|www\.\S+", msg.text):
                        replied_msg = await msg.get_reply_message()
                        if replied_msg and replied_msg.text:
                            weight = calculate_weight(replied_msg.text.strip() + msg.text.strip())

                            conversation = {
                                "messages": [
                                    {"role": "system", "content": SYS_PROMPT},
                                    {"role": "user", "content": replied_msg.text.strip()},
                                    {"role": "assistant", "content": msg.text.strip(), "weight": weight}
                                ]
                            }
                            # print(f"{replied_msg.text.strip()} | {msg.text.strip()} | Weight: {weight:.2f}")
                            outfile.write(json.dumps(conversation, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print(f"Extracted chat data saved to {fine_tuning_filename}")


async def main():
    await extract_chat_data()


with client:
    client.loop.run_until_complete(main())
