import os
from enum import Enum
import tempfile
from pathlib import Path


from telegram.constants import ChatAction, ParseMode
from telegram import Update, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters, Application, \
    CallbackQueryHandler, CallbackContext
from llama_cpp import Llama
import pyttsx3
from pydub import AudioSegment

BOT_TOKEN = os.getenv("BOT_TOKEN")
if BOT_TOKEN is None:
    print("Error: BOT_TOKEN environment variable is not set")
    exit(1)

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    print("Error: MODEL_PATH environment variable is not set or the file does not exist.")
    exit(1)

ALLOWED_USERS = os.getenv("ALLOWED_USERS", "")
GPU_LAYERS = os.getenv("GPU_LAYERS", 0)

llama = Llama(model_path=MODEL_PATH, n_gpu_layers=int(GPU_LAYERS))
try:
    print(f"GPU layers in environment: {GPU_LAYERS}")
    print(f"GPU layers in use: {llama.model.context.gpu_layers}")
except:
    None
user_db = {}
context_len = 250

PROMPT_TEMPLATE_FILE= os.getenv("PROMPT_TEMPLATE_FILE", 0)
with open(PROMPT_TEMPLATE_FILE, "r") as f:
    PROMPT_TEMPLATE = f.read()

print(f"Using prompt:\n{PROMPT_TEMPLATE}")

class ChatMode(Enum):
    TEXT = 1

# Saves last N characters of chat history in memory
def save_chat(user_id, chat_in, chat_out) -> None:
    chat_history = ""
    if user_id not in user_db:
        user_db[user_id] = {}

    try:
        chat_history = user_db[user_id]["history"]
    except KeyError:
        pass

    chat_history = f"{chat_history} {chat_in} {chat_out}"
    if len(chat_history) > context_len:
        chat_history = chat_history[-context_len:]

    user_db[user_id]["history"] = chat_history

    # print(f"history:  {chat_history}")


# Returns users chat history from memory
def get_chat_history(user_id):
    try:
        return user_db[user_id]["history"]
    except KeyError as e:
        print(e)
        pass

    return ""


# Clears users chat history in memory
def clear_chat_history(user_id):
    try:
        user_db[user_id]["history"] = ""
    except KeyError as e:
        print(e)
        pass


# Sets users chat mode
def set_chat_mode(user_id, mode):
    if user_id not in user_db:
        user_db[user_id] = {}

    try:
        user_db[user_id]["chat_mode"] = mode
    except KeyError as e:
        print(e)
        pass


# Returns users current chatmode. defaults to ChatMode.TEXT
def get_chat_mode(user_id):
    try:
        return user_db[user_id]["chat_mode"]
    except KeyError as e:
        print(e)
        pass

    return ChatMode.TEXT


# Returns greeting message on telegram /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"/start called by user={update.message.chat_id}")
    clear_chat_history(update.message.chat_id)
    await update.message.reply_text(f'Hello {update.effective_user.first_name}. I am Alex. Ask me anything. Choose: ',
                                    reply_markup=main_menu_keyboard())


# Clears chat history and returns greeting message on telegram /new_chat command
async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"/new_chat called by user={update.message.chat_id}")
    clear_chat_history(update.message.chat_id)
    await update.message.reply_text(f'Hello {update.effective_user.first_name}. I am Alex. Ask me anything. Choose:',
                                    reply_markup=main_menu_keyboard())


async def start_text_chat(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    set_chat_mode(query.message.chat_id, ChatMode.TEXT)
    await query.answer()
    await query.message.reply_text('Text chat enabled')

# Invokes llama api and returns generated chat response
async def generate_chat_response(prompt, temp_msg, context):
    chat_out = ""
    try:
        tokens = llama.create_completion(prompt, max_tokens=240, top_p=1, stop=["</s>"], stream=True)
        resp = []
        for token in tokens:
            tok = token["choices"][0]["text"]
            if not token["choices"][0]["finish_reason"]:
                resp.append(tok)
                chat_out = ''.join(resp)
                try:
                    # Edit response message on each token to simulate streaming.
                    await context.bot.editMessageText(text=chat_out, chat_id=temp_msg.chat_id,
                                                      message_id=temp_msg.message_id)
                except Exception as e:
                    print(e)
                    # telegram complaints on duplicate edits. pass it.
                    pass

        if not resp:
            print("Empty generation")
            await context.bot.editMessageText(text='Sorry, I am went blank. Try something else',
                                              chat_id=temp_msg.chat_id, message_id=temp_msg.message_id)
    except Exception as e:
        print(f"Unexpected error: {e}")
        await context.bot.editMessageText(text='Sorry, something went wrong :(',
                                          chat_id=temp_msg.chat_id, message_id=temp_msg.message_id)
        pass
    return chat_out


# Handles telegram user chat message
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # print(f"message received: {update.message}")
    # get chat history for user
    chat_history = get_chat_history(update.message.chat_id)
    chat_mode = get_chat_mode(update.message.chat_id)

    chat_in = update.message.text
    chat_id = update.message.chat_id
    print(f"user={chat_id}, chat: {chat_in}")

    # send typing action
    await update.message.chat.send_action(action=ChatAction.TYPING)

    prompt = PROMPT_TEMPLATE.format(chat_in=chat_in, chat_history=chat_history)
    print(f"user={chat_id}, prompt: {prompt}")

    # generate response
    if chat_mode == ChatMode.TEXT:
        temp = await update.message.reply_text("...")
        chat_out = await generate_chat_response(prompt, temp_msg=temp, context=context)
    else:
        None

    save_chat(chat_id, chat_in, chat_out)
    print(f"user={chat_id}, response: {chat_out}")


# Register telegram bot commands
async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new_chat", "Start new chat"),
    ])
    print("Bot commands added")


def main_menu_keyboard():
    keyboard = [[InlineKeyboardButton('Text Chat', callback_data='text')]]
    return InlineKeyboardMarkup(keyboard)


if __name__ == '__main__':

    # Build the telegram bot
    app = (
        ApplicationBuilder()
        .token(BOT_TOKEN)
        .concurrent_updates(4)
        .post_init(post_init)
        .read_timeout(60)
        .write_timeout(60)
        .build()
    )

    # Convert ALLOWED_USERS string to a list.
    allowed_users = [int(user.strip()) if user.strip().isdigit() else user.strip() for user in ALLOWED_USERS.split(",")
                     if ALLOWED_USERS.strip()]

    # make user filters
    user_filter = filters.ALL
    if len(allowed_users) > 0:
        usernames = [x for x in allowed_users if isinstance(x, str)]
        user_ids = [x for x in allowed_users if isinstance(x, int)]
        user_filter = filters.User(username=usernames) | filters.User(user_id=user_ids)

    # add handlers
    app.add_handler(CommandHandler("start", start, filters=user_filter))
    app.add_handler(CommandHandler("new_chat", new_chat, filters=user_filter))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND) & user_filter, handle_message))
    app.add_handler(CallbackQueryHandler(start_text_chat, pattern='text'))

    print("Bot started")
    if allowed_users:
        print(f"Allowed users: {allowed_users}")
    else:
        print(f"Whole world can talk to your bot. Consider adding your ID to ALLOWED_USERS to make it private")

    app.run_polling()
