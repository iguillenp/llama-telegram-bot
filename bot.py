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
from dotenv import load_dotenv
import time
import subprocess

##### Basic language manager for models #####
class ChatMessagesBase:
    lang= ''
    initial_message= ''
    chat_start= ''
    prompt_selected= ''
    empty_generation_response= ''
    unexpected_error_response= ''
    model_info= ''

class ChatMessagesEng(ChatMessagesBase):
    lang= 'eng'
    initial_message= 'Hello {name}. I am an interface for llama.cpp. I have the {model_name} loaded. Choose the language to start prompting me:'
    chat_start= 'Chat started. The prompt will be: \n---\n{prompt}\n---\n You can start the conversation now:'
    prompt_selected= 'You selected the prompt {prompt_filename}. The prompt will be: \n---\n{prompt}\n---\nYou can write me now:'
    empty_generation_response= 'Sorry, I am went blank. Try something else'
    unexpected_error_response= 'Sorry, something went wrong :('
    model_info= 'The model is: {model_name}\n---\nThe model info is: {model_info}\n---'

class ChatMessagesSpa(ChatMessagesBase):
    lang= 'spa'
    initial_message = 'Hola {name}. Soy una interfaz para llama.cpp. Tengo el modelo {model_name} cargado. Elige el idioma para comenzar a interactuar conmigo:'
    chat_start = 'Chat iniciado. El prompt será: \n---\n{prompt}\n---\n Puedes comenzar la conversación ahora:'
    prompt_selected = 'Seleccionaste el prompt {prompt_filename}. El prompt será: \n---\n{prompt}\n---\nYa puedes escribirme:'
    empty_generation_response = 'Lo siento, me he quedado en blanco. Prueba con algo diferente'
    unexpected_error_response = 'Lo siento, algo ha salido mal :('
    model_info= 'El modelo es: {model_name}\n---\nLa información del modelo es: {model_info}\n---'

class ChatMessagesManager:
    """Manager to handle chat message switching with class variable access. Default with English messages."""
    def __init__(self, language_class=ChatMessagesEng):
        self.switch_language(language_class)

    def switch_language(self, language_class):
        """Switch to a different language class."""
        if not issubclass(language_class, ChatMessagesBase):
            raise ValueError("The provided class must inherit from ChatMessagesBase.")
        self._apply_messages(language_class)

    def _apply_messages(self, language_class):
        """Update class variables to match the selected language."""
        for attr_name in dir(language_class):
            # Skip special/private attributes
            if not attr_name.startswith('__'):
                attr_value = getattr(language_class, attr_name)
                setattr(self.__class__, attr_name, attr_value)

ChatMessages= ChatMessagesManager()

###############################################################################################

load_dotenv("environment.env")

# Loading environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
if BOT_TOKEN is None:
    print("Error: BOT_TOKEN environment variable is not set")
    exit(1)

MODEL_PATH = os.getenv("MODEL_PATH")
if not MODEL_PATH or not os.path.isfile(MODEL_PATH):
    print("Error: MODEL_PATH environment variable is not set or the file does not exist.")
    exit(1)
MODEL_NAME= MODEL_PATH.split("/")[-1]

ALLOWED_USERS = os.getenv("ALLOWED_USERS")
GPU_LAYERS = os.getenv("GPU_LAYERS")

llama = Llama(model_path=MODEL_PATH, n_gpu_layers=int(GPU_LAYERS))
try:
    print(f"GPU layers in environment: {GPU_LAYERS}")
    print(f"GPU layers in use: {llama.model.context.gpu_layers}")
except:
    print("No GPU layers in use.")

PROMPT_TEMPLATE_FOLDER= os.getenv("PROMPT_TEMPLATE_FOLDER")
with open(f"{PROMPT_TEMPLATE_FOLDER}/{ChatMessages.lang}/default.prompt", "r") as f:
    PROMPT_TEMPLATE = f.read()



user_db = {}
context_len = 250
def get_comprehensive_model_info(llama):
    """Extract comprehensive Llama model information."""
    model_info = {}
    
    try:
        # Basic model information
        model_info['Model Path'] = llama.model_path
        
        # Try to get context-related info
        try:
            model_info['Context Size'] = llama.model.context.context_size
            model_info['GPU Layers'] = llama.model.context.gpu_layers
        except Exception as e:
            model_info['Context Info'] = f"Could not retrieve - {e}"
        
        # Additional model details
        model_details = [
            'n_ctx', 'n_batch', 'n_threads', 
            'n_gpu_layers', 'model_type', 'vocab_type'
        ]
        
        for detail in model_details:
            try:
                value = getattr(llama.model, detail, None)
                if value is not None:
                    model_info[detail] = str(value)
            except Exception:
                pass
        
    except Exception as e:
        model_info['Error'] = f"Model info retrieval failed: {e}"
    
    return model_info

def check_nvidia():
    return subprocess.check_output(['nvidia-smi']).decode('utf-8')

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

# Sets the language for the prompts
def set_chat_language(user_id, chat_messages:ChatMessagesBase):
    if user_id not in user_db:
        user_db[user_id] = {}
    try:
        user_db[user_id]["chat_lang"] = chat_messages.lang
    except KeyError as e:
        print(e)
        pass

    ChatMessages.switch_language(chat_messages)


# Sets the prompts template
def set_chat_prompt(user_id, selected_prompt):
    if user_id not in user_db:
        user_db[user_id] = {}
    try:
        user_db[user_id]["prompt"] = selected_prompt
    except KeyError as e:
        print(e)
        pass

    global PROMPT_TEMPLATE
    with open(f"{PROMPT_TEMPLATE_FOLDER}/{ChatMessages.lang}/{selected_prompt}", "r") as f:
        PROMPT_TEMPLATE = f.read()


# Menu for language selection
def language_menu_keyboard():
    keyboard = [[InlineKeyboardButton('Español', callback_data='language_spa')],
                [InlineKeyboardButton('English', callback_data='language_eng')]]
    return InlineKeyboardMarkup(keyboard)
    # keyboard = [[InlineKeyboardButton('Text Chat', callback_data='text')]]
    # return InlineKeyboardMarkup(keyboard)

# Menu for prompt selection
def prompt_menu_keyboard():
    keyboard= []

    for file in os.listdir(f"{PROMPT_TEMPLATE_FOLDER}/{ChatMessages.lang}"):
        if file.endswith('.prompt'):
            keyboard.append([InlineKeyboardButton(file.replace(".prompt", ""), callback_data=f'prompt_{file}')])

    return InlineKeyboardMarkup(keyboard)


# Clears chat history and starts a new chat from zero with language selection
async def new_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    print(f"/new_chat called by user={update.message.chat_id}")
    clear_chat_history(update.message.chat_id)
    await update.message.reply_text(ChatMessages.initial_message.format(name=update.effective_user.first_name,
                                                                        model_name=MODEL_NAME),
                                    reply_markup=language_menu_keyboard())

async def get_model_info(update: Update, context: ContextTypes.DEFAULT_TYPE):

    await update.message.reply_text(ChatMessages.model_info.format( model_name=MODEL_NAME,
                                                                    model_info=get_comprehensive_model_info(llama)))


# async def start_spanish_chat(update: Update, context: CallbackContext) -> None:
#     query = update.callback_query
#     set_chat_language(query.message.chat_id, ChatMessagesSpa)
#     await query.answer()
#     await query.message.reply_text(ChatMessages.chat_start.format(prompt=PROMPT_TEMPLATE))

# async def start_english_chat(update: Update, context: CallbackContext) -> None:
#     query = update.callback_query
#     set_chat_language(query.message.chat_id, ChatMessagesEng)
#     await query.answer()
#     await query.message.reply_text(ChatMessages.chat_start.format(prompt=PROMPT_TEMPLATE))


async def handle_language_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    query.answer()

    if query.data == 'language_spa':
        set_chat_language(query.message.chat_id, ChatMessagesSpa)
        await query.message.reply_text("Has seleccionado español. Ahora elige un prompt:", reply_markup=prompt_menu_keyboard())
    elif query.data == 'language_eng':
        set_chat_language(query.message.chat_id, ChatMessagesEng)
        await query.message.reply_text("You have selected English. Now choose a prompt:", reply_markup=prompt_menu_keyboard())

async def handle_prompt_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    query.answer()

    selected_prompt = query.data
    selected_prompt = selected_prompt[7:] # Removing the `prompt_` prefix
    set_chat_prompt(query.message.chat_id, selected_prompt)

    await query.message.reply_text(ChatMessages.prompt_selected.format(prompt_filename=selected_prompt, prompt=PROMPT_TEMPLATE))
    

# Invokes llama api and returns generated chat response
async def generate_chat_response(prompt, temp_msg, context, max_generation_minutes=5):
    max_generation_seconds= max_generation_minutes*60 # Transform minutes to seconds

    chat_out = ""
    try:
        start_time = time.time()
        tokens = llama.create_completion(prompt, max_tokens=240, top_p=1, stop=["</s>"], stream=True)
        resp = []
        for token in tokens:
            current_time = time.time()
            elapsed_time = current_time - start_time

            tok = token["choices"][0]["text"]
            if not token["choices"][0]["finish_reason"]:
                resp.append(tok)
                chat_out = ''.join(resp)
                # If time elapsed add it to the output
                if elapsed_time > max_generation_seconds:
                    chat_out += "[TimeOut]"
                try:
                    # Edit response message on each token to simulate streaming.
                    await context.bot.editMessageText(text=chat_out, chat_id=temp_msg.chat_id,
                                                      message_id=temp_msg.message_id)
                except Exception as e:
                    print(e)
                    # telegram complaints on duplicate edits. pass it.
                    pass

            # Break if time limit exceeded
            if elapsed_time > max_generation_seconds:
                return chat_out
            
        if not resp:
            print("Empty generation")
            await context.bot.editMessageText(text=ChatMessages.empty_generation_response,
                                              chat_id=temp_msg.chat_id, message_id=temp_msg.message_id)
    except Exception as e:
        print(f"Unexpected error: {e}")
        await context.bot.editMessageText(text= ChatMessages.unexpected_error_response,
                                          chat_id=temp_msg.chat_id, message_id=temp_msg.message_id)
        pass
    return chat_out


# Handles telegram user chat message
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # print(f"message received: {update.message}")
    # get chat history for user
    chat_history = get_chat_history(update.message.chat_id)

    chat_in = update.message.text
    chat_id = update.message.chat_id
    print(f"user={chat_id}, chat: {chat_in}")

    # send typing action
    await update.message.chat.send_action(action=ChatAction.TYPING)

    try:
        prompt = PROMPT_TEMPLATE.format(chat_in=chat_in, chat_history=chat_history)
    except KeyError:
        prompt = PROMPT_TEMPLATE.format(chat_in=chat_in)

    print(f"user={chat_id}, prompt: {prompt}")

    # generate response
    temp = await update.message.reply_text("...")
    chat_out = await generate_chat_response(prompt, temp_msg=temp, context=context)

    save_chat(chat_id, chat_in, chat_out)
    print(f"user={chat_id}, response: {chat_out}")


# Register telegram bot commands
async def post_init(application: Application):
    await application.bot.set_my_commands([
        BotCommand("/new_chat", "Start new chat"),
        BotCommand("/model_info", "Get info of the loaded model."),
        BotCommand("/nvidia_smi", "Get GPU info."),
        
    ])
    print("Bot commands added")




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
    app.add_handler(CommandHandler("new_chat", new_chat, filters=user_filter))
    app.add_handler(CommandHandler("model_info", get_model_info, filters=user_filter))
    app.add_handler(CommandHandler("nvidia_smi", check_nvidia, filters=user_filter))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND) & user_filter, handle_message))
    app.add_handler(CallbackQueryHandler(handle_language_selection, pattern='^language_'))
    app.add_handler(CallbackQueryHandler(handle_prompt_selection, pattern='^prompt_'))

    print("Bot started")
    if allowed_users:
        print(f"Allowed users: {allowed_users}")
    else:
        print(f"Whole world can talk to your bot. Consider adding your ID to ALLOWED_USERS to make it private")

    app.run_polling()
