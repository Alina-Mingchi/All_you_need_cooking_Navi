"""
Bot to covert text to speech using Together API
"""


import os
import re
import logging
from datetime import datetime
from pydub import AudioSegment
import librosa
import soundfile as sf
import numpy as np

from telegram import Update, InputFile, ForceReply
from telegram.ext import Application, ContextTypes, MessageHandler, filters
from telegram.ext import CommandHandler

from keys import TELEGRAM_KEY, TOGETHER_AI
from together import Together


client = Together(api_key=TOGETHER_AI)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Ensure the audios_generated directory exists
os.makedirs("/Users/alinahou/Code/apis-telegram/audios_generated", exist_ok=True)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Hi {user.mention_html()}! Send me a photo or name of an ingredient to begin.",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
    Bot Help:
    
    - Send an image or ingredient to get a recipe suggestion
    - Use /suggest to get a list of dishes based on an ingredient
    - Use /recipe to get a detailed recipe based on an ingredient
    - Use /grocery to get the grocery list
    - Use /startcooking to begin the recipe process, it will first give a voiced summary, and then detailed steps
    - Use /gen_image to generate an image for social media post
    - Play the generated audio navigation, the bot will guide you through each step 
    Enjoy cooking!
    """
    await update.message.reply_text(help_text)

def tts(text_msg, voice="sweet lady"):
    """
    Generate speech audio from text using Together API.
    """
    try:
        speech_file_path = f"/Users/alinahou/Code/apis-telegram/audios_generated/speech{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        response = client.audio.speech.create(
            model="cartesia/sonic",
            input=text_msg,
            voice=voice,
        )
        response.stream_to_file(speech_file_path)
        return speech_file_path
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return None

async def tts_output(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle text-to-speech conversion and send audio to the user.
    """
    try:
        # Get the input text from the user
        input_text = update.message.text
        await update.message.reply_text(f"Generating audio for: {input_text}")

        # Generate TTS audio
        audio_file_path = tts(input_text)
        if audio_file_path:
            # Send the audio file to the user
            with open(audio_file_path, "rb") as audio_file:
                await update.message.reply_audio(audio=InputFile(audio_file))
        else:
            await update.message.reply_text("Sorry, I couldn't generate the audio.")
    except Exception as e:
        logger.error(f"Error in tts_output: {e}")
        await update.message.reply_text("An error occurred while processing your request.")

async def suggest_dish(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Suggest a dish based on an ingredient provided by the user.
    """
    try:
        ingredient = update.message.text
        await update.message.reply_text(f"Looking for recipes with: {ingredient}")

        # Call Together API for recipe suggestion
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": "You are a helpful assistant, and you are good at suggesting recipes."},
                {"role": "user", "content": f"What can be made with {ingredient}?"},
            ],
        )
        suggestion = response.choices[0].message.content
        await update.message.reply_text(f"Here are some suggestions:\n{suggestion}")
    except Exception as e:
        logger.error(f"Error in suggest_dish: {e}")
        await update.message.reply_text("Sorry, I couldn't find any suggestions for that ingredient.")

async def fetch_recipe(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Suggest a dish based on an ingredient provided by the user.
    """
    try:
        ingredient = update.message.text
        await update.message.reply_text(f"Looking for recipes with: {ingredient}")

        # Call Together API for recipe suggestion
        response = client.chat.completions.create(
            model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "system", "content": "You are a helpful assistant, and you are good at suggesting recipes. \
                 You will first list all ingredients with amounts, then cooking time, and then the recipe."},
                {"role": "user", "content": f"Give detailed recipe of {ingredient}?"},
            ],
        )
        recipe = response.choices[0].message.content

        # Store recipe in user context
        context.user_data["recipe"] = recipe

        await update.message.reply_text(f"Here is the detailed recipe:\n{recipe}")
    except Exception as e:
        logger.error(f"Error in fetch_recipe: {e}")
        await update.message.reply_text("Sorry, I couldn't find any recipe for that ingredient.")

async def grocery_list(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Generate a grocery list based on the stored recipe in user context.
    """
    try:
        # Check if a recipe exists in user data
        if "recipe" in context.user_data:
            recipe = context.user_data["recipe"]
            
            # Extract ingredients from the recipe
            if "**Ingredients:**" in recipe and "**Cooking Time:**" in recipe:
                # Get the section between "**Ingredients:**" and "**Cooking Time:**"
                ingredients_section = recipe.split("**Ingredients:**")[1].split("**Cooking Time:**")[0]
                ingredients = ingredients_section.split("\n")
                
                # Filter out empty lines and format the list
                ingredients = [line.strip() for line in ingredients if line.strip()]
                grocery_list_text = "\n".join(ingredients)
                
                await update.message.reply_text(f"Here is your grocery list:\n{grocery_list_text}")
            else:
                await update.message.reply_text("No ingredients found in the recipe.")
        else:
            await update.message.reply_text("No recipe found. Please use /recipe to fetch a recipe first.")
    except Exception as e:
        logger.error(f"Error in grocery_list: {e}")
        await update.message.reply_text("An error occurred while generating the grocery list.")

import asyncio

class RecipeTimer:
    def __init__(self, chat_id, bot):
        self.chat_id = chat_id
        self.bot = bot
        self.current_timer = None

    async def start_timer(self, step_name: str, duration: int, callback_text: str):
        """
        Start a timer for the given duration and send a callback message when it ends.
        """
        if self.current_timer:
            self.current_timer.cancel()  # Cancel any existing timer

        async def timer_task():
            await asyncio.sleep(duration)
            await self.bot.send_message(chat_id=self.chat_id, text=callback_text)

        self.current_timer = asyncio.create_task(timer_task())

    def cancel_timer(self):
        """
        Cancel the current timer if it exists.
        """
        if self.current_timer:
            self.current_timer.cancel()
            self.current_timer = None

async def _parse_time(time_str: str) -> int:
    """
    Convert time strings to seconds.
    """
    if not time_str:
        return 0

    time_str = time_str.lower()
    total_seconds = 0

    # Match hours
    match = re.search(r"(\d+)\s*hour", time_str)
    if match:
        total_seconds += int(match.group(1)) * 3600  # Convert hours to seconds

    # Match minutes
    match = re.search(r"(\d+)\s*minute", time_str)
    if match:
        total_seconds += int(match.group(1)) * 60  # Convert minutes to seconds

    return total_seconds

async def _execute_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Execute the current recipe step."""
    steps = context.user_data["steps"]
    current_step = context.user_data["current_step"]

    if current_step >= len(steps):
        await update.message.reply_text("Recipe completed! Enjoy your meal!")
        return

    step = steps[current_step]
    step_text = f"Step {current_step + 1}: {step['step_description']}"

    await update.message.reply_text(step_text)
    audio_file_path = tts(step_text)
    with open(audio_file_path, "rb") as audio_file:
        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=InputFile(audio_file))

    if step.get("minutes"):
        await context.user_data["timer"].start_timer(
            step_name=f"Step {current_step + 1}",
            duration=await _parse_time(step["estimated_time"]),
            callback_text="Proceeding to the next step...",
        )

async def startcooking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Extract the dish name and total time from the recipe and save them into context.user_data["summary"].
    """
    try:
        # Check if a recipe exists in user data
        if "recipe" in context.user_data:
            recipe = context.user_data["recipe"]
            
            # Extract the dish name (line before "**Ingredients:**")
            if "**Ingredients:**" in recipe:
                lines = recipe.split("\n")
                ingredients_index = lines.index("**Ingredients:**")
                dish_name = lines[ingredients_index - 2].strip() if ingredients_index > 0 else "Unknown Dish"
                dish_name = dish_name.replace("*", "")  # Remove any '*' characters
            else:
                await update.message.reply_text("No ingredients section found in the recipe.")
                return
            
            # Extract the "Total Time:" line
            total_time = None
            match = re.search(r"Total Time:.*", recipe, re.IGNORECASE)
            if match:
                total_time = match.group(0).strip()  # Extract the matched line
            
            if not total_time:
                total_time = "Total Time: Not specified"
            
            # Save the summary in user data
            context.user_data["summary"] = {
                "dish_name": dish_name,
                "total_time": total_time,
            }
            
            text_output = f"Starting to cook:\n {dish_name}\n {total_time}"

            # Send the summary to the user
            await update.message.reply_text(text_output)

                # Generate TTS audio
            audio_file_path = tts(text_output)
            if audio_file_path:
                # Send the audio file to the user
                with open(audio_file_path, "rb") as audio_file:
                    await update.message.reply_audio(audio=InputFile(audio_file))
            else:
                await update.message.reply_text("Sorry, I couldn't generate the audio.")

            """Begin the cooking process."""
            # Extract steps that start with numbers
            steps = []
            if "**Recipe:**" in recipe:
                # Split the recipe text at '**Recipe:**' and take the part after it
                recipe_section = recipe.split("**Recipe:**")[1]
                lines = recipe_section.split("\n")
        
                # Extract lines that start with a number followed by a period
                for line in lines:
                    if re.match(r"^\d+\.", line.strip()):  # Match lines like "1. Step description"
                        steps.append(line.strip())
            await update.message.reply_text(steps)


            #####
            music_path = "/Users/alinahou/Code/apis-telegram/Lite Saturation - Calm Piano.mp3"
            music_segment = AudioSegment.from_file(music_path)

            # Split the recipe into sentences
            combined_audio = AudioSegment.silent(duration=0)  # Start with an empty audio segment
            silence_audio = AudioSegment.silent(duration=1000)

            for sentence in steps:
                # Generate TTS for the sentence
                sentence = sentence.replace("*", "")  # Remove any '*' characters
                audio_file_path = tts(sentence)
                # Load the generated audio file
                # sentence_audio = AudioSegment.from_file(audio_file_path)
                    # Load the generated audio file with librosa
                y, sr = librosa.load(audio_file_path, sr=None)  # Load with the original sampling rate

                # Convert librosa audio to pydub AudioSegment
                sentence_audio = AudioSegment(
                    (y * 32767).astype(np.int16).tobytes(),  # Convert to 16-bit PCM
                    frame_rate=sr,
                    sample_width=2,  # 16-bit audio
                    channels=1       # Mono audio
                )
                            
                combined_audio += sentence_audio 
                combined_audio += silence_audio

                # Check if the sentence contains "* minutes"
                match = re.search(r"(\d+)\s*minutes?", sentence)
                if match:
                    # duration_minutes = int(match.group(1))  # Extract the number of minutes
                    # duration_ms = (duration_minutes * 60 - 20) * 1000  # Convert minutes to milliseconds -20 seconds
                    duration_ms = 10000
                    # Slice the music segment to match the duration
                    music_slice = music_segment[:duration_ms]
                    combined_audio += music_slice
                    combined_audio += silence_audio

            # Save the combined audio to a file
            combined_audio_path = f"/Users/alinahou/Code/apis-telegram/audios_generated/combined_audio_with_music{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"


            # Convert AudioSegment to NumPy array
            samples = np.array(combined_audio.get_array_of_samples())

            # If the audio is stereo, reshape the array
            if combined_audio.channels == 2:
                samples = samples.reshape((-1, 2))

            # Export to WAV using librosa and soundfile
            sf.write(combined_audio_path, samples, combined_audio.frame_rate, format='WAV')


            ####
            with open(combined_audio_path, "rb") as audio_file:
                await update.message.reply_audio(audio=InputFile(audio_file))

        else:
            await update.message.reply_text("No recipe found. Please use /recipe to fetch a recipe first.")
    except Exception as e:
        logger.error(f"Error in startcooking: {e}")
        await update.message.reply_text("An error occurred while starting the cooking process.")

import requests  # Add this import if not already present

async def gen_image(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Generate an image based on user-specified text using Together AI and send it to the chat.
    """
    try:
        # Get the user-specified text
        user_text = update.message.text
        await update.message.reply_text(f"Generating an image for: {user_text}")

        # Call Together API to generate the image
        response = client.images.generate(
            prompt=f"{user_text}",
            model="black-forest-labs/FLUX.1-dev",
            steps=10,
            n=4
        )
        image_url = response.data[0].url

        if not image_url:
            await update.message.reply_text("Failed to generate the image. Please try again.")
            return

        # Download the image
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            image_path = f"/Users/alinahou/Code/apis-telegram/generated_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            with open(image_path, "wb") as image_file:
                image_file.write(image_response.content)

            # Send the image to the user
            with open(image_path, "rb") as image_file:
                await update.message.reply_photo(photo=image_file)
        else:
            await update.message.reply_text("Failed to download the generated image. Please try again.")
    except Exception as e:
        logger.error(f"Error in gen_image: {e}")
        await update.message.reply_text("An error occurred while generating the image.")


def main() -> None:
    """
    Start the bot.
    """
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_KEY).build()

    # Add handlers
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, tts_output, block=True)
    )
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("suggest", suggest_dish, block=True))
    application.add_handler(CommandHandler("recipe", fetch_recipe, block=True))
    application.add_handler(CommandHandler("grocery", grocery_list, block=True))
    application.add_handler(CommandHandler("startcooking", startcooking, block=True))
    application.add_handler(CommandHandler("gen_image", gen_image, block=True))


    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()