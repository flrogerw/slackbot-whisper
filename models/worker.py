#!/usr/bin/env python
"""Define the `SlackGemini` class for integrating Slack with the Gemini API and Google Docs.

The module relies on environment variables and configuration files for setup, and it includes various helper methods for interacting with Slack, parsing configuration files, and cleaning strings.

Key Features:
- Handles Slack events and file uploads
- Fetches user information from Slack
- Downloads and processes files using the Gemini AI model
- Interacts with Google Docs and Drive for document management
- Provides logging and error handling throughout the process

Dependencies:
- `requests`: For HTTP requests to download files from Slack
- `slack_sdk`: For interacting with the Slack API
- `dotenv`: For loading environment variables
- `configparser`: For parsing the Gemini configuration file
- `dateutil.tz`: For handling time zone information
- `models.gemini_model`: For interacting with the Gemini model

Configuration:
- The module expects the presence of a `.env` file for environment variables.
- A Gemini configuration file (INI format) should be provided to configure the AI model and other settings.

Functions:
- `get_user_name(user_id: str)`: Fetches the display name of a Slack user by their user ID.
- `parse_custom_config()`: Parses a custom configuration file with section headers and content.
- `clean_string(input_string: str)`: Removes non-alphanumeric characters and replaces spaces with underscores.
- `process_event(event: dict)`: Processes a Slack event, including file download, AI model processing, and Google Docs interaction.
- `get_prompt()`: Retrieves the prompt for the Gemini model from a configuration file.

"""

import configparser
import io
import json
import logging
import mimetypes
import multiprocessing
import os
import re
import string
import zipfile
from datetime import datetime
from pydub import AudioSegment

import time
from queue import Queue
from tempfile import NamedTemporaryFile

import requests
import whisper
from dateutil.tz import tz
from dotenv import load_dotenv


from models.gemini_model import GeminiQuery
from models.google_doc_model import GoogleDocsManager
from models.slack_model import SlackGemini
from models.paragraph_model import Paragraphs

# Load environment variables from .env file
load_dotenv()

# Parse Gemini config ini file
config = configparser.ConfigParser()
config.read("configure/configure.ini")
ai_config = config['gemini']
ai_system = config['system']

GEMINI_MODEL = ai_config['gemini_model']
AUDIO_FILE_FORMATS = ai_system['audio_file_formats'].split(',')
GOOGLE_FOLDER_ID = ai_system['google_folder_id']
SLACK_TOKEN = os.getenv("SLACK_TOKEN")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add missing mimetypes
mimetypes.add_type('audio/vnd.wave', '.wav')


class Worker(multiprocessing.Process):
    """Class for handling Slack file uploads and processing them with Google Gemini.

    This class provides methods for interacting with Slack, downloading files,
    processing them using the Gemini API, and sending responses back to Slack.
    It uses environment variables and configuration files for setup and includes
    error handling and logging.
    """

    def __init__(self, event_queue: Queue):
        super().__init__()
        self.event_queue = event_queue
        self.model = None

    def run(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model("base", device="cpu")
        """Continuously processes events from the queue."""
        while True:
            event = self.event_queue.get()  # Get event from queue
            if event:
                self.process_event(event)
            time.sleep(1)

    @staticmethod
    def convert_response(response: dict):
        try:
            # Format the results (optional)
            transcription = response["text"]
            words = []
            for segment in response["segments"]:
                for word in segment["words"]:
                    words.append({
                        "word": word["word"],
                        "start": word["start"],
                        "end": word["end"],
                    })

            response_data = {"transcription": transcription, "words": words}

        except Exception:
            logging.exception("Error transcribing")
            return None

        else:
            return response_data

    @staticmethod
    def extract_text_from_blocks(blocks: list) -> str:
        """
        Extracts all text of type 'text' in blocks.elements.elements and stitches them together.

        Args:
            blocks (list): A list of blocks containing elements.

        Returns:
            str: A single string with all 'text' elements stitched together.
        """
        extracted_texts = []

        for block in blocks:
            if 'elements' in block:
                for element in block['elements']:
                    if 'elements' in element:
                        for sub_element in element['elements']:
                            if sub_element.get('type') == 'text':
                                extracted_texts.append(sub_element['text'])

        # Remove non-alphanumeric characters except spaces
        cleaned_string = re.sub(r'[^a-zA-Z0-9 ]', '', " ".join(extracted_texts))

        # Make sure response is not blank
        response_string = cleaned_string if cleaned_string else 'title_was_not_supplied'

        # Replace spaces with underscores and return
        return response_string.replace(" ", "_")

    @staticmethod
    def convert_to_mp3_bytes(bytes_io: io.BytesIO, extension: str) -> io.BytesIO:
        # Load audio from BytesIO
        audio = AudioSegment.from_file(bytes_io, format=extension)

        # Whisper prefers mono
        audio = audio.set_channels(1)

        # Convert to flac in memory
        mp3_stream = io.BytesIO()
        audio.export(mp3_stream, format="mp3", bitrate="16k")

        # Move to the beginning of the stream
        mp3_stream.seek(0)

        return mp3_stream  # Returns a BytesIO object containing MP3 data

    @staticmethod
    def zip_files(audio: io.BytesIO, orca_html: str, audio_ext: str, zip_name: str):
        """Creates a ZIP file with the provided audio and HTML content."""

        # List of text-based files (name, content)
        files_to_zip = [("orca.html", orca_html)]

        # Create an in-memory buffer for the ZIP file
        zip_buffer = io.BytesIO()

        # Create the ZIP file in the buffer
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add text files
            for file_name, file_content in files_to_zip:
                zip_file.writestr(file_name, file_content)

            # Add the audio file (assuming it's named 'audio.wav')
            audio.seek(0)  # Ensure we're at the beginning of the BytesIO stream
            zip_file.writestr(f"{zip_name}{audio_ext}", audio.read())  # Read binary content and write to ZIP

        # Seek to the beginning of the buffer to read the ZIP file
        zip_buffer.seek(0)

        return zip_buffer  # Return the ZIP buffer

    @staticmethod
    def create_orca_file(paragraphs: list) -> str:
        html_output = ''
        try:
            for paragraph in paragraphs:
                html_output += '<p>'
                for word in paragraph:
                    word_content = word["word"].strip()
                    html_output += f"""
                        <span class="word" data-start="{word["start"]}" data-end="{word["end"]}" onclick="toggleTooltip(event)">
                          {word_content}
                        </span>
                        """
                html_output += '</p>'

        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON. {e}")

        except Exception:
            logging.error(f"Error sending response to Slack")

        else:
            return html_output

    def find_matching_sequence(self, word_dicts: list, paragraphs: list) -> list:
        """
           Finds all matching sentences in a list of word dictionaries and returns them as lists of word dicts.

           Args:
                word_dicts: A list of dictionaries, where each dictionary represents a word.
                paragraphs: A list of paragraphs.
           Returns:
               A list of lists, where each inner list contains the word dictionaries that form a complete, sequential match
               for a sentence in the input `sentences` list. Returns an empty list if no matches are found.
           """

        matching_sentences = []
        word_index = 0

        for paragraph in paragraphs:

            words = paragraph.split()
            current_match = []

            for i in range(len(word_dicts)):

                current_dict = word_dicts[i]
                current_word = current_dict['word']

                if word_index < len(words) and current_word.strip() == words[word_index]:
                    current_match.append(current_dict)  # append to current match
                    word_index += 1

                    if word_index == len(words):  # all words matched

                        matching_sentences.append(current_match)  # Add the matched dicts as a new list.
                        current_match = []  # reset for the next sentence
                        word_index = 0

                        # Continue checking for other occurrences of the same sentence
                        # Remove the next line to return only the first matching instance.

                        # break # Comment out to find ALL occurrences of the same sentence
                elif word_index > 0:

                    word_index = 0
                    current_match = []
                    if current_word.strip() == words[word_index].strip():
                        current_match.append(current_dict)
                        word_index += 1

        return matching_sentences

    def process_event(self, event: dict) -> None:
        """Process a Slack event.

        Including file download, AI model processing, and interaction with Google Docs and Google Drive.

        Args:
            event (dict): A dictionary containing event data from Slack,
                          including file and user information.

        Returns:
            None

        """
        start_time = datetime.now()
        logging.info(f"Picked up from Queue: {start_time}")

        try:
            # Notify the user that their request is being processed
            bot_message = "Processing your request..."
            SlackGemini.send_chat_message(event['channel'], bot_message)

            # Extract username and timestamp.
            user_name = SlackGemini.get_user_name(event['user'])
            current_date = datetime.now(tz=tz.tzlocal()).strftime("%Y_%m_%d")
            channel_name = SlackGemini.get_channel_name(event['channel'])

            # Retrieve message text and format.
            logging.info(f"Retrieve message text and format: {datetime.now() - start_time}")
            if 'blocks' not in event:
                bot_message = "The message appears to be blank."
                SlackGemini.send_chat_message(event['channel'], bot_message)
                return

            message = self.extract_text_from_blocks(event['blocks'])

            # Retrieve file from Slack
            logging.info(f"Retrieve file from Slack: {datetime.now() - start_time}")
            event_file = event['files'][0]

            # Set some variable values
            file_url = event_file['url_private_download']
            file_mime_type = event_file['mimetype']

            file_extension = mimetypes.guess_extension(file_mime_type)
            file_date = datetime.now(tz=tz.tzlocal()).strftime("%Y%m%d")
            file_date_time = datetime.now(tz=tz.tzlocal()).strftime("%Y-%m-%d %I:%M %p")
            file_name = f"{file_date}-{user_name}-{message}"

            # Make sure the file is something we can process.
            logging.info(f"Make sure the file is something we can process: {datetime.now() - start_time}")
            if file_mime_type not in AUDIO_FILE_FORMATS:
                bot_message = "The file is not a recognized file type."
                SlackGemini.send_chat_message(event['channel'], bot_message)
                return

            # Download the file
            # file_url = "https://dare.wisc.edu/wp-content/uploads/sites/1051/2008/04/Arthur.mp3"
            headers = {'Authorization': f'Bearer {SLACK_TOKEN}'}
            response = requests.get(file_url, headers=headers, timeout=5, stream=True)
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
            file_bytes = response.content
            # Read content into memory
            file_data = io.BytesIO(file_bytes)

            logging.info("Converting audio format for best results.")
            file_data = self.convert_to_mp3_bytes(file_data, file_extension.lstrip("."))
            file_mime_type = "audio/mp3"
            file_extension = ".mp3"

            # Create a temporary file and save the content
            with NamedTemporaryFile(dir="/app/tmp", delete=True, suffix=file_extension) as temp_file:
                temp_file.write(file_data.getvalue())  # Write memory content to file
                temp_file.flush()  # Ensure data is written

                logging.info(f"Sending to Whisper: {datetime.now() - start_time}")
                model_response = self.model.transcribe(temp_file.name, word_timestamps=True)

                # Delete temp file.
                temp_file.close()

                # Parse the response
                whisper_response = self.convert_response(model_response)

            """
            # Read prompt file for Gemini query
            logging.info(f"Initialize the Gemini: {datetime.now() - start_time}")
            gemini_prompt, gemini_instructions = GeminiQuery.get_prompt(model_response['text'])

            # Configure the Gemini model
            gemini_config = {
                "temperature": float(ai_config['temperature']),
                "top_p": float(ai_config['top_p']),
                "top_k": int(ai_config['top_k']),
                "max_output_tokens": int(ai_config['max_output_tokens']),
                "response_mime_type": "text/plain",
            }

            logging.info(f"Initialize Gemini: {datetime.now() - start_time}")
            gemini = GeminiQuery(GEMINI_MODEL, None, gemini_instructions, gemini_config)

            # Process the query using the Gemini model
            logging.info(f"Posting to Gemini: {datetime.now() - start_time}")

            summary = gemini.process_query(gemini_prompt)

            print("GEMINI: ", summary)
            """

            logging.info(f"Split into paragraphs: {datetime.now() - start_time}")
            paragraphs = Paragraphs(model_response['text'])
            paragraphs_list = paragraphs.get_paragraphs()
            summary = "Hello World"

            logging.info(f"Initialize the Google Docs manager: {datetime.now() - start_time}")
            # Initialize the Google Docs manager
            docs_manager = GoogleDocsManager(GOOGLE_FOLDER_ID, "./google_service.json")

            google_folder_id = docs_manager.get_folder_id([current_date, user_name], None)

            # Create a new Google Document
            doc = docs_manager.create_document(file_name)

            # Add content to the document
            header_info = (user_name, file_date_time, channel_name, message.replace('_', ' '), summary)
            docs_manager.add_content(doc['documentId'], header_info, paragraphs_list)

            # Upload the document to Google Drive
            docs_manager.upload_to_drive(doc['documentId'], google_folder_id)

            # Upload the original file to Google Drive
            file_name_in_drive = f"{file_name}{file_extension}"
            docs_manager.upload_bytesio(file_data, file_name_in_drive, google_folder_id, file_mime_type)

            # Add paragraphs to words list
            formatted_words = self.find_matching_sequence(whisper_response['words'], paragraphs_list)

            logging.info(f"org: {len(whisper_response['words'])} paragrah: {sum(isinstance(item, dict) for sublist in formatted_words for item in sublist)}")

            # Get HTML from transcription
            html = self.create_orca_file(formatted_words)

            # Upload Orca zip file
            zip_buffer = self.zip_files(file_data, html, file_extension, file_name)
            docs_manager.upload_bytesio(zip_buffer, f"{file_name}.orca.zip", google_folder_id, "application/zip")

        except requests.RequestException as e:
            logging.error(f"Failed to download audio: {e}")
            raise f"Failed to download audio: {e}"

        except Exception:
            bot_message = f"An unexpected error occurred while processing your files."
            SlackGemini.send_chat_message(event['channel'], bot_message)
            raise "Error interacting with Google Docs or Drive."

        else:
            bot_message = f"Your files have been processed and saved as: {current_date}/{user_name}/{file_name}."
            SlackGemini.send_chat_message(event['channel'], bot_message)
