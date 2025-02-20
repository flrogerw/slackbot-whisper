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
import time
import zipfile
from datetime import datetime
from pathlib import Path
from queue import Queue
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List

import ffmpeg
import numpy as np
import requests
import whisper
from dateutil.tz import tz
from dotenv import load_dotenv
from pydub import AudioSegment
from whisper.tokenizer import get_tokenizer

from models.gemini_model import GeminiQuery
from models.google_doc_model import GoogleDocsManager
from models.paragraph_model import Paragraphs
from models.slack_model import SlackGemini

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
BQ_BUCKET_ID = ai_system['big_query_bucket_id']

SLACK_TOKEN = os.getenv("SLACK_TOKEN")

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add missing mimetypes
mimetypes.add_type('audio/vnd.wave', '.wav')

# Set timezone
timezone = tz.gettz('America/New_York')


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
        self.tokenizer = None

    def run(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        device = "cpu"
        self.model = whisper.load_model("base", device=device)
        self.tokenizer = get_tokenizer(self.model.is_multilingual)
        """Continuously processes events from the queue."""
        while True:
            event = self.event_queue.get()  # Get event from queue
            if event:
                self.process_event(event)
            time.sleep(1)

    @staticmethod
    def convert_response(response):
        try:
            text = response['text']
            word_dicts = [word for segment in response['segments'] for word in segment.get('words', [])]
            words = text.split()  # Get words from text
            reconstructed = []
            word_index = 0  # Pointer for word_dicts

            for word in words:
                merged_word = ""
                start_time = None
                end_time = None

                # Collect dict entries until we fully reconstruct `word`
                while word_index < len(word_dicts) and merged_word.replace(" ", "") != word.replace(" ", ""):
                    word_data = word_dicts[word_index]
                    token_word = word_data["word"]
                    token_start = word_data["start"]
                    token_end = word_data["end"]

                    # Set start time for first part
                    if start_time is None:
                        start_time = token_start

                    # Merge word
                    merged_word += token_word
                    end_time = token_end  # Always take the last end time
                    word_index += 1  # Move to next dict entry

                # Add reconstructed entry
                reconstructed.append({"word": merged_word.strip(), "start": start_time, "end": end_time})

            response_data = {"transcription": text, "words": reconstructed}

        except Exception:
            logging.exception("Error transcribing")
            return None

        else:
            return response_data

    @staticmethod
    def extract_text_from_blocks(blocks: list) -> str:
        """Extracts all text of type 'text' in blocks.elements.elements and stitches them together.

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
    def convert_audio(bytes_io: io.BytesIO, extension: str) -> io.BytesIO:
        # Load audio from BytesIO
        audio = AudioSegment.from_file(bytes_io, format=extension)

        # Whisper prefers mono
        audio = audio.set_channels(1)

        # Export to BytesIO as WAV (needed for ffmpeg input)
        audio_buffer = io.BytesIO()
        audio.export(audio_buffer, format="wav")
        audio_buffer.seek(0)  # Reset buffer position

        # Process with FFmpeg
        output_buffer = io.BytesIO()
        process = (
            ffmpeg.input("pipe:0")
            .output("pipe:1", format="ogg", ac=1, c="libopus", b="12k", application="voip", map_metadata="-1")
            .run(input=audio_buffer.read(), capture_stdout=True, capture_stderr=True)
        )

        # Store the FFmpeg output in BytesIO
        output_buffer.write(process[0])
        output_buffer.seek(0)  # Reset buffer position for reading

        return output_buffer

    @staticmethod
    def zip_files(audio: io.BytesIO, orca_html: str, audio_ext: str, zip_name: str) -> io.BytesIO:

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
    def create_orca_file(paragraphs: list[list[dict[str, Any]]]) -> str:
        """Generate an HTML string representing paragraphs with word-level timing metadata.

        Each word is wrapped in a `<span>` with `data-start` and `data-end` attributes
        to store timestamp information.

        Args:
            paragraphs (List[List[Dict[str, Any]]]): A list of paragraphs, where each paragraph
            is a list of dictionaries containing word details (word text, start time, and end time).

        Returns:
            str: The generated HTML string containing word spans.

        Raises:
            ValueError: If the input `paragraphs` is not formatted correctly.
            RuntimeError: If an unexpected error occurs during processing.

        """
        html_output = ''

        try:
            # Validate input structure
            if not isinstance(paragraphs, list) or not all(isinstance(p, list) for p in paragraphs):
                raise ValueError("Invalid input: `paragraphs` must be a list of lists containing word dictionaries.")

            # Process each paragraph
            for paragraph in paragraphs:
                html_output += '<p>'
                for word in paragraph:
                    # Extract and sanitize word content
                    word_content = word.get("word", "").strip()
                    start_time = word.get("start", 0)
                    end_time = word.get("end", 0)

                    # Append word span with timing data
                    html_output += f"""
                    <div class="tooltip-container">
                        <span class="word" data-start="{start_time}" data-end="{end_time}">
                          {word_content}
                        </span>
                         <div class="tooltip">{start_time}</div>
                         </div>
                        """
                html_output += '</p>'

        except json.JSONDecodeError:
            logging.exception("Error: Failed to parse JSON.")

        except ValueError:
            logging.exception("Input validation error.")
            raise

        except Exception:
            logging.exception("Unexpected error while generating HTML.")

        return html_output

    @staticmethod
    def strided_app(a: np.ndarray, L: int, S: int) -> np.ndarray:
        """Create a 2D sliding window view of a 1D NumPy array.

        This function generates a 2D array where each row is a windowed
        view of the input array, using a specified window length and stride.

        Args:
            a (np.ndarray): The input 1D NumPy array.
            L (int): The length of each window.
            S (int): The stride length (step size) between windows.

        Returns:
            np.ndarray: A 2D NumPy array where each row represents a windowed
            segment of the original array.

        Raises:
            ValueError: If the window length `L` is greater than the input array size.
            TypeError: If the input array is not a NumPy array or if `L` and `S` are not integers.

        """
        try:
            # Validate input types
            if not isinstance(a, np.ndarray):
                raise TypeError("Input `a` must be a NumPy array.")
            if not isinstance(L, int) or not isinstance(S, int):
                raise TypeError("Window length `L` and stride `S` must be integers.")

            # Ensure the window length is valid
            if a.size < L:
                error = "Window length `L` greater than input array."
                raise ValueError(error)

            # Calculate number of rows for the output 2D array
            nrows = ((a.size - L) // S) + 1  # Number of windows that fit in the array

            # Get the byte step size for each element
            n = a.strides[0]

        except Exception:
            # Catch any unexpected errors and provide debug information
            logging.exception("An error occurred in strided_app()")

        else:
            # Generate the 2D strided view using NumPy stride tricks
            return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))

    def pattern_index_broadcasting(self, all_data: list, search_data: list) -> np.ndarray:
        """Find the starting indices of occurrences where search_data fully matches a sublist in all_data.

        This function utilizes NumPy broadcasting and a sliding window approach to efficiently locate
        matching sequences within a larger dataset.

        Args:
            all_data (list): The full sequence of data to be searched.
            search_data (list): The pattern to search for within all_data.

        Returns:
            np.ndarray: An array of indices where the search_data sequence starts within all_data.
                        Returns an empty array if no matches are found.

        """
        try:
            n = len(search_data)  # Length of the search pattern
            all_data = np.asarray(all_data)  # Ensure all_data is a NumPy array

            # Generate a 2D view of all_data with a sliding window approach
            all_data_2d = self.strided_app(all_data, n, S=1)

            # Find indices where all elements of search_data match the sliding window
            match_indices = np.flatnonzero((all_data_2d == search_data).all(1))

        except Exception:
            logging.exception("Error in pattern_index_broadcasting.")

        else:
            return match_indices  # Return matching indices as a NumPy array

    def find_matching_sequence(self, word_dicts: list, paragraphs: list) -> list:
        """Find matching sequences of words in given paragraphs based on a list of word dictionaries.

        This function hashes the words in both the word_dicts and paragraphs to efficiently search for
        matching sequences and retrieves the corresponding word dictionaries.

        Args:
            word_dicts (list): A list of dictionaries, each containing a 'word' key.
            paragraphs (list): A list of strings (paragraphs) to be searched for matching sequences.

        Returns:
            list: A list of lists, where each inner list contains the matching word dictionaries
                  corresponding to the sequences found in the paragraphs.

        """
        try:
            # Hash words from word_dicts to maintain order
            hashed_text = [hash(word_dict["word"].strip()) for word_dict in word_dicts]
            matched_response = []

            # Iterate through each paragraph
            for paragraph in paragraphs:
                # Hash each word in the paragraph after encoding
                words = [hash(word.encode()) for word in paragraph.split()]

                # Find indices where hashed words match in the given sequence
                match_indices = np.squeeze(
                    self.pattern_index_broadcasting(hashed_text, words)[:, None] + np.arange(len(words)),
                )

                # Collect the matched word dictionaries
                matched_response.append([word_dicts[i] for i in match_indices])

        except Exception as e:
            print(f"Error in find_matching_sequence: {e}")
            return []

        else:
            return matched_response

    def process_event(self, event: dict) -> None:
        """Process a Slack event.

        Including file download, AI model processing, and interaction with Google Docs and Google Drive.

        Args:
            event (dict): A dictionary containing event data from Slack,
                          including file and user information.

        Returns:
            None

        """
        start_time = datetime.now(tz=timezone)
        logging.info("Picked up from Queue: %s", start_time)

        try:
            # Notify the user that their request is being processed
            bot_message = "Processing your request..."
            SlackGemini.send_chat_message(event['channel'], bot_message)

            # Extract username and timestamp.
            user_name = SlackGemini.get_user_name(event['user'])
            current_date = datetime.now(tz=timezone).strftime("%Y_%m_%d")
            channel_name = SlackGemini.get_channel_name(event['channel'])

            # Retrieve message text and format.
            logging.info("Retrieve message text and format: %s - %s ", datetime.now(tz=timezone), start_time)
            if 'blocks' not in event:
                bot_message = "The message appears to be blank."
                SlackGemini.send_chat_message(event['channel'], bot_message)
                return

            message = self.extract_text_from_blocks(event['blocks'])

            # Retrieve file from Slack
            logging.info("Retrieve file from Slack: %s - %s ", datetime.now(tz=timezone), start_time)
            event_file = event['files'][0]

            # Set some variable values
            file_url = event_file['url_private_download']
            file_mime_type = event_file['mimetype']

            file_extension = mimetypes.guess_extension(file_mime_type)
            file_date = datetime.now(tz=timezone).strftime("%Y%m%d")
            file_date_time = datetime.now(tz=timezone).strftime("%Y-%m-%d %I:%M %p")
            file_name = f"{file_date}-{user_name}-{message}"

            # Make sure the file is something we can process.
            logging.info(
                "Make sure the file is something we can process: %s - %s ", datetime.now(tz=timezone), start_time)
            if file_mime_type not in AUDIO_FILE_FORMATS:
                bot_message = "The file is not a recognized file type."
                SlackGemini.send_chat_message(event['channel'], bot_message)
                return

            headers = {'Authorization': f'Bearer {SLACK_TOKEN}'}
            response = requests.get(file_url, headers=headers, timeout=5, stream=True)
            response.raise_for_status()  # Raise an error for bad responses (4xx and 5xx)
            file_bytes = response.content

            # Read content into memory
            file_data = io.BytesIO(file_bytes)

            logging.info("Converting audio format for best results.")
            ogg_file_data = self.convert_audio(file_data, file_extension.lstrip("."))

            # Create a temporary file and save the content
            with NamedTemporaryFile(dir="/app/tmp", delete=True, suffix=".ogg") as temp_file:
                temp_file.write(ogg_file_data.getvalue())  # Write memory content to file
                temp_file.flush()  # Ensure data is written

                # Check for max size
                file_path = Path(temp_file.name)
                file_size_mb = file_path.stat().st_size / (1024 * 1024)
                if file_size_mb > 25:
                    temp_file.close()
                    bot_message = f"Your audio file is greater than the allowed 25MB: {file_size_mb}."
                    SlackGemini.send_chat_message(event['channel'], bot_message)
                    logging.info("File to large: %s quiting.", file_size_mb)
                    return

                logging.info("Sending to Whisper: %s - %s", datetime.now(tz=timezone), start_time)
                model_response = self.model.transcribe(temp_file.name, word_timestamps=True)

                # Delete temp file.
                temp_file.close()

                # Parse the response
                whisper_response = self.convert_response(model_response)

            """
            # Read prompt file for Gemini query
            logging.info(f"Initialize the Gemini: {datetime.now(tz=timezone) - start_time}")
            gemini_prompt, gemini_instructions = GeminiQuery.get_prompt(model_response['text'])

            # Configure the Gemini model
            gemini_config = {
                "temperature": float(ai_config['temperature']),
                "top_p": float(ai_config['top_p']),
                "top_k": int(ai_config['top_k']),
                "max_output_tokens": int(ai_config['max_output_tokens']),
                "response_mime_type": "text/plain",
            }

            logging.info(f"Initialize Gemini: {datetime.now(tz=timezone) - start_time}")
            gemini = GeminiQuery(GEMINI_MODEL, None, gemini_instructions, gemini_config)

            # Process the query using the Gemini model
            logging.info(f"Posting to Gemini: {datetime.now(tz=timezone) - start_time}")
            summary = gemini.process_query(gemini_prompt)
            """
            summary = "Summary Goes Here"
            logging.info("Split into paragraphs: %s - %s ", datetime.now(tz=timezone), start_time)
            paragraphs = Paragraphs(model_response['text'])
            paragraphs_list = paragraphs.get_paragraphs()

            logging.info("Initialize the Google Docs manager: %s - %s", datetime.now(tz=timezone), start_time)
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

            logging.info("words sent: %s words returned: %s", len(whisper_response['words']),
                         sum(isinstance(item, dict) for sublist in formatted_words for item in sublist))

            # Get HTML from transcription
            html = self.create_orca_file(formatted_words)

            # Upload Orca zip file
            zip_buffer = self.zip_files(file_data, html, file_extension, file_name)
            docs_manager.upload_bytesio(zip_buffer, f"{file_name}.orca.zip", google_folder_id, "application/zip")

            # Upload process log.
            file_metadata = {
                "user": user_name,
                "date": file_date_time,
                "channel": channel_name,
                "title": message,
                "summary": summary,
                "transcript": model_response['text'],
                "word_count": len(model_response['text'].split())
            }

            docs_manager.upload_dict_as_jsonl(BQ_BUCKET_ID, f"{current_date}/{user_name}/{file_name}.json", file_metadata)

        except requests.RequestException:
            logging.exception("Failed to download audio.")
            error = "Failed to download audio."
            raise error

        except Exception:
            bot_message = "An unexpected error occurred while processing your files."
            SlackGemini.send_chat_message(event['channel'], bot_message)
            error = "An unexpected error occurred while processing the files."
            raise error

        else:
            bot_message = f"Your files have been processed and saved as: {current_date}/{user_name}/{file_name}."
            SlackGemini.send_chat_message(event['channel'], bot_message)
