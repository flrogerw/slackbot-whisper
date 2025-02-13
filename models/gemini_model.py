#!/usr/bin/env python
"""A module that provides the GeminiQuery class for processing audio files using generative AI models.

The module includes functionality to upload and process audio files, handle model interactions,
and manage errors during file processing.
"""
from __future__ import annotations

import configparser
import contextlib
import io
import json
import logging
import os
import typing
from pathlib import Path

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import NotFound, PermissionDenied
from googleapiclient.errors import HttpError

# Conditional import for type checking
if typing.TYPE_CHECKING:
    from google.generativeai.types import File

# Load environment variables from .env file
load_dotenv()

# Access API_KEY from the environment.
API_KEY = os.getenv("API_KEY")

# Parse Gemini config ini file
config = configparser.ConfigParser()
config.read("configure/configure.ini")
ai_config = config['gemini']
ai_system = config['system']

PROMPT_FILE = ai_system['prompt_file']

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GeminiQuery:
    """GeminiQuery class for processing audio files with generative AI models.

    This class provides functionality to interact with a generative AI model, such as Gemini,
    to process audio files and generate content based on a provided prompt. It supports
    handling large and small audio files differently and manages file uploads and deletions
    on the generative AI platform.

    Attributes:
        model_name (str): The name of the generative model to use.
        content_file (str): The path to the audio file to process.
        model (genai.GenerativeModel): The configured generative AI model.

    Methods:
        process_query(prompt: str) -> str:
            Processes an audio file with the generative AI model and returns the generated content.

        delete_remote_file(file_to_delete: File) -> None:
            Deletes a remote file from the generative AI platform.

        process_lg(prompt: str) -> str:
            Processes large audio files by uploading them and generating a summary.

        process_sm(prompt: str) -> str:
            Processes small audio files using a stream and generating a summary.

    """

    def __init__(self, model: str, content_file: str | None, instructions: str, gen_config: dict) -> None:
        """Initialize an instance of the class with the specified model and audio file, and configures the Generative AI model.

        Args:
            model (str): The name of the generative model to use.
            content_file (str): Path to the file to process.
            instructions (str): List of instructions to pass to Gemini.
            gen_config (dict): Configuration properties to pass to Gemini.

        Attributes:
            content_file (str): Path to the provided file.
            model_name (str):  Name of the model.
            model (genai.GenerativeModel): An instance of the generative model created using the specified name.

        """
        # Assign class attributes.
        self.content_file = content_file
        self.model_name = model
        self.instructions = instructions
        self.gen_config = gen_config

        # Configure the GenAI object
        genai.configure(api_key=API_KEY)

        # Create the Gemini model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config=gen_config,
            system_instruction=instructions,
        )

    def get_config(self) -> tuple:
        """Retrieve the configuration details of the instance.

        Returns:
            tuple: A tuple containing the following:
                - gen_config (dict): The general configuration settings.
                - instructions (str): The instructions or parameters associated with the configuration.

        """
        return self.gen_config, self.instructions

    @staticmethod
    def parse_custom_config() -> dict:
        """Parse a custom configuration file with section headers and content.

        This function reads a configuration file line by line, identifying sections marked by headers
        enclosed in square brackets (e.g., `[section_name]`). It associates each section with its
        corresponding content and returns a dictionary where keys are section names and values are
        the content strings.

        Returns:
            dict: A dictionary where keys are section names and values are the concatenated content
                  strings for each section.

        Raises:
            FileNotFoundError: If the configuration file specified by `PROMPT_FILE` is not found.
            PermissionError: If the file cannot be accessed due to insufficient permissions.
            Exception: For any other unexpected errors.

        """
        try:
            sections = {}
            current_section = None
            current_content = []

            # Read the file line by line
            with Path.open(Path(PROMPT_FILE)) as file:
                for file_line in file:
                    line = file_line.strip()  # Remove leading/trailing whitespace
                    # Check for section headers
                    if line.startswith('[') and line.endswith(']'):
                        # Save the current section before starting a new one
                        if current_section:
                            sections[current_section] = "\n".join(current_content).strip()

                        # Start a new section
                        current_section = line[1:-1]  # Remove the square brackets
                        current_content = []
                    elif current_section:  # Collect content under the current section
                        current_content.append(line)

            # Save the last section
            if current_section:
                sections[current_section] = "\n".join(current_content).strip()

        except FileNotFoundError:
            logging.exception("Configuration file '%s' not found.", PROMPT_FILE)

        except PermissionError:
            logging.exception("Permission denied when trying to access '%s'.", PROMPT_FILE)

        except Exception:
            logging.exception("An unexpected error occurred while parsing the configuration file.")

        else:
            return sections
    @staticmethod
    def get_prompt(text: str) -> tuple:
        """Read the first line from a text file to use as a prompt.

        Returns:
            str: The first line of the text file.

        Raises:
            FileNotFound: The specified file was not found.
            Exception: General catch all.

        """
        try:

            with open("configure/schema.json", "r", encoding="utf-8") as file:
                schema = json.load(file)  # Load JSON into a Python dictionary
            logging.info("Retrieving the Gemini prompt...")

            # Read the prompt config file.
            parsed_sections = GeminiQuery.parse_custom_config()

            prompt = parsed_sections.get('prompt', '').replace('\n', ' ')
            prompt = f"{prompt}```{text}```\n\nReturn a JSON object conforming to the following schema: ```json{json.dumps(schema)}```"
            # Access each section as a string


            instructions = parsed_sections.get('instructions', '').replace('\n', '')

            logging.info("Acquired the Gemini prompt.")

        except Exception:  # General exception to catch any other issues
            logging.exception("An unexpected error occurred.")

        else:
            return prompt, instructions

    def process_query(self, prompt: str) -> tuple:
        """Configure the generative AI model and processes an audio file to generate content.

        Args:
            prompt (str): The prompt to guide the content generation.

        Returns:
            tuple: Gemini JSON response of summary and paragraphs.

        Raises:
            FileNotFound: The specified file was not found.
            NameError: A variable, function, or object name that has not been defined.
            NotFound:  Gemini model not found.
            Exception: General catch all.

        """
        try:
            query_results = self.model.generate_content([prompt])
            json_string_clean = query_results.text.replace("```json\n", "").replace("\n```", "")
            gemini_response = json.loads(json_string_clean)

        except HttpError:
            logging.exception("A Http Error occurred while contacting Gemini.")

        except FileNotFoundError:  # File not found error.
            logging.exception("The specified audio file was not found: %s", self.content_file)

        except NameError:
            logging.exception("A NameError has occurred.")

        except NotFound:  # Gemini model not found.
            logging.exception("The Gemini model %s not found.", self.model_name)

        except Exception:  # General exception to catch any other issues
            logging.exception("An unexpected error occurred.")

        else:
            return (gemini_response['summary'], gemini_response['paragraphs'])
