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


import logging
import os

from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

# Load environment variables from .env file
load_dotenv()

SLACK_TOKEN = os.getenv("SLACK_TOKEN")

client = WebClient(token=SLACK_TOKEN)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class SlackGemini:
    """Class for handling Slack Interactions.

    This class provides methods for interacting with Slack, downloading files,
    processing them using the Gemini API, and sending responses back to Slack.
    It uses environment variables and configuration files for setup and includes
    error handling and logging.
    """

    @staticmethod
    def get_channel_name(channel_id: str) -> str:
        """Fetch the friendly name of a Slack channel given its channel ID.

        Args:
            channel_id (str): The unique ID of the Slack channel.

        Returns:
            str: The friendly name of the channel (e.g., 'general', 'random').

        Raises:
            ValueError: If the channel name cannot be retrieved.
        """
        try:
            # Fetch channel information using the Slack API
            response = client.conversations_info(channel=channel_id)

            # Extract the channel information from the response
            channel_info = response.get("channel", {})

        except SlackApiError as e:
            # Handle API errors, such as insufficient permissions or invalid channel ID
            error_message = f"Error fetching channel info: {e.response['error']}"
            raise ValueError(error_message) from e

        except KeyError:
            # Handle cases where 'name' key is missing in the response
            error_message = "Channel name not found in the response."
            raise ValueError(error_message)

        except Exception as e:
            # Catch any other unexpected exceptions
            error_message = f"An unexpected error occurred: {str(e)}"
            raise ValueError(error_message)

        else:
            # Return the channel name
            return channel_info['name']

    @staticmethod
    def get_user_name(user_id: str) -> str:
        """Fetch the display name of a Slack user by their user ID.

        Args:
            user_id (str): The Slack user ID (e.g., 'U12345ABC').

        Returns:
            str: The user's display/real name.

        """
        try:
            # Call the users_info API method
            response = client.users_info(user=user_id)
            if response["ok"]:
                user_info = response["user"]

        except SlackApiError:
            logging.exception("Error fetching user info.")
            return user_id

        else:
            # Return the display name (fallback to real_name if display_name is not set)
            return user_info.get("display_name", user_info.get("real_name", user_id))

    @staticmethod
    def send_chat_message(channel_id: str, message: str) -> None:
        try:
            # Notify the user that the process is complete
            client.chat_postMessage(
                channel=channel_id,
                text=message
            )

        except Exception:
            logging.error(f"Error sending response to Slack")



