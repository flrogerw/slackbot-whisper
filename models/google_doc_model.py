#!/usr/bin/env python
"""Google Docs and Drive Management Module.

This module provides a `GoogleDocsManager` class to interact with Google Docs and Google Drive APIs.
It allows you to perform operations such as creating documents, adding content, uploading files,
and managing file storage in Google Drive.

To use this module, you need a Google Cloud service account and the corresponding JSON key file.
The service account should have the necessary permissions for the Google Docs and Drive APIs.

Features:
- Create and manage Google Docs.
- Add content to existing documents.
- Upload files to Google Drive from memory or local storage.
- Organize files in specific folders in Google Drive.

Dependencies:
- `google-auth`: For authentication with Google APIs.
- `google-api-python-client`: For interacting with Google Docs and Drive APIs.
- `requests`: For HTTP transport used by Google Auth.
- Logging is configured to provide helpful debugging information.

"""

from __future__ import annotations

import json
import logging
import io

from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseUpload, MediaFileUpload
from google.oauth2.service_account import Credentials

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class GoogleDocsManager:
    """A manager class for Google Docs and Google Drive operations.

    This class provides methods to interact with Google Docs and Google Drive APIs, enabling
    seamless document creation, content insertion, and file management. It uses a service
    account for authentication and supports uploading files directly to Google Drive.

    Attributes:
        service_account_file (str): Path to the service account JSON key file.
        scopes (list): The required scopes for accessing Google Docs and Drive APIs.
        credentials (google.oauth2.service_account.Credentials): The credentials for API access.
        docs_service (googleapiclient.discovery.Resource): The service object for Google Docs API.
        drive_service (googleapiclient.discovery.Resource): The service object for Google Drive API.

    Methods:
        create_document(title: str) -> dict:
            Creates a new Google Doc with the specified title.
        add_content(document_id: str, content: str) -> None:
            Inserts text content into a specified Google Doc.
        upload_audio_bytes(byte_stream: io.BytesIO, file_name: str, folder_id: Optional[str]) -> dict:
            Uploads an audio file from an in-memory byte stream to Google Drive.
        upload_to_drive(document_id: str, folder_id: Optional[str]) -> dict:
            Copies a Google Doc to Google Drive and optionally places it in a specified folder.

    Example:
        manager = GoogleDocsManager("path/to/service_account.json")
        doc = manager.create_document("My New Doc")
        manager.add_content(doc['documentId'], "Hello, world!")
        manager.upload_to_drive(doc['documentId'], folder_id="your-folder-id")

    """

    def __init__(self, gdrive_id: str, service_account_file) -> None:
        """Initialize the GoogleDocsManager with the given service account file.

        This constructor sets up the credentials and initializes the Google Docs
        and Google Drive services using the provided service account JSON key file.

        Args:
            service_account_file (str): Path to the service account JSON key file.

        Raises:
            FileNotFoundError: If the service account file cannot be found.
            ValueError: If there is an issue with the credentials or scopes.
            Exception: For any other unexpected errors.

        """
        try:
            self.gdrive_id = gdrive_id

            self.service_account_file = service_account_file
            self.scopes = [
                'https://www.googleapis.com/auth/documents',
                'https://www.googleapis.com/auth/drive',
            ]

            # Load credentials from the service account file
            self.credentials = Credentials.from_service_account_file(self.service_account_file, scopes=self.scopes)

            # Refresh the credentials if needed (for handling expired tokens)
            if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                self.credentials.refresh(Request())

            # Initialize the Google Docs API service
            self.docs_service = build('docs', 'v1', credentials=self.credentials)

            # Initialize the Google Drive API service
            self.drive_service = build('drive', 'v3', credentials=self.credentials)

            logging.info("GoogleDocsManager initialized successfully.")

        except FileNotFoundError:
            logging.exception("Service account file not found.")
            raise  # Reraise the exception after logging

        except ValueError:
            logging.exception("Error with credentials or scopes.")
            raise  # Reraise the exception after logging

        except Exception:
            logging.exception("An unexpected error occurred while initializing GoogleDocsManager.")
            raise  # Reraise the exception after logging

    def upload_json(self, metadata: dict, file_name: str) -> None:
        json_data = json.dumps(metadata, indent=4)
        json_bytes = io.BytesIO(json_data.encode("utf-8"))

        # Upload JSON directly to Google Drive
        file_metadata = {'name': f'{file_name}.json'}  # Name of the file in Google Drive
        media = MediaIoBaseUpload(json_bytes, mimetype='application/json')

        self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    def upload_bytesio(self, byte_stream: io.BytesIO, file_name: str, folder_id: str | None, mimetype: str) -> dict:
        """Upload an audio file from a BytesIO object to Google Drive.

        Args:
            byte_stream (io.BytesIO): The in-memory byte stream of the audio file.
            file_name (str): The name of the file in Google Drive.
            folder_id (str, optional): The ID of the folder where the file will be uploaded. Defaults to None.

        Returns:
            dict: Metadata of the uploaded file.

        """
        try:
            # File metadata
            file_metadata = {'name': file_name}
            if folder_id:
                file_metadata['parents'] = [folder_id]

            # Media upload object
            media = MediaIoBaseUpload(byte_stream, mimetype=mimetype, resumable=True)

            # Upload the file
            file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                supportsAllDrives=True,
                fields='id, name, mimeType',
            ).execute()

        except Exception as e:
            logging.exception(f"Failed to upload file '{file_name}' to Google Drive.")
            raise

        else:
            logging.info(f"Uploaded file '{file_name}' with ID: {file['id']}")
            return file

    def create_document(self, title: str) -> dict:
        """Create a new Google Doc with the given title.

        Args:
            title (str): Title of the new Google Doc.

        Returns:
            dict: The response containing document metadata, including the document ID.

        Raises:
            Exception: If an error occurs while creating the document.
        """
        try:
            # Create a new Google Doc
            document = self.docs_service.documents().create(body={'title': title}).execute()

        except Exception as e:
            logging.exception(f"Failed to create document with title '{title}'.")
            raise

        else:
            logging.info(f"Created document with ID: {document.get('documentId')}")
            return document

    def add_content(self, document_id: str, header: tuple, response_paragraphs: list) -> None:
        """Add content to an existing Google Doc.

        Args:
                document_id: The ID of the Google Doc.
                header: Tuple of header data.
                response_paragraphs: The response content.

        Raises:
            Exception: If an error occurs while adding content to the document.
        """
        try:

            # Parse header info
            author, datetime, channel, slug, summary = header
            response_text = "".join(response_paragraphs)

            requests = [
                # Add the title
                {
                    'insertText': {
                        'location': {'index': 1},
                        'text': f"Submitted by: {author}\nDate/Time: {datetime}\nChannel: {channel}\nSlug: {slug}\nSummary: {summary}\n\n{response_text}\n"
                    }
                },
                # Apply bold formatting to the headings
                {
                    'updateTextStyle': {
                        'range': {
                            'startIndex': 1,
                            'endIndex': 14  # "Submitted by: "
                        },
                        'textStyle': {'bold': True},
                        'fields': 'bold'
                    }
                },
                {
                    'updateTextStyle': {
                        'range': {
                            'startIndex': 15 + len(author),  # Start after the author name
                            'endIndex': 25 + len(author)  # "Date/Time: "
                        },
                        'textStyle': {'bold': True},
                        'fields': 'bold'
                    }
                },
                {
                    'updateTextStyle': {
                        'range': {
                            'startIndex': 26 + len(author) + len(datetime) + 1,  # Start after "Date/Time: "
                            'endIndex': 34 + len(author) + len(datetime) + 1  # "Channel: "
                        },
                        'textStyle': {'bold': True},
                        'fields': 'bold'
                    }
                },
                {
                    'updateTextStyle': {
                        'range': {
                            'startIndex': 35 + len(author) + len(datetime) + len(channel) + 2,
                            'endIndex': 40 + len(author) + len(datetime) + len(channel) + 3
                        },
                        'textStyle': {'bold': True},
                        'fields': 'bold'
                    }
                },
                {
                    'updateTextStyle': {
                        'range': {
                            'startIndex': 40 + len(author) + len(datetime) + len(channel) + len(slug) + 4,
                            'endIndex': 50 + len(author) + len(datetime) + len(channel) + len(slug) + 3  # "Summary: "
                        },
                        'textStyle': {'bold': True},
                        'fields': 'bold'
                    }
                }
            ]

            p_start = 50 + len(author) + len(datetime) + len(channel) + len(slug) + len(summary) + 5
            for paragraph in response_paragraphs:
                requests.append(
                    {
                        "updateParagraphStyle": {
                            "range": {"startIndex": p_start, "endIndex": p_start + len(paragraph)},  # Adjust range to cover text
                            "paragraphStyle": {
                                "namedStyleType": "NORMAL_TEXT",
                                "alignment": "START"
                            },
                            "fields": "namedStyleType,alignment"
                        }
                    }

                )
                p_start += len(paragraph)

            # Execute the batch update
            self.docs_service.documents().batchUpdate(
                documentId=document_id,
                body={'requests': requests}
            ).execute()

            logging.info(f"Content added to document ID: {document_id}")

        except Exception:
            logging.exception("Failed to add content to the document.")
            raise

    def get_folder_id(self, folder_hierarchy, parent_folder_id: str | None) -> str:
        """Upload a file to a nested folder structure in Google Drive.

        Args:
            folder_hierarchy (list): A list of folder names representing the nested structure.

        Returns:
            dict: The metadata of the uploaded file.
        """

        def get_or_create_folder(folder_name, parent_folder_id=None):
            """Helper function to check or create a folder."""
            if not parent_folder_id:
                parent_folder_id = parent_folder_id if parent_folder_id else self.gdrive_id

            try:
                query = f"""mimeType='application/vnd.google-apps.folder' 
                    and name='{folder_name}'
                    and '{parent_folder_id}' in parents 
                    and trashed=false"""

                results = self.drive_service.files().list(
                    q=query,
                    corpora='drive',
                    driveId=self.gdrive_id,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    fields='files(id, name, parents)',
                    pageSize=1
                ).execute()

                items = results.get('files', [])

                # If folder exists, return its ID
                if items:
                    logging.info(f"Folder '{folder_name}' already exists with ID: {items[0]['id']}")
                    return items[0]['id']

                else:
                    # Create the folder if it doesn't exist
                    folder_metadata = {
                        'name': folder_name,
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [parent_folder_id]
                    }

                    folder = self.drive_service.files().create(
                        body=folder_metadata,
                        supportsAllDrives=True,
                        fields='id'
                    ).execute()

            except Exception:
                logging.exception("An unexpected error occurred")

            else:
                return folder['id']

        # Traverse the folder hierarchy and ensure all levels exist
        for folder_name in folder_hierarchy:
            parent_folder_id = get_or_create_folder(folder_name, parent_folder_id)

        return parent_folder_id

    def upload_to_drive(self, document_id: str, folder_id: str | None) -> dict:
        """Upload the Google Doc to Google Drive, optionally placing it in a specific folder."""
        try:
            # Debugging: Print the document ID
            logging.info(f"Uploading document with ID: {document_id}")

            # Get the document metadata
            doc_metadata = self.docs_service.documents().get(documentId=document_id).execute()
            title = doc_metadata.get('title', 'Untitled Document')

            # File metadata for Google Drive
            file_metadata = {'name': title, 'mimeType': 'application/vnd.google-apps.document'}

            if folder_id:
                file_metadata['parents'] = [folder_id]

            # Copy the file to Google Drive
            file = self.drive_service.files().copy(
                fileId=document_id,
                supportsAllDrives=True,
                body=file_metadata
            ).execute()

        except HttpError as e:
            logging.exception("HttpError occurred during upload.")
            raise e

        except KeyError:
            logging.exception("A key error occurred while uploading the document to Google Drive.")
            raise

        except Exception:
            logging.exception("An unexpected error occurred during the upload process.")
            raise

        else:
            logging.info("Document uploaded to Google Drive.")
            return file
