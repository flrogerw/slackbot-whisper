# Slack File Processor with Gemini

This Flask application integrates with Slack and the Google Gemini API to process files uploaded by users.  When a file is shared in Slack, the app downloads it, sends it to Gemini for AI processing, and returns the results to the Slack channel.

## Features

- **Seamless Slack Integration:** Processes files directly from Slack uploads.
- **Gemini-Powered AI Processing:** Uses the powerful Gemini API to analyze and summarize uploaded files.
- **Customizable Prompts:** Define specific prompts and instructions for Gemini to tailor the processing to your needs.
- **Secure Verification:**  Verifies incoming Slack requests for security.
- **Asynchronous Processing:** Handles file processing in the background to avoid blocking Slack interactions.
- **Easy Deployment:**  Deployable as a standard Flask application.


## Setup and Deployment

1. **Prerequisites:**  Python 3.11+, Flask, `google.generativeai`, `slack_sdk`, `python-dotenv`. Install with:

   ```bash
   pip install -r requirements.txt
   ```

2. **Configuration:**
    - Create a `.env` file in the project root and set the following environment variables:
        - `SLACK_TOKEN`: Your Slack app's Bot User OAuth Token.
        - `SLACK_SIGNING_SECRET`: Your Slack app's Signing Secret.
        - `GEMINI_INI`: Path to your Gemini configuration INI file (e.g., `config/gemini.ini`).

    - Create a `gemini.ini` file (or use the path you specified) and configure Gemini parameters:

      ```ini
      [gemini]
      gemini_model = models/gemini-pro  # Or your chosen model
      temperature = 0.7
      top_p = 0.95
      top_k = 40
      max_output_tokens = 256

      [system]
      prompt_file = prompts/my_prompt.txt # Path to your prompt file.
      ```

    - Create a prompt file (`prompts/my_prompt.txt` or the path you specified). This file should contain sections defining the prompt and instructions for Gemini.  Example:

        ```
        [prompt]
        Summarize the key information in this document:

        [instructions]
        Please provide a concise summary, focusing on the main topics and any action items.
        ```

3. **Slack App Configuration:**
    - Create a Slack app.
    - Enable the `files:read` and `files:write` scopes under "OAuth & Permissions".
    - Subscribe to the `file_shared` event under "Event Subscriptions".
    - Set the "Request URL" to your application's `/slack/events` endpoint (e.g., `https://your-app-url/slack/events`).

4. **Run the App:**

   ```bash
   python your_app_name.py  # Replace your_app_name.py with the actual filename
   ```
   The app defaults to running on `0.0.0.0:3000`, but you can change this within the code if needed.



## Usage

1. In your Slack workspace, upload a file to any channel where your app is a member.
2. The app will receive the `file_shared` event, download the file, and send it to Gemini for processing with the prompt and instructions you defined.
3. The app will post Gemini's response back to the Slack channel.



##  Additional Notes:
- Ensure the user associated with your `SLACK_TOKEN` has the necessary permissions in the Slack workspace.
- Carefully craft your prompt and instructions in the prompt file to get the desired results from Gemini.  Be explicit in telling Gemini what you want it to do with the uploaded file content.
- Consider adding more sophisticated error handling and logging to your application for production environments.




