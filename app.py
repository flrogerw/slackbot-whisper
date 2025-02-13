import io
import logging
import multiprocessing
import os

from datetime import datetime
from tempfile import NamedTemporaryFile

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request
from slack_sdk.signature import SignatureVerifier
from models.worker import Worker
# from models.whisper_worker import WhisperWorker


app = Flask(__name__)

# Event Queue (Multiprocessing-Safe)
event_queue = multiprocessing.Queue()


# Start worker in a background thread
whisper_worker = Worker(event_queue)
whisper_worker.start()

# Load environment variables from .env file
load_dotenv()

SLACK_SIGNING_SECRET = os.getenv("SLACK_SIGNING_SECRET")

verifier = SignatureVerifier(SLACK_SIGNING_SECRET)

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route("/health", methods=['GET'])
def health() -> Response:
    # Return the response
    response = jsonify({"status": "ok"})
    response.status_code = 200
    return response


@app.route("/transcribe", methods=['POST'])
def transcribe() -> Response:
    if 'file' not in request.files:
        response = jsonify({"error": "No file uploaded"})
        response.status_code = 400
        return response

    uploaded_file = request.files['file']  # Get the uploaded file
    file_data = io.BytesIO(uploaded_file.read())  # Convert it to io.BytesIO

    with NamedTemporaryFile(dir="/app/tmp", delete=False, suffix=".mp3") as temp_file:
        temp_file.write(file_data.getvalue())
        temp_file.flush()  # Ensure all data is written
        #temp_file.close()

        event_queue.put("/app/sample_2.wav")
        event_queue.put(temp_file.name)

    # This will be automatically converted to JSON.
    return jsonify({"response": "OK"})


@app.route('/slack/events', methods=['POST'])
def slack_events() -> Response:
    """Handle Slack event requests sent to the server.

    Validates incoming requests from Slack using a verifier. If the request contains an
    event of type 'file_shared', it returns an HTTP 200 response and processes the event
    in a background thread.

    Returns:
        - HTTP 403 response if the request is invalid.
        - HTTP 200 response for valid requests with status "OK".

    """
    start_time = datetime.now()
    logging.info(f"Slack made POST call: {start_time}")

    try:

        event_data = request.json
        # Validate Slack challenge (if first-time setup)
        if "challenge" in event_data:
            response = jsonify({"challenge": event_data["challenge"]})
            response.status_code = 200
            return response

        # Validate the incoming request using the verifier
        if not verifier.is_valid_request(request.get_data(), request.headers):
            logging.exception("Invalid request.")
            response = jsonify({"status": "Invalid request."})
            response.status_code = 403
            return response

        event_data = request.json

        logging.info(f"Slack cleared verification: {datetime.now() - start_time}")

        # Check if the request contains an 'event' key
        if 'event' in event_data:
            event = event_data['event']

            # Handle 'file_shared' events
            if event.get('type') == 'message' and event.get('subtype') == 'file_share':
                try:
                    logging.info(f"Added to Queue: {datetime.now() - start_time}")
                    # Start a background thread to process the event
                    event_queue.put(event)

                except Exception:
                    # Log or handle any errors during thread creation
                    logging.exception("Error starting background thread.")

    except Exception:
        # Handle unexpected errors and log them
        logging.exception("An unexpected error has occurred.")
        response = jsonify({"status": "An unexpected error has occurred."})
        response.status_code = 500
        return response

    else:
        # Return the response
        response = jsonify({"status": "OK"})
        response.status_code = 200
        return response


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
