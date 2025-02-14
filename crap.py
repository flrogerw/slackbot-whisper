import numpy as np
import whisper
from whisper.tokenizer import get_tokenizer

# Load Whisper model (any size)
model = whisper.load_model("base", device="cpu")
# Get Whisper's tokenizer
tokenizer = get_tokenizer(model.is_multilingual)


def strided_app(a, L, S):  # Window len = L, Stride len/stepsize = S
    nrows = ((a.size - L) // S) + 1
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a, shape=(nrows, L), strides=(S * n, n))


def pattern_index_broadcasting(all_data, search_data):
    n = len(search_data)
    all_data = np.asarray(all_data)
    all_data_2D = strided_app(np.asarray(all_data), n, S=1)
    return np.flatnonzero((all_data_2D == search_data).all(1))


def filter_tokens(response):
    return [
        t for segment in response['segments'] for t in segment['tokens']
        if tokenizer.decode([t]).strip()  # Remove tokens that decode to empty strings
    ]


# Example text
response = model.transcribe("test.mp3", word_timestamps=True)

# Whisper
token_pieces = [tokenizer.decode([t]).strip() for t in filter_tokens(response)]

# Tokenize the whisper text as Whisper does
tokens = tokenizer.encode(response['text'])
pieces_tokens = [tokenizer.decode([t]).strip() for t in tokens]


x =[t for t in response['segments'][0]['tokens']
        if tokenizer.decode([t]).strip()  # Remove tokens that decode to empty strings
    ]

print(x)

print("Whisper Tokens:", filter_tokens(response))
print("Whisper Tokenized Pieces:", token_pieces)
print("Tokenizer Tokens:", tokens)
print("Tokenizer Tokenized Pieces:", pieces_tokens)

l = filter_tokens(response)
m = [17189, 365, 14332, 9795]

out = np.squeeze(pattern_index_broadcasting(l, m)[:, None] + np.arange(len(m)))

print(out)

def normalize(response):
    word_token_list = []
    transcription = response["text"]
    for segment in response['segments']:
        words = segment['words']
        tokens = [t for t in segment['tokens']
            if tokenizer.decode([t]).strip()  # Remove tokens that decode to empty strings
        ]

        token_index = 0
        for word_info in words:
            if token_index < len(tokens):
                word_token_list.append({
                    'word': word_info['word'].strip(),
                    'token': tokens[token_index],
                    'start': float(word_info['start']),
                    'end': float(word_info['end'])
                })
                token_index += 1

    # Print result
    response_data = {"transcription": transcription, "words": word_token_list}
    print(response_data)


normalize(response)