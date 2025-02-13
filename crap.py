import re
import string

import numpy as np

paragraphs = ['The peace league meant to discuss their plans.', 'The rise to fame of a person takes luck.']
word_objects = [{'word': ' The', 'start': np.float64(4.76), 'end': np.float64(4.86)},
                {'word': ' peace', 'start': np.float64(4.86), 'end': np.float64(5.24)},
                {'word': ' league', 'start': np.float64(5.24), 'end': np.float64(5.44)},
                {'word': ' meant', 'start': np.float64(5.44), 'end': np.float64(5.84)},
                {'word': ' to', 'start': np.float64(5.84), 'end': np.float64(6.26)},
                {'word': ' discuss', 'start': np.float64(6.26), 'end': np.float64(6.54)},
                {'word': ' their', 'start': np.float64(6.54), 'end': np.float64(6.84)},
                {'word': ' plans.', 'start': np.float64(6.84), 'end': np.float64(7.34)},
                {'word': ' The', 'start': np.float64(8.16), 'end': np.float64(8.22)},
                {'word': ' rise', 'start': np.float64(8.22), 'end': np.float64(8.52)},
                {'word': ' to', 'start': np.float64(8.52), 'end': np.float64(8.82)},
                {'word': ' fame', 'start': np.float64(8.82), 'end': np.float64(9.14)},
                {'word': ' of', 'start': np.float64(9.14), 'end': np.float64(9.32)},
                {'word': ' a', 'start': np.float64(9.32), 'end': np.float64(9.38)},
                {'word': ' person', 'start': np.float64(9.38), 'end': np.float64(9.78)},
                {'word': ' takes', 'start': np.float64(9.78), 'end': np.float64(10.38)},
                {'word': ' luck.', 'start': np.float64(10.38), 'end': np.float64(10.72)},
                {'word': ' Paper', 'start': np.float64(11.78), 'end': np.float64(12.24)},
                {'word': ' is', 'start': np.float64(12.24), 'end': np.float64(12.46)},
                {'word': ' scarce,', 'start': np.float64(12.46), 'end': np.float64(12.82)},
                {'word': ' so', 'start': np.float64(13.38), 'end': np.float64(13.44)},
                {'word': ' right', 'start': np.float64(13.44), 'end': np.float64(13.7)},
                {'word': ' with', 'start': np.float64(13.7), 'end': np.float64(14.14)},
                {'word': ' much', 'start': np.float64(14.14), 'end': np.float64(14.44)},
                {'word': ' care.', 'start': np.float64(14.44), 'end': np.float64(14.8)},
                {'word': ' The', 'start': np.float64(15.780000000000001), 'end': np.float64(16.46)},
                {'word': ' quick', 'start': np.float64(16.46), 'end': np.float64(16.8)},
                {'word': ' fox', 'start': np.float64(16.8), 'end': np.float64(17.28)},
                {'word': ' jumped', 'start': np.float64(17.28), 'end': np.float64(17.8)},
                {'word': ' on', 'start': np.float64(17.8), 'end': np.float64(18.08)},
                {'word': ' the', 'start': np.float64(18.08), 'end': np.float64(18.18)},
                {'word': ' sleeping', 'start': np.float64(18.18), 'end': np.float64(18.48)},
                {'word': ' cat.', 'start': np.float64(18.48), 'end': np.float64(18.8)},
                {'word': ' The', 'start': np.float64(20.0), 'end': np.float64(20.34)},
                {'word': ' nozzle', 'start': np.float64(20.34), 'end': np.float64(20.66)},
                {'word': ' of', 'start': np.float64(20.66), 'end': np.float64(20.88)},
                {'word': ' the', 'start': np.float64(20.88), 'end': np.float64(20.98)},
                {'word': ' fire', 'start': np.float64(20.98), 'end': np.float64(21.22)},
                {'word': ' hose', 'start': np.float64(21.22), 'end': np.float64(21.58)},
                {'word': ' was', 'start': np.float64(21.58), 'end': np.float64(21.9)},
                {'word': ' bright', 'start': np.float64(21.9), 'end': np.float64(22.2)},
                {'word': ' brass.', 'start': np.float64(22.2), 'end': np.float64(22.64)},
                {'word': ' Screwed', 'start': np.float64(24.14), 'end': np.float64(24.76)},
                {'word': ' around', 'start': np.float64(24.76), 'end': np.float64(25.0)},
                {'word': ' cat', 'start': np.float64(25.0), 'end': np.float64(25.3)},
                {'word': ' on', 'start': np.float64(25.3), 'end': np.float64(25.72)},
                {'word': ' as', 'start': np.float64(25.72), 'end': np.float64(25.9)},
                {'word': ' tight', 'start': np.float64(25.9), 'end': np.float64(26.1)},
                {'word': ' as', 'start': np.float64(26.1), 'end': np.float64(26.3)},
                {'word': ' needed.', 'start': np.float64(26.3), 'end': np.float64(26.64)}]


def find_matching_sequence(word_objects: list, paragraphs: list) -> list:
    result = []
    for paragraph in paragraphs:
        target_words = paragraph.split()  # Split the target string into words
        print(target_words)
        for i in range(len(word_objects) - len(target_words) + 1):
            match = True
            for j in range(len(target_words)):
                if word_objects[i + j]['word'].rstrip(string.punctuation).strip() != target_words[j].rstrip(
                        string.punctuation):
                    match = False
                    break
            if match:
                result.append(word_objects[i: i + len(target_words)][0])  # Return matching objects

    return result  # Return empty if no match is found


#print(find_matching_sequence(word_objects, paragraphs))

results = []
# Normalize words in word_objects for matching (remove leading spaces)
word_map = {obj['word'].rstrip(string.punctuation).strip(): obj for obj in word_objects}

# Function to tokenize sentences into words while preserving punctuation
def tokenize(sentence):
    return re.findall(r"\b\w+\b|[.,!?;]", sentence.rstrip(string.punctuation))

# Process each paragraph
matched_words = []
for paragraph in paragraphs:
    words = tokenize(paragraph)  # Tokenize the paragraph
    for word in words:
        if word.strip() in word_map:
            matched_words.append(word_map[word])

# Output the matched word objects
    results.append(matched_words)
    matched_words = []
print(results)