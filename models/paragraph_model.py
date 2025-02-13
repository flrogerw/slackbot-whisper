import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import math
import pysbd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest

seg = pysbd.Segmenter(language="en", clean=False)
model = SentenceTransformer('all-mpnet-base-v2')
nlp = spacy.load('en_core_web_sm')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Paragraphs:

    def __init__(self, transcript: str):
        self.transcript = transcript

    def get_summary(self, text, per):
        doc = nlp(text)
        tokens = [token.text for token in doc]
        word_frequencies = {}
        for word in doc:
            if word.text.lower() not in list(STOP_WORDS):
                if word.text.lower() not in punctuation:
                    if word.text not in word_frequencies.keys():
                        word_frequencies[word.text] = 1
                    else:
                        word_frequencies[word.text] += 1
        max_frequency = max(word_frequencies.values())
        for word in word_frequencies.keys():
            word_frequencies[word] = word_frequencies[word] / max_frequency
        sentence_tokens = [sent for sent in doc.sents]
        sentence_scores = {}
        for sent in sentence_tokens:
            for word in sent:
                if word.text.lower() in word_frequencies.keys():
                    if sent not in sentence_scores.keys():
                        sentence_scores[sent] = word_frequencies[word.text.lower()]
                    else:
                        sentence_scores[sent] += word_frequencies[word.text.lower()]
        select_length = int(len(sentence_tokens) * per)
        summary = nlargest(select_length, sentence_scores, key=sentence_scores.get)
        final_summary = [word.text for word in summary]
        summary = ''.join(final_summary)
        return summary

    @staticmethod
    def rev_sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(0.5 * x))

    def activate_similarities(self, similarities: np.array, p_size=10) -> np.array:
        """ Function returns list of weighted sums of activated sentence similarities
            Args:
                similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
                p_size (int): number of sentences are used to calculate weighted sum
            Returns:
                list: list of weighted sums
            """
        # To create weights for sigmoid function we first have to create space.
        # P_size will determine number of sentences used and the size of weights vector.
        x = np.linspace(-10, 10, p_size)

        # Then we need to apply activation function to the created space
        y = np.vectorize(self.rev_sigmoid)

        # Because we only apply activation to p_size number of sentences
        # we have to add zeros to neglect the effect of every additional
        # sentence and to match the length of vector we will multiply
        activation_weights = np.pad(y(x), (0, similarities.shape[0] - p_size))

        # 1. Take each diagonal to the right of the main diagonal
        diagonals = [similarities.diagonal(each) for each in range(0, similarities.shape[0])]

        # 2. Pad each diagonal by zeros at the end.
        diagonals = [np.pad(each, (0, similarities.shape[0] - len(each))) for each in diagonals]

        # 3. Stack those diagonals into new matrix
        diagonals = np.stack(diagonals)

        # 4. Apply activation weights to each row. Multiply similarities with our activation.
        diagonals = diagonals * activation_weights.reshape(-1, 1)

        # 5. Calculate the weighted sum of activated similarities
        activated_sims = np.sum(diagonals, axis=0)
        return activated_sims

    def get_paragraphs(self):
        # split text into sentences.
        sentences = seg.segment(self.transcript)
        """
        # Get the length of each sentence
        sentence_length = [len(each) for each in sentences]

        # Determine longest outlier
        long = np.mean(sentence_length) + np.std(sentence_length) * 2

        # Determine shortest outlier
        short = np.mean(sentence_length) - np.std(sentence_length) * 2

        # Shorten long sentences
        text = ''
        for each in sentences:
            if len(each) > long:
                # let's replace all the commas with dots
                comma_splitted = each.replace(',', '.')
            else:
                text += f'{each}. '
        sentences = text.split('. ')

        # Now let's concatenate short ones
        text = ''
        for each in sentences:
            if len(each) < short:
                text += f'{each}. '
            else:
                text += f'{each}. '

        # Split text into sentences
        sentences = text.split('. ')
        """
        # Embed sentences
        embeddings = model.encode(sentences)

        # Create similarities matrix
        similarities = cosine_similarity(embeddings)

        # Normalize the embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        activated_similarities = self.activate_similarities(similarities, p_size=10)

        minmimas = argrelextrema(activated_similarities, np.less, order=2)

        # Create empty string
        split_points = [each for each in minmimas[0]]
        text_array = []
        current_text = ''

        for num, each in enumerate(sentences):
            if num in split_points:
                text_array.append(current_text.strip())  # Append the accumulated text
                current_text = f'{each} '  # Start a new text block
            else:
                if len(each) > 1:
                    current_text += f'{each} '

            # Append the last accumulated text if it's not empty
        if current_text:
            text_array.append(current_text)

        return text_array
