import logging
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema
import math
import pysbd

seg = pysbd.Segmenter(language="en", clean=False)
model = SentenceTransformer('all-mpnet-base-v2')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class Paragraphs:

    def __init__(self, transcript: str):
        self.transcript = transcript

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

        # Embed sentences
        embeddings = model.encode(sentences)

        # Create similarities matrix
        similarities = cosine_similarity(embeddings)

        # Normalize the embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # embeddings = embeddings / norms

        activated_similarities = self.activate_similarities(similarities, p_size=2)

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
            text_array.append(current_text.strip())

        return text_array
