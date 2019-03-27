from collections import Counter, defaultdict
import math

UNKNOWN_MARK = "<UNK>"
# The vocabulary size is chosen to be the max amount of characters that can be represented by one byte (which is the
# size of character for this code)
VOCABULARY_SIZE = 256

def count_sequences(corpus_file, sequences_size):
    """
    Count the occurences of each sequence of length sequences_size.
    If sequences_size is 0, the returned dict contains a key of empty string
    and the value is the amount of characters in the corpus.
    """
    count_sequences = defaultdict(lambda:0)
    for i in range(0, len(corpus_file) - sequences_size):
        t = corpus_file[i:i+sequences_size]
        count_sequences[t] += 1
    return count_sequences

def n_gram_count(corpus_file, n_gram):
    n_counts = defaultdict(Counter)
    for i in range(0, len(corpus_file) - n_gram + 1):
        t = corpus_file[i:i+n_gram-1]
        n_char = corpus_file[i+n_gram-1]
        n_counts[t][n_char] += 1
    return n_counts

def lm(corpus_file, model_file):
    with open(corpus_file, "r") as data_file:
        corpus_file = data_file.read()
    model_file = open(model_file, "w")

    # Create the model (probabilities) for each of the three n-grams
    for i in range(3, 0, -1):
        n_counts = n_gram_count(corpus_file, i)
        letter_count = count_sequences(corpus_file, i-1)

        # Add the probability for unknown (unseen) characters, using Add One (Laplace) Smoothing
        if i == 1:
            model_file.write(UNKNOWN_MARK + "\t" + str(1.0 / (letter_count.values()[0] + VOCABULARY_SIZE)) + "\n")

        for a in n_counts:
            for b in n_counts[a]:
                if i == 1:
                    prob = float(n_counts[a][b] + 1) / (letter_count[a] + VOCABULARY_SIZE)
                else:
                    prob = float(n_counts[a][b]) / letter_count[a]
                model_file.write(a + b + "\t" + str(prob) + "\n")
        model_file.write("\n")

    model_file.close()
