from collections import Counter, defaultdict
import math

START_MARK = "\1"
END_MARK = "\2"

def load_probs(model_file_path):
    with open(model_file_path, "r") as f:
        model = f.read()

    probs = dict()
    for line in model.split("\n"):
        try:
            chars, prob = line.rsplit("\t", 1)
        except ValueError:
            # This is an empty line separating between blocks of different n-grams
            continue
        probs[chars] = float(prob)

    return probs

def eval(input_file, model_file, weights):
    # Make sure the weights are valid
    assert len(weights) == 3, "Must supply 3 weights"
    assert sum(weights) == 1, "The sum of the weights must be 1"
    assert len(filter(lambda weight: weight < 0 or weight > 1, weights)) == 0, "All weights must be between 0 to 1 (inclusive)"

    probs = load_probs(model_file)
    unknown_prob = probs.get("<UNK>", 0)

    # Prepare the test text
    with open(input_file, "r") as f:
        test_text = f.read()

    # Find the probablities of n-grams for n from 1 to 3
    probs_inter = dict()
    for i in range(2, len(test_text) - 1):
        # "Cut" the three sequences that end with the current character from test_text
        grams = [test_text[i - n_gram + 1 : i + 1] for n_gram in range(1, 4)]        
        # Calculate the interpolation of probabilities of the three n-grams
        probs_inter[test_text[i - 2 : i + 1]] = \
            weights[2] * probs.get(grams[0], unknown_prob) + weights[1] * probs.get(grams[1], unknown_prob) + weights[0] * probs.get(grams[2], unknown_prob)

    # Filter out the 0 probabilities - this is relevant when there's no smoothing
    probs_inter = {text : probs_inter[text] for text in probs_inter if probs_inter[text] != 0}
    # Calculate the log probabilities
    log_probs = [math.log(prob, 2) for prob in probs_inter.values()]
    # Calculate the perplexity
    perplexity = 2 ** (-sum(log_probs) / len(log_probs))
    return perplexity
