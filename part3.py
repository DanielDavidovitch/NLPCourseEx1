from part1 import lm
from part2 import eval
import os

START_MARK = "\1"
END_MARK = "\2"
LANGUAGES = ["en", "es", "fr", "in", "it", "nl", "pt", "tl"]

def create_train_and_test_files():
    for lang in LANGUAGES:
        print "Creating training and test files for language {0}".format(lang)

        with open("data/{0}.csv".format(lang)) as f:
            data = f.read()

        lines = data.split("\n")
        train_text = END_MARK
        test_text = END_MARK

        for i in xrange(1, len(lines)):
            # Split the line to tweet ID and tweet content
            split = lines[i].split(",", 1)
            if len(split) != 2:
                continue
            # Take the first 90% of the tweets for training, and the rest for 10%
            if i < 0.9 * len(lines):
                train_text += START_MARK + split[1] + END_MARK
            else:
                test_text += START_MARK + split[1] + END_MARK
        train_text += START_MARK
        test_text += START_MARK

        # Save the training text and testing text to files
        if not os.path.exists("train_and_test_data"):
            os.mkdir("train_and_test_data")
        with open(r"train_and_test_data/{0}_train.csv".format(lang), "w") as f:
            f.write(train_text)
        with open(r"train_and_test_data/{0}_test.csv".format(lang), "w") as f:
            f.write(test_text)

def test_languages(weights):
    SEP_CHAR = "|"
    result = "   " + SEP_CHAR + SEP_CHAR.join(LANGUAGES) + SEP_CHAR + "\n"

    if not os.path.exists("result"):
        os.mkdir("result")

    for train_lang in LANGUAGES:
        print "Training {0}".format(train_lang)
        result += SEP_CHAR + train_lang + SEP_CHAR
        model_path = "result/{0}.model".format(train_lang)
        train_path = "train_and_test_data/{0}_train.csv".format(train_lang)
        # Create the model file for this train language
        lm(train_path, model_path)
        for test_lang in LANGUAGES:
            print "Testing {0} on {1} model".format(test_lang, train_lang)
            perplexity = eval("train_and_test_data/{0}_test.csv".format(test_lang), model_path, weights)
            result += str(perplexity) + SEP_CHAR
        result += "\n"

    with open("result/part3_result.txt", "w") as f:
        f.write(result)

    return result

def main():
    weights = [0.4, 0.3, 0.3]

    # Create train data and test data for each language from the given data
    print "Creating training and test files"
    create_train_and_test_files()
    # Test the model of each language against all languages
    print "Testing languages with weights: {0}".format(weights)
    test_languages(weights)


if __name__ == '__main__':
    main()
