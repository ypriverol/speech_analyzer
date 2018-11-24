from nltk.corpus import stopwords

from models.constants import get_stop_words


def get_clean_word_list(text):
    """
    This method return a clean list of words from a text tokenize.
    All the stop words are removed.
    :param text: text
    :return: list of words
    """
    word_list = []
    stop_words = stopwords.words('spanish')
    for word in get_stop_words():
        stop_words.append(word)

    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.
    for word in text.lower().split():
        word = word.replace(".", "")
        word = word.replace(",", "")
        word = word.replace(":", "")
        word = word.replace("\"", "")
        word = word.replace("!", "")
        word = word.replace("â€œ", "")
        word = word.replace("â€˜", "")
        word = word.replace("*", "")
        if word not in stop_words:
            word_list.append(word)

    return word_list