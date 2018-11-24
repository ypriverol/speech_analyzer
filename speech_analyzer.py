import glob
from collections import defaultdict
from pathlib import Path

import nltk
import os

from matplotlib import collections
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import collections
from plotly.offline.offline import matplotlib
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

from nltk.corpus import stopwords

from models.constants import get_stop_words
from models.utils import get_clean_word_list

plt.style.use('ggplot')

import codecs, sys

from nltk.stem.wordnet import WordNetLemmatizer
import html2text


def get_wordnet_tag(tag):
    """
  Maps a tag from the Treebank tagger to a WordNet tag. If no match is
  found, the function returns None.
  """
    if tag.startswith('JJ'):
        return 'a'
    elif tag.startswith('RB') or tag == "WRB":
        return 'r'
    elif tag.startswith('NN') or tag.startswith("WP"):
        return 'n'
    elif tag.startswith('VB'):
        return 'v'
    else:
        return None


speeches = dict()
for filename in glob.glob("Data/*.txt"):
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    name = name.replace("_", " ")
    year = name[:4]
    name = year + "-" + name[5:]
    with open(filename) as f:
        speech = f.read()
        speeches[name] = speech

lemmatizer = WordNetLemmatizer()

num_sentences = dict()
num_words = dict()
avg_sentence_len = dict()
num_unique_lemmas = dict()
num_unique_words = dict()

print("# YYYY name num_sentences num_words avg_sentence_len num_unique_words num_unique_lemmas")

for president in sorted(speeches.keys()):
    speech = speeches[president]
    sentences = sent_tokenize(speech)
    words = word_tokenize(speech)

    avg_sentence_len[president] = 0.0
    for sentence in sentences:
        avg_sentence_len[president] += len(word_tokenize(sentence))
    avg_sentence_len[president] /= len(sentences)

    num_sentences[president] = len(sentences)
    num_words[president] = len(words)
    num_unique_words[president] = len(set(words))

    tagged = nltk.pos_tag(words)
    lemmas = set()

    for word, tag in tagged:
        pos = get_wordnet_tag(tag)
        if pos:
            lemmas.add(lemmatizer.lemmatize(word, pos=pos))
        else:
            lemmas.add(word)

    year = int(president[:4])
    name = president[5:]

    num_unique_lemmas[president] = len(lemmas)
    print('%d "%s" %d %d %f %d %d' % (
        year, name, num_sentences[president], num_words[president], avg_sentence_len[president],
        num_unique_words[president], num_unique_lemmas[president]))


def generate_speeches_per_year(speeches):
    x = []
    y = []
    for key in sorted(speeches.items()):
        value = len(key[1])
        x.append(key[0])
        y.append(value)
    x_pos = [i for i, _ in enumerate(x)]
    plt.bar(x_pos, y, color='green')
    plt.xlabel("Year")
    plt.ylabel("Number of Speeches")
    plt.title("Fidel's Speech Per Year")

    plt.xticks(x_pos, x)
    plt.setp(plt.gca().get_xticklabels(), rotation=90, horizontalalignment='right')
    plt.savefig('output/number-speeches-per-year.png')


def generate_world_cloud(speeches):
    # Generate a word cloud image
    stop_words = stopwords.words('spanish')
    text = ""
    for key in speeches:
        for value in speeches[key]:
            text = text + value

    for word in stop_words:
        text = text.replace(" " + word + " ", " ")

    wordcloud = WordCloud().generate(text)

    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    # lower max_font_size
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()


def compute_word_count(text, number_words, year):
    stop_words = stopwords.words('spanish')
    for word in get_stop_words():
        stop_words.append(word)

    wordcount = {}
    # To eliminate duplicates, remember to split by punctuation, and use case demiliters.
    word_list = get_clean_word_list(text)
    for word in word_list:
        if word not in wordcount:
            wordcount[word] = 1
        else:
            wordcount[word] += 1

    print("\nOK. The {} most common words in " + year + " are as follows\n".format(number_words))
    word_counter = collections.Counter(wordcount)

    for word, count in word_counter.most_common(number_words):
        print(word, ": ", count)

    lst = word_counter.most_common(number_words)
    new_style = {'grid': False}
    matplotlib.rc('axes', **new_style)
    df = pd.DataFrame(lst, columns=['Word', 'Count'])
    df.plot.bar(x='Word', y='Count', title='More frequently words in -- ' + year)
    plt.savefig('output/frequently-words-' + str(year) + ".png")


def most_common_words_per_year(speeches, number_words):
    word_count = defaultdict(list)
    for key in speeches:
        text = ""
        for speech in speeches[key]:
            text = text + " " + speech
        word_count[key] = compute_word_count(speech, number_words, key)


def main():
    speeches = defaultdict(list)
    pathlist = Path('data/').glob('**/*.txt')
    for path in pathlist:
        if "esp" in str(path):
            year = str(path).split('/', 2)[1]

            with open(str(path), "r", encoding='ISO-8859-1', errors='ignore') as f:
                raw = f.read()
            print(raw)
            speeches[year].append(raw)

    for key in speeches:
        print(str(key) + " : " + str(len(speeches[key])))

    generate_speeches_per_year(speeches)

    most_common_words_per_year(speeches, 50)

    generate_world_cloud(speeches)


if __name__ == "__main__":
    main()
