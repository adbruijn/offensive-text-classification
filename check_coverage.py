from tqdm import tqdm
import pandas as pd
from data_loader import load_embeddings
from gensim.models import KeyedVectors
import operator
import wordsegment
import re
import contractions
from nltk.tokenize import TweetTokenizer
from wordsegment import load, segment
from ekphrasis.classes.tokenizer import SocialTokenizer
import emoji_extra

tt = TweetTokenizer(preserve_case=False,
                    strip_handles=True,
                    reduce_len=True)

social_tokenizer = SocialTokenizer(lowercase=True).tokenize

load()

train = pd.read_csv("data/SemEval/olid-training-v1.0.tsv", delimiter="\t")

def build_vocab(sentences, verbose =  True):
    """
    :param sentences: list of list of words
    :return: dictionary of words and their count
    """
    vocab = {}
    for sentence in tqdm(sentences, disable = (not verbose)):
        for word in sentence:
            try:
                vocab[word] += 1
            except KeyError:
                vocab[word] = 1
    return vocab

def check_coverage(vocab,embeddings_index):
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
        try:
            a[word] = embeddings_index[word]
            k += vocab[word]
        except:

            oov[word] = vocab[word]
            i += vocab[word]
            pass

    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))
    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))
    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]

    return sorted_x

embedding_file_fasttext = 'data/fastText/wiki-news-300d-1M.vec'
embedding_file_glove_twitter = 'data/GloVe/glove.twitter.27B.200d.txt'
embedding_file_glove = 'data/GloVe/glove.840B.300d.txt'
embedding_file_word2vec = 'data/Word2Vec/GoogleNews-vectors-negative300.bin'

embeddings_index_fasttext = load_embeddings(embedding_file_fasttext)
embeddings_index_glove_twitter = load_embeddings(embedding_file_glove_twitter)
embeddings_index_glove = load_embeddings(embedding_file_glove)
embeddings_index_word2vec = KeyedVectors.load_word2vec_format(embedding_file_word2vec, binary=True)

sentences = train["tweet"].apply(lambda x: x.split()).values
vocab = build_vocab(sentences)

print("fastText")
oov_fasttext = check_coverage(vocab,embeddings_index_fasttext)
print("Glove Twitter")
oov_glove_twitter = check_coverage(vocab,embeddings_index_glove_twitter)
print("GloVe Common Crawl")
oov_glove = check_coverage(vocab,embeddings_index_glove)
print("Word2Vec")
oov_word2vec = check_coverage(vocab,embeddings_index_word2vec)

def segment_hashtag_word2vec(match):
    """
    Takes a re 'match object' and segments the hashtag
    """

    hashtag_body = match.group(1)
    result = " hashtag {} ".format(" ".join(segment(hashtag_body)))
    return result

def segment_hashtag_glove(match):
    """
    Takes a re 'match object' and segments the hashtag
    """

    hashtag_body = match.group(1)
    if hashtag_body.isupper():
        result = "<hashtag> {} <allcaps>".format(" ".join(segment(hashtag_body)))
    else:
        result = "<hashtag> {} ".format(" ".join(segment(hashtag_body)))

    return result

def segment_hashtag_fasttext(match):
    """
    Takes a re 'match object' and segments the hashtag
    """

    hashtag_body = match.group(1)
    if hashtag_body.isupper():
        result = " hashtag {} allcaps ".format(" ".join(segment(hashtag_body)))
    else:
        result = " hashtag {} ".format(" ".join(segment(hashtag_body)))

    return result

def clean_glove(text):

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", '<user>', text)

    #Remove URL
    text = re.sub("\s*URL\s*", '<url>', text)

    #Numbers
    text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)

    #Emoticons
    eyes = r"[8:=;]"
    nose = r"['`\-]?"

    text = re.sub(r"{}{}[)dD]+|[)dD]+{}{}".format(eyes, nose, nose, eyes), "<smile>", text)
    text = re.sub(r"{}{}p+".format(eyes, nose), "<lolface>", text)
    text = re.sub(r"{}{}\(+|\)+{}{}".format(eyes, nose, nose, eyes), "<sadface>", text)
    text = re.sub(r"{}{}[\/|l*]".format(eyes, nose), "<neutralface>", text)

    #Emojis
    text = emoji_extra.demojize(text)

    #Remove skin tones
    text = re.sub(" medium-dark skin tone", "", text)
    text = re.sub(" medium-light skin tone", "", text)
    text = re.sub(" medium skin tone", "", text)
    text = re.sub(" dark skin tone", "", text)
    text = re.sub(" light skin tone", "", text)

    #Hashtag
    #Source: https://pypi.org/project/wordsegment/
    text = re.sub(r"#(\w+)", segment_hashtag_glove, text)

    #Repeat
    # Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
    text = re.sub(r"([!?.]){2,}", r"\1 <repeat>", text)

    #Elong
    # Mark elongated words (eg. "wayyyy" => "way <ELONG>")
    text = re.sub(r"\b(\S*?)(.)\2{2,}\b", r"\1\2 <elong>", text)

    #Lowercase
    text = re.sub(r"([A-Z]){2,}", text.lower() + " <allcaps>", text)
    text = text.lower()

    #Replace contractions and slang of word
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = re.sub("i'm", "i am", text)
    text = re.sub("i've", "i have", text)
    text = contractions.fix(text, slang=True)

    #text = " ".join(tt.tokenize(text))
    text = " ".join(social_tokenizer(text))

    return text

def clean_word2vec(text):

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", ' user ', text)

    #Remove URL
    text = re.sub("\s*URL\s*", ' url ', text)

    #Numbers
    #text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)

    #Emojis
    text = emoji_extra.demojize(text)

    #Remove skin tones
    text = re.sub(" medium-dark skin tone", "", text)
    text = re.sub(" medium-light skin tone", "", text)
    text = re.sub(" medium skin tone", "", text)
    text = re.sub(" dark skin tone", "", text)
    text = re.sub(" light skin tone", "", text)

    #Hashtag
    #Source: https://pypi.org/project/wordsegment/
    text = re.sub(r"#(\w+)", segment_hashtag_word2vec, text)

    #Punctuation
    text = re.sub("&amp;", "and", text)
    text = re.sub("&lt;", "<", text)
    text = re.sub("&gt", ">", text)
    text = re.sub("&", "and", text)

    #Lowercase
    text = text.lower()

    #Replace contractions and slang of word
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = re.sub("i'm", "i am", text)
    text = re.sub("i've", "i have", text)
    text = contractions.fix(text, slang=True)

    #Punctuation
    text = re.sub(r'[^\w\s]', ' ', text)

    #text = " ".join(tt.tokenize(text))
    stopwords = ['to','and','a','of']
    tokenized_text = social_tokenizer(text)
    filtered_stopwords = [w for w in tokenized_text if w not in stopwords]

    text = " ".join(filtered_stopwords)

    return text

def clean_fasttext(text):

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", ' user ', text)

    #Remove URL
    text = re.sub("\s*URL\s*", ' url ', text)

    #Numbers
    #text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", "<number>", text)

    #Emojis
    text = emoji_extra.demojize(text)

    #Remove skin tones
    text = re.sub(" medium-dark skin tone", "", text)
    text = re.sub(" medium-light skin tone", "", text)
    text = re.sub(" medium skin tone", "", text)
    text = re.sub(" dark skin tone", "", text)
    text = re.sub(" light skin tone", "", text)

    #Hashtag
    #Source: https://pypi.org/project/wordsegment/
    text = re.sub(r"#(\w+)", segment_hashtag_fasttext, text)

    #Lowercase
    text = re.sub(r"([A-Z]){2,}", text.lower() + " allcaps ", text)
    text = text.lower()

    #Replace contractions and slang of word
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")

    text = re.sub("i'm", "i am", text)
    text = re.sub("i've", "i have", text)
    text = contractions.fix(text, slang=True)

    #text = " ".join(tt.tokenize(text))
    text = " ".join(social_tokenizer(text))

    return text

train["clean_tweet_word2vec"] = train["tweet"].apply(lambda x: clean_word2vec(x))
sentences_word2vec = train["clean_tweet_word2vec"].apply(lambda x: x.split())
vocab_word2vec_clean = build_vocab(sentences_word2vec)

train["clean_tweet_glove_twitter"] = train["tweet"].apply(lambda x: clean_glove(x))
sentences_glove_twitter = train["clean_tweet_glove_twitter"].apply(lambda x: x.split())
vocab_glove_twitter_clean = build_vocab(sentences_glove_twitter)

train["clean_tweet_glove"] = train["tweet"].apply(lambda x: clean_glove(x))
sentences_glove = train["clean_tweet_glove"].apply(lambda x: x.split())
vocab_glove_clean = build_vocab(sentences_glove)

train["clean_tweet_fasttext"] = train["tweet"].apply(lambda x: clean_fasttext(x))
sentences_fasttext = train["clean_tweet_fasttext"].apply(lambda x: x.split())
vocab_fasttext_clean = build_vocab(sentences_fasttext)

print("Word2Vec")
oov_word2vec_clean = check_coverage(vocab_word2vec_clean,embeddings_index_word2vec)
print("fastText")
oov_fasttext_clean = check_coverage(vocab_fasttext_clean,embeddings_index_fasttext)
print("GloVe twitter")
oov_glove_twitter_clean = check_coverage(vocab_glove_twitter_clean,embeddings_index_glove_twitter)
print("Glove Common Crawl")
oov_glove_clean = check_coverage(vocab_glove_clean,embeddings_index_glove)
