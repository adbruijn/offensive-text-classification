def remove_emoji(tweet):
    tweet_emojis = emojis.get(tweet)

    not_emoji = []

    for token in tweet.split(" "):
        if token not in tweet_emojis:
            not_emoji.append(token)

    not_emoji = ' '.join(not_emoji)
    return(not_emoji)

stopwords = set(STOPWORDS)

features = pd.DataFrame()

def generate_features(df):

    #URL
    features["count_mention"] = df["text"].apply(lambda x: len(re.findall("URL", str(x))))
    print(features["count_mention"])

    #Hastags
    features["count_hashtag"] = df["text"].apply(lambda x: len(re.findall("#", str(x))))
    print(features["count_hashtag"]

    #!Emoticons
    features["count_emojis"] = df["text"].apply(lambda x: emoji.emoji_count(x))
    print(features["count_emojis"])

    #Remove URLS, # and mentions
    features["text"] = df["text"].apply(lambda x: re.sub("@USER", ' ', str(x)))
    print(features["count_emojis"])

    #Remove URL
    features["text"] = df["text"].apply(lambda x: re.sub("URL", ' ', str(x)))
    print(features["count_emojis"])

    #Remove hashtag + following word(s)
    features["text"] = df["text"].apply(lambda x: re.sub("#\w+", ' ', str(x)))
    print(features["count_stopwords"])

    #Remove emojis
    features["text"] = df["text"].apply(lambda x: re.sub(":\w+:",' ',emojis.decode(x)))
    print(features["count_stopwords"])

    #Punctuation
    features["count_punctuations"] = df["text"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    print(features["count_stopwords"])

    #Remove punctuation
    features["text"] = df["text"].apply(lambda x: re.sub("[^a-zA-z]", ' ', str(x)))
    print(features["count_stopwords"])

    #Upper case
    features["count_words_upper"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    print(features["count_stopwords"])

    #Title case
    features["count_words_title"] = df["text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    print(features["count_stopwords"])

    #Stopwords
    features["count_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in stopwords]))
    print(features["count_stopwords"])

    #Sentences
    features['count_sent'] = df["text"].apply(lambda x: len(re.findall("\n",str(x)))+1)
    print(features["count_sent"])

    #Words
    features['count_word'] = df["text"].apply(lambda x: len(str(x).split()))
    print(features["count_word"])

    #Unique words
    features['count_unique_word'] = df["text"].apply(lambda x: len(set(str(x).split())))
    print(features["count_unique_word"])

    #Letters
    features['count_letters'] = df["text"].apply(lambda x: len(str(x)))
    print(features["count_letters"])

    return(features)


#Preprocess Data
def clean_text(text, remove_punt_number_special_chars=False,remove_stopwords=False, apply_stemming=False):
    """Clean text
    Args:
        text: (str) Text
        remove_punt_number_special_chars: (bool) Remove punctuations, numbers and special characters
        remove_stopwords: (bool) Remove stopwords
        apply_stemming: (bool) Apply stemming on the words on the text
    """
    #Remove emojis
    text = re.sub(":[a-zA-Z\-\_]*:","", emoji.demojize(text)) #:hear-no-evil_monkey:
    text = re.sub(":\w+:","", emoji.demojize(text))
    text = re.sub(":\w+\’\w+:","", emoji.demojize(text)) #:woman's_boot:

    #Remove mentions, usernames (@USER)
    text = re.sub("\s*@USER\s*", '', text)

    #Remove URL
    text = re.sub("\s*URL\s*", '', text)

    #And
    text = re.sub("&amp;", "and", text)
    text = re.sub("&lt;", "<", text)
    text = re.sub("&gt", ">", text)
    text = re.sub("&", "and", text)

    #Replace contractions and slang of word
    text = re.sub("i'm", "I'm", text)
    text = contractions.fix(text, slang=True)

    #Lowercase
    text = text.lower()

    #Remove Hashtags + Words
    text = re.sub("#\s*\w+\s*", '', text)

    #Remove repeating whitespaces
    text = re.sub("\s[2, ]"," ", text)

    #Remove non ascii characters
    text.encode("ascii", errors="ignore").decode()

    #Remove punctuations, numbers and special characters (remove emoticons)
    if remove_punt_number_special_chars:
        text = re.sub('[^a-zA-Z]', ' ', text)

    #Tokenize text
    tt = TweetTokenizer(preserve_case=False,
                    strip_handles=True,
                    reduce_len=True)

    text_tokens = tt.tokenize(text)

    #Remove stopwords
    if remove_stopwords:
        stopwords = set(STOPWORDS)
        text_tokens = [token for token in text_tokens if token not in stopwords]

    #Stemming
    if apply_stemming:
        text_stem = [stemmer.stem(token) for token in text_tokens]

    clean = " ".join(text_tokens)

    return clean
