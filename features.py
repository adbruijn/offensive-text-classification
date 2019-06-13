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
    print(features["count_hashtag"])

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
