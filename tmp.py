## FEATURES ###
text = "HALLO hallo how are you? 😐😐😥 #hashtag #hash"

import emoji
import re

#Emooji
print("Emojis:", emoji.emoji_count(text))

#Hashtags
print("Hashtags:", len(re.findall("#",text)))

#Upper case
print("Upper case:", len([x for x in text.split() if x.isupper()]))

#Stopwords
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)
print("Stopwords:", len([x for x in text.split() if x in stopwords]))

#URL
print("URL:", len(re.findall("URL",text)))

#USER
print("URL:", len(re.findall("URL",text)))

#Punctuation

#Upper case
#Title case

#Stopwords
#Sentences
#Words
print(len(str(x)))
#Unique words
#Letters
