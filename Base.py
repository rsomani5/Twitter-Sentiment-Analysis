# Importing Libraries
import collections
import itertools
import tweepy
from textblob import TextBlob
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from nltk.corpus import stopwords
from nltk import bigrams
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.wordnet import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import networkx as nx
stop_words = stopwords.words('english')


# Set the key and token obtained from twitter developer account
consumer_key = "Vh9DCjF8WeMXqSgVxXTmKgvaJ"
consumer_secret = "YBNhg0HpVwxPfUYMnCKNVhojZ0sm06hoXrF4DpZ9RogrSJ2hOt"
access_token = "95426470-LNsn9Bdm2VZrmOqQ0hFC29ZEa2TdJ4oaxTBLFpkRb"
access_token_secret = "oelcsv1GUxf3pg52Mqruui7Rdye5CLLh2FXbAln9t3QyG"

# Create the authentication object
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Set the access token and access token secret
auth.set_access_token(access_token, access_token_secret)

# Creating the API object while passing in auth information
api = tweepy.API(auth)

# search Tweets on custom keyword on twitter
#public_tweet=api.search("corona",count=100,lang="en")

# ignore all retweets by adding -filter:retweets to your query.
search_words="corona"

#--------------------------------------------------------------------------------------------

#Assigning data
date_since = "2020-03-16"

Tweets = tweepy.Cursor(api.search,
              q=search_words+ " -filter:retweets",
              lang="en",
              since=date_since).items(1000)
print(Tweets)

streamData =  [[tweet.user.screen_name,tweet.user.location, tweet.text]  for tweet in Tweets]

df= pd.DataFrame(data=streamData,
                    columns=['user', "location","Tweets"])

#--------------------------------------------------------------------------------------------


# Create a function to clean the Tweets re.sub allows you to substitute a selection of characters defined using a regular expression, with something else.
def cleanTxt(text):
    text = re.sub('@[A-Za-z0–9]+', '', text)  # Removing @mentions
    text = re.sub('([^0-9A-Za-z \t])|(\w+:\/\/\S+)', '', text)  # Removing hyperlink and '#' tag
    text = re.sub('RT[\s]+', '', text)  # Removing RT
    return text

# Clean the Tweets
df['Tweets'] = df['Tweets'].apply(cleanTxt)

#--------------------------------------------------------------------------------------------

# Exploring the frequency of words

# splitting words in each tweet
words_in_tweet = [tweet.lower().split() for tweet in df['Tweets']]

stop_words = set(stopwords.words('english'))
# removing stop words

tweets_nsw = [[word for word in tweet_words if not word in stop_words]
              for tweet_words in words_in_tweet]

# Remove collection words
collection_words = search_words


tweets_nsw_nc = [[w for w in word if not w in collection_words]
                 for word in tweets_nsw]

# List of all words across Tweets

all_words = list(itertools.chain(*tweets_nsw_nc))

# Stemming the words to its root form using the lancasterStemmer
# lst = LancasterStemmer()
#
# all_words = [lst.stem(word) for word in all_words]

# Lemmatizing

from nltk.stem.wordnet import WordNetLemmatizer

wdnl= WordNetLemmatizer()

all_words = [wdnl.lemmatize(word) for word in all_words]

# Create counter
counts_no_words = collections.Counter(all_words)


wordcount_in_tweet = pd.DataFrame(counts_no_words.most_common(10),
                             columns=['words', 'count'])

# Plot horizontal bar graph

fig, ax = plt.subplots(figsize=(8, 8))
wordcount_in_tweet.sort_values(by='count').plot.barh(x='words',
                      y='count',
                      ax=ax,
                      color="purple")

ax.set_title("Common Words Found in Tweets (Including All Words)")
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()

#--------------------------------------------------------------------------------------------

#Explore Co-occurring Words (Bigrams)


# Create list of lists containing bigrams in Tweets
terms_bigram = [list(bigrams(tweet)) for tweet in tweets_nsw_nc]

# View bigrams for the first tweet
print(terms_bigram[0])

# Flatten list of bigrams in clean Tweets
bigrams = list(itertools.chain(*terms_bigram))

# Create counter of words in clean bigrams
bigram_counts = collections.Counter(bigrams)

print(bigram_counts.most_common(20))

bigram_df = pd.DataFrame(bigram_counts.most_common(20),
                             columns=['bigram', 'count'])

bigram_df

# Create dictionary of bigrams and their counts
d = bigram_df.set_index('bigram').T.to_dict('records')
# Create network plot
G = nx.Graph()

# Create connections between nodes
for k, v in d[0].items():
    G.add_edge(k[0], k[1], weight=(v * 10))

fig, ax = plt.subplots(figsize=(10, 8))

pos = nx.spring_layout(G, k=1)

# Plot networks
nx.draw_networkx(G, pos,
                 font_size=16,
                 width=3,
                 edge_color='grey',
                 node_color='purple',
                 with_labels=False,
                 ax=ax)

# Create offset labels
for key, value in pos.items():
    x, y = value[0] + .135, value[1] + .045
    ax.text(x, y,
            s=key,
            bbox=dict(facecolor='red', alpha=0.25),
            horizontalalignment='center', fontsize=13)

plt.show()

#--------------------------------------------------------------------------------------------

# Sentimental Analysis

# Create a function to get the subjectivity
def getSubjectivity(text):
    return TextBlob(text).sentiment.subjectivity


# Create a function to get the polarity
def getPolarity(text):
    return TextBlob(text).sentiment.polarity

# Create a function to get the polarity Positive scores
def getPosScores(text):
    return SentimentAnalyser.polarity_scores(text)["pos"]

# Create a function to get the polarity Negative scores
def getNegScores(text):
    return SentimentAnalyser.polarity_scores(text)["neg"]

# Create a function to get the polarity Neutral scores
def getNeuScores(text):
    return SentimentAnalyser.polarity_scores(text)["neu"]

# Create a function to get the polarity Compound scores
def getCompScores(text):
    return SentimentAnalyser.polarity_scores(text)["compound"]

SentimentAnalyser = SentimentIntensityAnalyzer()

# Create two new columns 'Subjectivity' & 'Polarity'
df['Subjectivity'] = df['Tweets'].apply(getSubjectivity)
df['Polarity'] = df['Tweets'].apply(getPolarity)
df['Positive'] = df['Tweets'].apply(getPosScores)
df['Negative'] = df['Tweets'].apply(getNegScores)
df['Neutral'] = df['Tweets'].apply(getNeuScores)
df['compound']= df['Tweets'].apply(getCompScores)

# Plotting the polarity and subjectivity scores
plt.figure(figsize=(8, 8))
for i in range(0, df.shape[0]):
    plt.scatter(df["Polarity"][i], df["Subjectivity"][i], color='Blue')  # plt.scatter(x,y,color)

plt.title('Sentiment Analysis')
plt.xlabel('Polarity')
plt.ylabel('Subjectivity')
plt.show()

# Create a function to compute negative (-1), neutral (0) and positive (+1) for TextBlobAnalysis analysis
def getTextBlobAnalysis(score):
 if score < 0:
  return 'Negative'
 elif score == 0:
  return 'Neutral'
 else:
  return 'Positive'

df['TextBlobAnalysis'] = df['Polarity'].apply(getTextBlobAnalysis)

# Create a function to compute negative, neutral and positive  VADER analysis
def getVADERAnalysis(score):
 if score <= -0.5:
  return 'Negative'
 elif score >= 0.5:
  return 'Positive'
 else:
  return 'Neutral'

# Storing the sentiment from VADER (Valence Aware Dictionary and sEntiment Reasoner)
df['VADERAnalysis'] = df['compound'].apply(getVADERAnalysis)


# Plotting and visualizing the counts of Text Blob Analysis
plt.figure(figsize=(8, 8))
plt.title('TextBlob Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.gcf().subplots_adjust(bottom=0.15)
df['TextBlobAnalysis'].value_counts().plot(kind = 'bar')
plt.show()

# Plotting and visualizing the counts of VADER Analysis
plt.figure(figsize=(8, 8))
plt.title('VADER Sentiment Analysis')
plt.xlabel('Sentiment')
plt.ylabel('Counts')
plt.gcf().subplots_adjust(bottom=0.15)
df['VADERAnalysis'].value_counts().plot(kind = 'bar')
plt.show()

#--------------------------------------------------------------------------------------------

# Building and plotting word cloud

# List of all words across Tweets
allWords = ' '.join(all_words)
wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=100,background_color="white").generate(allWords)
plt.figure(figsize=(8, 8))
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.gcf().subplots_adjust(bottom=0.15)
plt.show()

#Saving the data into csv file
df.to_csv('twitterMining.csv')


