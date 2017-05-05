"""
605.433 Social Media Analysis
Final
Chin-Ting Ko

This program is a sentiment analysis classifier using the Sentiment 140 corpus and NLTK.
reference: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
"""

import nltk
import glob
import math
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

pos_tweets = []
neg_tweets = []

list_of_negfiles = glob.glob('train/neg/*')

for fileName in list_of_negfiles:
    neg_reader= open(fileName, "r").readline()
    neg_tweets.append((neg_reader, 'Negative'))

list_of_posfiles = glob.glob('train/pos/*')

for fileName in list_of_posfiles:
        pos_reader = open(fileName, "r").readline()
        pos_tweets.append((pos_reader, 'Positive'))

#print neg_tweets
print "Training set loaded."
neg_tweets_filtered = []
for (words, sentiment) in neg_tweets:
    words_filtered = [e.lower() for e in words.split() if (len(e) > 3)]
    neg_tweets_filtered.append((words_filtered, sentiment))

pos_tweets_filtered = []
for (words, sentiment) in pos_tweets:
    words_filtered = [e.lower() for e in words.split() if (len(e) > 3)]
    pos_tweets_filtered.append((words_filtered, sentiment))

tweets_filtered = neg_tweets_filtered + pos_tweets_filtered

print "Training set filtered."

test_pos_tweets = []
test_neg_tweets = []
list_of_test_negfiles = glob.glob('test/neg/*')

for fileName in list_of_test_negfiles:
    test_neg_reader= open(fileName, "r").readline()
    test_neg_tweets.append((test_neg_reader, 'Negative'))

list_of_test_posfiles = glob.glob('test/pos/*')

for fileName in list_of_test_posfiles:
        test_pos_reader = open(fileName, "r").readline()
        test_pos_tweets.append((test_pos_reader, 'Positive'))

print "Testing set loaded."

test_neg_tweets_filtered = []
for (words, sentiment) in test_neg_tweets:
    words_filtered = [e.lower() for e in words.split() if (len(e) > 3)]
    test_neg_tweets_filtered.append((words_filtered, sentiment))

test_pos_tweets_filtered = []
for (words, sentiment) in test_pos_tweets:
    words_filtered = [e.lower() for e in words.split() if (len(e) > 3)]
    test_pos_tweets_filtered.append((words_filtered, sentiment))

test_tweets_filtered = test_neg_tweets_filtered + test_pos_tweets_filtered
print "Testing set filtered."

stop_words = set(stopwords.words('english'))


def get_words_in_tweets(words_filtered):
    all_words = []
    for (words, sentiment) in words_filtered:
        all_words.extend(words)
    return all_words


def do_stemming(filtered):
    stemmed = []
    filtered_not_stopwords = (words for words in filtered if words not in stop_words)
    for f in filtered_not_stopwords:
        stemmed.append(SnowballStemmer('english').stem(f.decode("ISO-8859-1")))
    return stemmed


def get_word_features(all_words):
    words_list = nltk.FreqDist(all_words)
    word_features = words_list.keys()
    return word_features


#neg_word_features = get_word_features(do_stemming(get_words_in_tweets(neg_tweets_filtered)))
#pos_word_features = get_word_features(do_stemming(get_words_in_tweets(pos_tweets_filtered)))
total_word_features = get_word_features(get_words_in_tweets(tweets_filtered))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in total_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#selects 1/10 of the features to be used for training
cutoff = int(math.floor((len(neg_tweets_filtered)*1/10 + len(pos_tweets_filtered)*1/10))/2)
trainFeatures = neg_tweets_filtered[:cutoff] + pos_tweets_filtered[:cutoff]
#trainFeatures= tweets_filtered
training_set = nltk.classify.apply_features(extract_features, trainFeatures)
#svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(training_set)
nb_classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Classifier trained."
print ""

inputfile= open('twitterMovie.txt')
input = inputfile.read()
#input = "For Keloglan Kara Prens'e Karsi, I thought that, this movie is the worst movie ever made and in my whole life, I won't be able to see another one this bad. But now I understand, I was totally wrong.  is much more successful in being a bad movie than Keloglan Kara Prens'e Karsi. Now I think, as long as the Turkish film producers continue to make movies, I'll continue to see unbelievably bad movies. The money, which I gave for this film was not much, but I feel sorry for my 2 hours spent for nothing. Yes, this movie is nothing. Please take my recommendation: Even if someone gives you money to watch this movie, don't even think about it. I can tell you numerous reasons about why I like a movie. But for this one, there's simply nothing to say. It was supposed to be a comedy film, but I didn't even smile!! Come on guys, is the script writer an 8 year old boy or what? What were you thinking? The original cult film "" was also a bad movie but at least it was funny, not boring. There were scenes, which one wants to see. But for this, there's nothing to say. I assume that, this film is a bad joke for Turkish people."

#print "Test Data:", input
print "Test Result: ", nb_classifier.classify(extract_features(input.split()))








