"""
605.433 Social Media Analysis
Final
Chin-Ting Ko

This program is a sentiment analysis classifier using the Sentiment 140 corpus and NLTK.
reference: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
"""

import nltk
import csv
import glob
import math
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.svm import LinearSVC

pos_tweets = []
neg_tweets = []

file1 = open('/Users/TimKo/Documents/final/testdata.manual.2009.06.14.csv')
reader = csv.reader(file1)

for line in reader:
    if line[0] == "4":
        pos_tweets.append((line[5], 'Positive'))
    if line[0] == "2":
        neg_tweets.append((line[5], 'Negative'))


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
list_of_test_negfiles = glob.glob('/Users/TimKo/Documents/final/test/neg/*')

for fileName in list_of_test_negfiles:
    test_neg_reader= open(fileName, "r").readline()
    test_neg_tweets.append((test_neg_reader, 'Negative'))

list_of_test_posfiles = glob.glob('/Users/TimKo/Documents/final/test/pos/*')

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

total_word_features = get_word_features(get_words_in_tweets(tweets_filtered))


def extract_features(document):
    document_words = set(document)
    features = {}
    for word in total_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

trainFeatures= tweets_filtered
training_set = nltk.classify.apply_features(extract_features, trainFeatures)
svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(training_set)
#nb_classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Classifier trained."
print ""

cutoff = int(math.floor((len(test_neg_tweets_filtered)*1/100 + len(test_pos_tweets_filtered)*1/100))/2)
#trainFeatures = neg_tweets_filtered[:cutoff] + pos_tweets_filtered[:cutoff]
testFeatures = test_neg_tweets_filtered[:cutoff] + test_pos_tweets_filtered[:cutoff]
testing_set = nltk.classify.apply_features(extract_features, testFeatures)

print 'Trained on %d instances, Tested on %d instances' % (len(trainFeatures), len(testFeatures))
#print 'NB Accuracy:', nltk.classify.accuracy(nb_classifier, testing_set)* 100, "%"
print 'SVM Accuracy:', nltk.classify.accuracy(svm_classifier, testing_set)* 100, "%"







