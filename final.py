"""
605.433 Social Media Analysis
Final
Chin-Ting Ko

This program is a sentiment analysis classifier using NLTK.
reference: http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
"""

import nltk
import glob
import math
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.svm import LinearSVC

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


def do_stemming(filtered):
    stemmed = []
    for f in filtered:
        stemmed.append(SnowballStemmer('english').stem(f))
    return stemmed

stop_words = set(stopwords.words('english'))

neg_tweets_filtered = []
for (words, sentiment) in neg_tweets:
    words_filtered = [e.lower().decode('ISO-8859-1') for e in words.split() if (len(e) > 3)]
    words_filtered_stemmed = do_stemming(words_filtered)
    neg_tweets_filtered.append((words_filtered_stemmed, sentiment))

pos_tweets_filtered = []
for (words, sentiment) in pos_tweets:
    words_filtered = [e.lower().decode('ISO-8859-1') for e in words.split() if (len(e) > 3) ]
    words_filtered_stemmed = do_stemming(words_filtered)
    pos_tweets_filtered.append((words_filtered_stemmed, sentiment))

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
    words_filtered = [e.lower().decode('ISO-8859-1') for e in words.split() if (len(e) > 3)]
    words_filtered_stemmed = do_stemming(words_filtered)
    test_neg_tweets_filtered.append((words_filtered_stemmed, sentiment))

test_pos_tweets_filtered = []
for (words, sentiment) in test_pos_tweets:
    words_filtered = [e.lower().decode('ISO-8859-1') for e in words.split() if (len(e) > 3)]
    words_filtered_stemmed = do_stemming(words_filtered)
    test_pos_tweets_filtered.append((words_filtered_stemmed, sentiment))

test_tweets_filtered = test_neg_tweets_filtered + test_pos_tweets_filtered
print "Testing set filtered."


def get_words_in_tweets(words_filtered):
    all_words = []
    for (words, sentiment) in words_filtered:
        all_words.extend(words)
    return all_words


def get_word_features(all_words):
    words_list = nltk.FreqDist(all_words)
    word_features = words_list.keys()
    return word_features

#print "neg_tweets: ", neg_tweets

#neg_word_features = get_word_features(do_stemming(get_words_in_tweets(neg_tweets_filtered)))
#pos_word_features = get_word_features(do_stemming(get_words_in_tweets(pos_tweets_filtered)))
#total_word_features= words_filtered

#print total_word_features
#print do_stemming(get_words_in_tweets(neg_tweets_filtered))
total_word_features = get_word_features(get_words_in_tweets(tweets_filtered))

#print neg_word_features
#print "total_word_features:", total_word_features

def extract_features(document):
    document_words = set(document)
    features = {}
    for word in total_word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

#selects 1/10 of the features to be used for training
cutoff = int(math.floor((len(neg_tweets_filtered)*2/10 + len(pos_tweets_filtered)*2/10))/2)
trainFeatures = neg_tweets_filtered[:cutoff] + pos_tweets_filtered[:cutoff]

#trainFeatures = do_stemming(get_words_in_tweets(neg_tweets_filtered[:cutoff])) + do_stemming(get_words_in_tweets(pos_tweets_filtered[:cutoff]))
#trainFeatures= tweets_filtered
training_set = nltk.classify.apply_features(extract_features, trainFeatures)

svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(training_set)
nb_classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Classifier trained."
print ""

#testFeatures= test_tweets_filtered
#print cutoff
#print len(neg_word_features)
#print len(pos_word_features)

#selects 1/100 of the features to be used for training
cutoff = int(math.floor((len(test_neg_tweets_filtered)*1/100 + len(test_pos_tweets_filtered)*1/100))/2)
#trainFeatures = neg_tweets_filtered[:cutoff] + pos_tweets_filtered[:cutoff]
testFeatures = test_neg_tweets_filtered[:cutoff] + test_pos_tweets_filtered[:cutoff]
testing_set = nltk.classify.apply_features(extract_features, testFeatures)


print 'Trained on %d instances, Tested on %d instances' % (len(trainFeatures), len(testFeatures))
print 'NB Accuracy:', nltk.classify.accuracy(nb_classifier, testing_set)* 100, "%"
print 'SVM Accuracy:', nltk.classify.accuracy(svm_classifier, testing_set)* 100, "%"






nb_classifier = nltk.NaiveBayesClassifier.train(training_set)
print 'NB Accuracy:', nltk.classify.accuracy(nb_classifier, testing_set)* 100, "%"

svm_classifier = nltk.classify.SklearnClassifier(LinearSVC()).train(training_set)
print 'SVM Accuracy:', nltk.classify.accuracy(svm_classifier, testing_set)* 100, "%"




