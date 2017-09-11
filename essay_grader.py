import pandas as pd
import numpy as np
import enchant
from sklearn import preprocessing, model_selection
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from nltk.corpus import stopwords
from nltk.stem import *

training_data = pd.read_csv("training_set_rel3.tsv", header=0, delimiter="\t", quoting=3)
d = enchant.Dict("en_US")

spelling_errors = []

def clean_text(text):
    # Make everything lowercase
    words = text.lower().split()

    # Get length of essay
    text_length = len(words)

    # Remove stopwords
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    meaningful_words = [unicode(w, errors='replace') for w in meaningful_words]

    # Count misspelled words
    misspelled_words = 0
    for w in meaningful_words:
        if not d.check(w) and w[0] != '@':
            misspelled_words += 1

    return " ".join(meaningful_words), misspelled_words, text_length

num_rows = training_data["essay_id"].size
clean_training_data = []
misspelled_words_count = []
text_lengths = []

print "cleaning data..."
for i in xrange(num_rows):
    clean_data, misspelled_words, text_length = clean_text(training_data["essay"][i])
    clean_training_data.append(clean_data)
    misspelled_words_count.append(misspelled_words)
    text_lengths.append(text_length)

print "making bag of words..."
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
training_data_features = vectorizer.fit_transform(clean_training_data)
training_data_features = training_data_features.toarray()
misspelled_words_count = np.asarray(misspelled_words_count)
text_lengths = np.asarray(text_lengths)

vocab = vectorizer.get_feature_names()

X_all = pd.DataFrame(training_data_features, columns=vocab)
X_all = X_all.rename(columns = {'fit': 'fit_feature'})
X_all["misspellling"] = misspelled_words_count
X_all["text_length"] = text_lengths

y_all = training_data["domain1_score"]

num_test = 0.25
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_all, y_all, test_size=num_test, random_state=23)

clf = RandomForestClassifier()
print "training model..."
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print accuracy

print "tf-idf transformation..."
X_all = TfidfTransformer().fit_transform(X_all)

print "singular value decomposition..."
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, preprocessing.Normalizer(copy=False))

# Run SVD on the training data
X_all = lsa.fit_transform(X_all)
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_all, y_all, test_size=num_test, random_state=23)

clf = RandomForestClassifier()
print "training model..."
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print accuracy
