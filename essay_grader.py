import pandas as pd
import numpy as np
import enchant
from sklearn import preprocessing, model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords

training_data = pd.read_csv("training_set_rel3.tsv", header=0, delimiter="\t", quoting=3)
d = enchant.Dict("en_US")

spelling_errors = []

def clean_text(text):
    words = text.lower().split()

    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]

    meaningful_words = [unicode(w, errors='replace') for w in meaningful_words]
    misspelled_words = 0

    index = 0
    for w in meaningful_words:
        if not w[0] == '@' and not d.check(w):
            options = d.suggest(w)
            if len(options) > 0:
                meaningful_words[index] = options[0]
            misspelled_words += 1
        index += 1

    return " ".join(meaningful_words), misspelled_words

num_rows = training_data["essay_id"].size
clean_training_data = []
misspelled_words_count = []

print "cleaning data..."
for i in xrange(num_rows):
    clean_data, misspelled_words = clean_text(training_data["essay"][i])
    clean_training_data.append(clean_data)
    misspelled_words_count.append(misspelled_words)


print "making bag of words..."
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
training_data_features = vectorizer.fit_transform(clean_training_data)
training_data_features = training_data_features.toarray()
misspelled_words_count = np.asarray(misspelled_words_count)

vocab = vectorizer.get_feature_names()

X_all = pd.DataFrame(training_data_features, columns=vocab)
X_all = X_all.rename(columns = {'fit': 'fit_feature'})
X_all["misspellling"] = misspelled_words_count

y_all = training_data["domain1_score"]

num_test = 0.20
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X_all, y_all, test_size=num_test, random_state=23)

clf = RandomForestClassifier()
print "training model..."
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print accuracy
