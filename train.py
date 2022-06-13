from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import joblib
import numpy as np

twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
print(twenty_train.target_names)
# # print(twenty_train.target_names) # categories (target classifications)

# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(twenty_train.data)
# # X_train_counts.shape
#
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# # print(X_train_tfidf.shape)
#
# clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
#
# print(clf)


#
#   Vectorize, TF-IDF, then Naive Bayes classifier
#       fit the training data and target classifications
#
text_clf = Pipeline([
    ('vect', CountVectorizer(stop_words='english')),  # adding stop words actually reduced the accuracy by .6%
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])
# train
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


#
#   Specify parameters and run GridSearchCV to find best predictors
#
#
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],  #try n-grams and bi-grams
    'tfidf__use_idf': (True, False),    #why
    'clf__alpha': (1e-2, 1e-3), #why
}
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
gs_clf = gs_clf.fit(twenty_train.data, twenty_train.target)
joblib.dump(gs_clf, 'newsGroupClassifier.pkl')
#
# print(gs_clf.best_score_)
# print(gs_clf.best_params_)


#
#       T E S T
#
#
# twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
# predicted = gs_clf.predict(twenty_test.data) # use the grid-search classifier
# print(np.mean(predicted == twenty_test.target))


#
#   Custom test
#
#
# myTestCase = twenty_test.data[1020]
# print(myTestCase)
# computedCategory = gs_clf.predict([myTestCase])[0]
# print(twenty_train.target_names[computedCategory])
# probabilityByCategory = gs_clf.predict_proba([myTestCase])
# print(probabilityByCategory[0][computedCategory])
# print("{0:.0%}".format(probabilityByCategory[0][computedCategory]))
