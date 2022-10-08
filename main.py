# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk import ngrams
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.models import Sequential
from keras import layers
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
    # Use a breakpoint in the code line below to debug your script.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\alize\Desktop\webapp\sqle.csv", encoding='utf-16')
    print(df)
    vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english'))
    posts = vectorizer.fit_transform(df['Sentence'].values.astype('U')).toarray()
    transformed_posts = pd.DataFrame(posts)
    df = pd.concat([df, transformed_posts], axis=1)
    X = df[df.columns[2:]]
    y = df['Label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc=accuracy_score(y_test, y_pred)
    print(acc)
    input_dim = X_train.shape[1]  # Number of features

    model = Sequential()
    model.add(layers.Dense(20, input_dim=input_dim, activation='relu'))
    model.add(layers.Dense(10, activation='tanh'))
    model.add(layers.Dense(1024, activation='relu'))

    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    print(model)
    classifier_nn = model.fit(X_train, y_train,
                              epochs=10,
                              verbose=True,
                              validation_data=(X_test, y_test),
                              batch_size=15)
    print(classifier_nn)
    pred = model.predict(X_test)
    for i in range(len(pred)):
        if pred[i] > 0.5:
            pred[i] = 1
        elif pred[i] <= 0.5:
            pred[i] = 0
    accu= accuracy_score(y_test, pred)
    print(accu)


    def accuracy_function(tp, tn, fp, fn):

        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return accuracy


    def precision_function(tp, fp):

        precision = tp / (tp + fp)

        return precision


    def recall_function(tp, fn):

        recall = tp / (tp + fn)

        return recall


    def confusion_matrix(truth, predicted):

        true_positive = 0
        true_negative = 0
        false_positive = 0
        false_negative = 0

        for true, pred in zip(truth, predicted):
            if true == 1:
                if pred == true:
                    true_positive += 1
                elif pred != true:
                    false_negative += 1

            elif true == 0:
                if pred == true:
                    true_negative += 1
                elif pred != true:
                    false_positive += 1

        accuracy = accuracy_function(true_positive, true_negative, false_positive, false_negative)
        precision = precision_function(true_positive, false_positive)
        recall = recall_function(true_positive, false_negative)

        return (accuracy,
                precision,
                recall)


    accuracy, precision, recall = confusion_matrix(y_test, pred)
    print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))
    pre=precision_score(y_test, pred)
    rec=recall_score(y_test, pred)
    print(pre)
    print(rec)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
