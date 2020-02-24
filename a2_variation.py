import argparse
import random
from sklearn.datasets import fetch_20newsgroups
from sklearn.base import is_classifier
import numpy as np
random.seed(42)
import pprint
from sklearn.decomposition import TruncatedSVD
from nltk import corpus
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

stopwords = corpus.stopwords.words("english")

# MY_TOKENIZER PIPLE LINE
# lower -> length control -> stopwrods crontrol -> only alphabetic -> lemmatizing
def my_tokenizer(s):
    wordnet_lemmatizer = WordNetLemmatizer()
    s = s.lower()
    tokens = s.split(" ")
    tokens = [t for t in tokens if len(t) > 2]
    tokens = [t for t in tokens if t not in stopwords]
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens]
    return tokens


###### PART 1
#DONT CHANGE THIS FUNCTION
def part1(samples):
    #extract features
    X = extract_features(samples)
    FILTER = (np.sum(X, axis = 0)>20)
    X = X.T[FILTER].T
    assert type(X) == np.ndarray
    print("Example sample feature vec: ", X[0])
    print("Data shape: ", X.shape)

    return X


def extract_features(samples):
    print("Extracting features ...")
    word_index_map = {}
    current_index = 0
    all_tokens = []
    index_word_map = []
    for text in samples:
        tokens = my_tokenizer(text)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)

    def token_to_vector(tokens):
        x = np.zeros(len(word_index_map))
        for token in tokens:
            index = word_index_map[token]
            x[index] += 1
        return x

    N = len(samples)
    D = len(word_index_map)
    X = np.zeros((N, D))
    i = 0
    for tokens in all_tokens:
        X[i, :] = token_to_vector(tokens)
        i += 1
    return X



##### PART 2
#DONT CHANGE THIS FUNCTION
def part2(X, n_dim):
    #Reduce Dimension
    print("Reducing dimensions ... ")
    X_dr = reduce_dim(X, n=n_dim)
    assert X_dr.shape != X.shape
    assert X_dr.shape[1] == n_dim
    print("Example sample dim. reduced feature vec: ", X[0])
    print("Dim reduced data shape: ", X_dr.shape)
    return X_dr


def reduce_dim(X,n=10):
    svd = PCA(n_components=n)
    X_dr = svd.fit_transform(X)
    return X_dr



##### PART 3
#DONT CHANGE THIS FUNCTION EXCEPT WHERE INSTRUCTED
def get_classifier(clf_id):
    if clf_id == 1:
        clf = GaussianNB() # <--- REPLACE THIS WITH A SKLEARN MODEL
    elif clf_id == 2:
        clf =  DecisionTreeClassifier() # <--- REPLACE THIS WITH A SKLEARN MODEL
    else:
        raise KeyError("No clf with id {}".format(clf_id))

    assert is_classifier(clf)
    print("Getting clf {} ...".format(clf.__class__.__name__))
    return clf

#DONT CHANGE THIS FUNCTION
def part3(X, y, clf_id):
    #PART 3
    X_train, X_test, y_train, y_test = shuffle_split(X,y)

    #get the model
    clf = get_classifier(clf_id)

    #printing some stats
    print()
    print("Train example: ", X_train[0])
    print("Test example: ", X_test[0])
    print("Train label example: ",y_train[0])
    print("Test label example: ",y_test[0])
    print()


    #train model
    print("Training classifier ...")
    train_classifer(clf, X_train, y_train)


    # evalute model
    print("Evaluating classcifier ...")
    evalute_classifier(clf, X_test, y_test)


def shuffle_split(X,y):
    temp = np.zeros((X.shape[0],X.shape[1]+1))
    temp[:,:-1] = X
    temp[:,-1] = y
    np.random.shuffle(temp)
    X = temp[:,:-1]
    y = temp[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return  X_train, X_test, y_train, y_test


def train_classifer(clf, X, y):
    assert is_classifier(clf)
    clf.fit(X, y)


def evalute_classifier(clf, X, y):
    assert is_classifier(clf)
    y_true = y
    y_pred = clf.predict(X)
    print(classification_report(y_true, y_pred, range(0,20),list(load_data()[2])))


######
#DONT CHANGE THIS FUNCTION
def load_data():
    print("------------Loading Data-----------")
    data = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)
    print("Example data sample:\n\n", data.data[0])
    print("Example label id: ", data.target[0])
    print("Example label name: ", data.target_names[data.target[0]])
    print("Number of possible labels: ", len(data.target_names))
    return data.data, data.target, data.target_names


#DONT CHANGE THIS FUNCTION
def main(model_id=None, n_dim=False):

    # load data
    samples, labels, label_names = load_data()


    #PART 1
    print("\n------------PART 1-----------")
    X = part1(samples)

    #part 2
    if n_dim:
        print("\n------------PART 2-----------")
        X = part2(X, n_dim)

    #part 3
    if model_id:
        print("\n------------PART 3-----------")
        part3(X, labels, model_id)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_dim",
                        "--number_dim_reduce",
                        default=False,
                        type=int,
                        required=False,
                        help="int for number of dimension you want to reduce the features for")

    parser.add_argument("-m",
                        "--model_id",
                        default=False,
                        type=int,
                        required=False,
                        help="id of the classifier you want to use")

    args = parser.parse_args()
    main(   
            model_id=int(args.model_id),
            n_dim=args.number_dim_reduce
            )
