from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
import argparse
from sklearn.svm import SVC
import os
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    ii = 0
    ij = 0
    for i in range(0, len(C)):
        ii += C[i][i]
        for j in range(0, len(C[0])):
            ij += C[i][j]
    total = ii/ij
    return total

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    scores = []
    for k in range(0, 4):
        kj = 0
        for j in range(0, len(C)):
            kj += C[k][j]
        score = (C[k][k])/kj if kj > 0 else 0
        scores.append(score)
    return scores

def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    scores = []
    for k in range(0, 4):
        ki = 0
        for i in range(0, len(C)):
            ki += C[i][k]
        score = (C[k][k])/ki if ki > 0 else 0
        scores.append(score)
    return scores

def linear_svc(X_train, X_test, y_train, y_test):
    """
    Returns results of trained classifier

    :param X_train: np array of features
    :type X_train: nump array
    :param y_train: np array of results corrosponding to features
    :type y_train: numpy array
    :return: Accuracy, precision, recall
    :rtype: dict
    """
    l_svc = SVC(kernel="linear", max_iter=1000)
    l_svc.fit(X_train, y_train)
    l_svc_res = l_svc.predict(X_test)
    l_svc_conf = confusion_matrix(y_test, l_svc_res)
    l_svc_acc = accuracy(l_svc_conf)
    l_svc_pre = precision(l_svc_conf)
    l_svc_rec = recall(l_svc_conf)
    print("LINEAR SVC", l_svc_acc, l_svc_pre, l_svc_rec)
    return {"acc": l_svc_acc, "pre": l_svc_pre, "rec": l_svc_rec, "conf": l_svc_conf}


def radial_svc(X_train, X_test, y_train, y_test):
    """
    Returns results of trained classifier

    :param X_train: np array of features
    :type X_train: nump array
    :param y_train: np array of results corrosponding to features
    :type y_train: numpy array
    :return: Accuracy, precision, recall
    :rtype: dict
    """
    r_svc = SVC(kernel="rbf", gamma=2, max_iter=1000)
    r_svc.fit(X_train, y_train)
    r_svc_res = r_svc.predict(X_test)
    r_svc_conf = confusion_matrix(y_test, r_svc_res)
    r_svc_acc = accuracy(r_svc_conf)
    r_svc_pre = precision(r_svc_conf)
    r_svc_rec = recall(r_svc_conf)
    print("RADIAL SVC", r_svc_acc, r_svc_pre, r_svc_rec)
    return {"acc": r_svc_acc, "pre": r_svc_pre, "rec": r_svc_rec, "conf": r_svc_conf}


def ran_forest(X_train, X_test, y_train, y_test):
    """
    Returns results of trained classifier

    :param X_train: np array of features
    :type X_train: nump array
    :param y_train: np array of results corrosponding to features
    :type y_train: numpy array
    :return: Accuracy, precision, recall
    :rtype: dict
    """
    rf = RandomForestClassifier(max_depth=5, n_estimators=10)
    rf.fit(X_train, y_train)
    rf_res = rf.predict(X_test)
    rf_conf = confusion_matrix(y_test, rf_res)
    rf_acc = accuracy(rf_conf)
    rf_pre = precision(rf_conf)
    rf_rec = recall(rf_conf)
    print("RANDOM FOREST", rf_acc, rf_pre, rf_rec)
    return {"acc": rf_acc, "pre": rf_pre, "rec": rf_rec, "conf": rf_conf}


def MLP(X_train, X_test, y_train, y_test):
    """
    Returns results of trained classifier

    :param X_train: np array of features
    :type X_train: nump array
    :param y_train: np array of results corrosponding to features
    :type y_train: numpy array
    :return: Accuracy, precision, recall
    :rtype: dict
    """
    mlp = MLPClassifier(alpha=0.05)
    mlp.fit(X_train, y_train)
    mlp_res = mlp.predict(X_test)
    mlp_conf = confusion_matrix(y_test, mlp_res)
    mlp_acc = accuracy(mlp_conf)
    mlp_pre = precision(mlp_conf)
    mlp_rec = recall(mlp_conf)
    print("MLP", mlp_acc, mlp_pre, mlp_rec)
    return {"acc": mlp_acc, "pre": mlp_pre, "rec": mlp_rec, "conf": mlp_conf}


def ADA(X_train, X_test, y_train, y_test):
    """
    Returns results of trained classifier

    :param X_train: np array of features
    :type X_train: nump array
    :param y_train: np array of results corrosponding to features
    :type y_train: numpy array
    :return: Accuracy, precision, recall
    :rtype: dict
    """
    ada = AdaBoostClassifier()
    ada.fit(X_train, y_train)
    ada_res = ada.predict(X_test)
    ada_conf = confusion_matrix(y_test, ada_res)
    ada_acc = accuracy(ada_conf)
    ada_pre = precision(ada_conf)
    ada_rec = recall(ada_conf)
    print("ADA BOOST", ada_acc, ada_pre, ada_rec)
    return {"acc": ada_acc, "pre": ada_pre, "rec": ada_rec, "conf": ada_conf}


def class31(filename):
    ''' This function performs experiment 3.1

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    subdir = os.path.abspath(filename)
    feats = np.load(subdir, "r")
    feats_np = feats[feats.files[0]]
    print("# of data:", len(feats_np))
    x = np.zeros([len(feats_np), 173])
    y = np.zeros(len(feats_np))
    index = 0
    for row in feats_np:
        x[index] = row[:-1]
        y[index] = row[-1]
        index += 1

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, train_size=0.8)

    results = {"l_svc": linear_svc(X_train, X_test, y_train, y_test),
                "r_svc": radial_svc(X_train, X_test, y_train, y_test),
                 "rf": ran_forest(X_train, X_test, y_train, y_test),
                 "mlp": MLP(X_train, X_test, y_train, y_test),
                "ada": ADA(X_train, X_test, y_train, y_test)}

    classifier_index = {"l_svc": 1, "r_svc": 2, "rf": 3, "mlp": 4, "ada": 5}
    best_clf = None
    best_acc = -1
    # find best result
    for key in results:
        if results[key]["acc"] > best_acc:
            best_acc = results[key]["acc"]
            best_clf = key

    iBest = classifier_index[best_clf]
    print("BESt classifier", iBest)
    csv = open("a1_3.1.csv", "w+")
    for key in results:
        conf = results[key]["conf"]
        conf = conf.flatten()
        index = classifier_index[key]
        csv.write(str(index))
        csv.write("," + str(results[key]["acc"]))
        for rec in results[key]["rec"]:
            csv.write("," + str(rec))
        for pre in results[key]["pre"]:
            csv.write("," + str(pre))
        for c in conf:
            csv.write("," + str(c))
        csv.write("\n")

    return (X_train, X_test, y_train, y_test, iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    clf_index = {1: SVC(kernel="linear", max_iter=1000),2: SVC(kernel="rbf", gamma=2, max_iter=1000),
           3: RandomForestClassifier(max_depth=5, n_estimators=10), 4: MLPClassifier(alpha=0.05),
           5: AdaBoostClassifier()}

    clf = clf_index[iBest]
    X_1k = X_train[:1000]
    y_1k = y_train[:1000]
    train_1k = (X_1k, y_1k)
    X_5k = X_train[:5000]
    y_5k = y_train[:5000]
    train_5k = (X_5k, y_5k)
    X_10k = X_train[:10000]
    y_10k = y_train[:10000]
    train_10k = (X_10k, y_10k)
    X_15k = X_train[:15000]
    y_15k = y_train[:15000]
    train_15k = (X_15k, y_15k)
    X_20k = X_train[:20000]
    y_20k = y_train[:20000]
    train_20k = (X_20k, y_20k)
    csv = open("a1_3.2.csv", "w+")
    clf.fit(train_1k[0], train_1k[1])
    result = clf.predict(X_test)
    conf = confusion_matrix(y_test, result)
    csv.write(str(accuracy(conf)))

    clf.fit(train_5k[0], train_5k[1])
    result = clf.predict(X_test)
    conf = confusion_matrix(y_test, result)
    csv.write("," + str(accuracy(conf)))

    clf.fit(train_10k[0], train_10k[1])
    result = clf.predict(X_test)
    conf = confusion_matrix(y_test, result)
    csv.write("," + str(accuracy(conf)))

    clf.fit(train_15k[0], train_15k[1])
    result = clf.predict(X_test)
    conf = confusion_matrix(y_test, result)
    csv.write("," + str(accuracy(conf)))

    clf.fit(train_20k[0], train_20k[1])
    result = clf.predict(X_test)
    conf = confusion_matrix(y_test, result)
    csv.write("," + str(accuracy(conf)))
    csv.write("\n")
    csv.write("The expected behaviour is that an increase in training set leads to an increase in accuracy, "
              "as then the classifier will have more data to learn from. We can "
              "see here that as "
              "the number of training sets increase, the accuracies have slightly increased and "
              "oscillates around 39%. This may be due to limitations in the features extracted and that the "
              "features may not directly correlate to the resulting party.")

    return (X_1k, y_1k)

def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3

    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    clf_index = {1: SVC(kernel="linear", max_iter=1000), 2: SVC(kernel="rbf", gamma=2, max_iter=1000),
                 3: RandomForestClassifier(max_depth=5, n_estimators=10), 4: MLPClassifier(alpha=0.05),
                 5: AdaBoostClassifier()}

    clf = clf_index[i]
    csv = open("a1_3.3.csv", "w+")
    count = 0
    b1_feat = []
    best_feat = []
    for data in [(X_1k, y_1k), (X_train, y_train)]:
        for k in [5, 10, 20, 30, 40, 50]:
            selector = SelectKBest(f_classif, k)
            selector.fit_transform(data[0], data[1])
            pp = selector.pvalues_
            indexes = selector.get_support()
            best = pp[indexes]

            # top features of 1k training set
            if count == 0:
                if k == 5:
                    print("len", len(indexes.tolist()))
                    print("indexes", indexes.tolist())
                    indexes = indexes.tolist()
                    for index in range(0, len(indexes)):
                        if indexes[index] is True:
                            b1_feat.append(index)

            # top features for 32k training set
            elif count == 1:
                if k == 5:
                    print("len", len(indexes.tolist()))
                    print("indexes", indexes.tolist())
                    indexes = indexes.tolist()
                    for index in range(0, len(indexes)):
                        if indexes[index] is True:
                            best_feat.append(index)
                csv.write(str(k))
                for p in best:
                    csv.write("," + str(p))
                csv.write("\n")
        count += 1
    print("best 5 features", best_feat, b1_feat)
    X_1k_best = np.zeros((1000, 5))
    X_test_best = np.zeros((8000, 5))
    X_train_best = np.zeros((32000, 5))
    for j in range(5):
        for i in range(0, len(X_test)):
            X_test_best[i][j] = X_test[i][best_feat[j]]
        for i in range(0, len(X_train)):
            X_train_best[i][j] = X_train[i][best_feat[j]]
        for i in range(0, len(X_1k)):
            X_1k_best[i][j] = X_1k[i][best_feat[j]]

    clf.fit(X_1k_best, y_1k)
    result = clf.predict(X_test_best)
    print("result len", len(result))
    csv.write(str(accuracy(confusion_matrix(y_test, result))) + ",")
    clf.fit(X_test_best, y_test)
    result = clf.predict(X_test_best)
    csv.write(str(accuracy(confusion_matrix(y_test, result))) + "\n")

    csv.write("liwc_sexual, receptiviti_cautious, receptiviti_type_a are the common best features in both low and high"
              "amounts of data. We can see that the cautious feature may be a good indicator since people in "
              "different political groups may be more wary of some topics, hence are more cautious. Or it can "
              "possibly be an indicator that conspiracy theorists correlate to certain parties.\n")
    csv.write("P values are generally higher given more data. This may be because there is less bias a set of data"
              "can have towards particular features.\n")

    csv.write("liwc_sexual, receptiviti_cautious,receptiviti_type_a, number of commas, number of common nouns are the top 5"
              "features for the 32K training case. This seems to suggest that different parties tends to have "
              "different speech habits since the features are so diverse. This makes sense since different parties "
              "would attract a specific type of demographic, as such they may be more prone to use a similar tone and"
              "sentence structure.")


def class34( filename, i ):
    ''' This function performs experiment 3.4

    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)
        '''
    subdir = os.path.abspath(filename)
    feats = np.load(subdir, "r")
    feats_np = feats[feats.files[0]]
    x = np.zeros([len(feats_np), 173])
    y = np.zeros(len(feats_np))
    index = 0
    for row in feats_np:
        x[index] = row[:-1]
        y[index] = row[-1]
        index += 1

    kf = KFold(n_splits=5, shuffle=True)
    csv = open("a1_3.4.csv", "w+")
    l_svc_acc = []
    r_svc_acc = []
    rf_acc = []
    mlp_acc = []
    ada_acc = []
    # splits training and testing set
    for train, test in kf.split(x):
        X_train, X_test, y_train, y_test = x[train], x[test], y[train], y[test]
        l_svc = linear_svc(X_train, X_test, y_train, y_test)["acc"]
        r_svc = radial_svc(X_train, X_test, y_train, y_test)["acc"]
        rf = ran_forest(X_train, X_test, y_train, y_test)["acc"]
        mlp = MLP(X_train, X_test, y_train, y_test)["acc"]
        ada = ADA(X_train, X_test, y_train, y_test)["acc"]
        csv.write(str(l_svc) + "," + str(r_svc) + "," + str(r_svc) + "," + str(rf) + "," + str(mlp) + "," + str(ada)
                  + "\n")
        l_svc_acc.append(l_svc)
        r_svc_acc.append(r_svc)
        rf_acc.append(rf)
        mlp_acc.append(mlp)
        ada_acc.append(ada)

    all_acc = [l_svc_acc, r_svc_acc, rf_acc, mlp_acc, ada_acc]
    best_acc = all_acc[i-1]
    all_acc.pop(i-1)
    count = 0
    # calculate p-values
    for clf_acc in all_acc:
        p = stats.ttest_rel(best_acc, clf_acc)[1]
        string_p = str(p) + "," if count < 3 else str(p)
        csv.write(string_p)
        count += 1
    csv.write("\n")
    csv.write("We can see that our best classifier, AdaBoost, MLP performs better it however it is "
              "much better than Linear SVC and Radial SVC and Random Forest. MLP has a low p-value of 0.2%, "
              "thus the null hypothesis is probably false and MLP statistically performs better than our best "
              "classifier.."
              "This may be "
              "because AdaBoost readjusts weighting on labels similar "
              "to how MLP layers calculates a feature's bias.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()
    filename = args.input
    X_train, X_test, y_train, y_test, iBest = class31(filename)
    X_1k, y_1k = class32(X_train, X_test, y_train, y_test, iBest)
    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)
    class34(filename, iBest)
