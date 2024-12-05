import random
from math import sqrt, exp

import numpy as np
import pandas
from sklearn import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, \
    cohen_kappa_score, confusion_matrix, auc
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def entropy(y):
    proportions = np.bincount(y) / len(y)
    entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
    return entropy


def create_split(X, thresh):
    left_idx = np.argwhere(X <= thresh).flatten()
    right_idx = np.argwhere(X > thresh).flatten()
    return left_idx, right_idx


def information_gain(X, y, thresh):
    parent_loss = entropy(y)
    # print(parent_loss)
    left_idx, right_idx = create_split(X, thresh)
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)
    if n_left == 0 or n_right == 0:
        return 0

    child_loss = (n_left / n) * entropy(y[left_idx]) + (n_right / n) * entropy(y[right_idx])
    return parent_loss - child_loss


def gain_ratio(X, y, thresh):
    parent_loss = entropy(y)
    # print(parent_loss)
    left_idx, right_idx = create_split(X, thresh)
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)
    if n_left == 0 or n_right == 0:
        return 0

    child_loss = (n_left / n) * entropy(y[left_idx]) + (n_right / n) * entropy(y[right_idx])
    return (parent_loss - child_loss) / parent_loss


def gini_impurity(x):
    class_labels = list(np.unique(x))
    freq_labels = dict()
    for c in class_labels:
        freq_labels[c] = 0
    n = 0
    for d in x:
        freq_labels[d] += 1
        n += 1
    p = 0
    for k in freq_labels:
        p += np.square(freq_labels[k] / n)
    return 1 - p


def gini(X, y, thresh):
    left_idx, right_idx = create_split(X, thresh)
    n, n_left, n_right = len(y), len(left_idx), len(right_idx)
    gini_index = (n_left / n) * gini_impurity(y[left_idx]) + (n_right / n) * gini_impurity(y[right_idx])
    return gini_index


def weighted_euclidean_distance(d1, d2, weights):
    ans = 0
    for i in range(len(d1)):
        ans += weights[i] * (np.power(d2[i] - d1[i], 2))
    return np.sqrt(ans)


def knn(training, results, test, actual, weights, k=5):
    dist = np.zeros(len(training))
    accurate = 0
    preds=[]
    # print('PREDICTED\tACTUAL\tCORRECT?')
    for i in range(len(test)):
        for j in range(len(training)):
            dist[j] = weighted_euclidean_distance(test[i], training[j], weights)
        ind = np.argsort(dist)[:k]
        # print(dist)
        # print(ind)
        values = [results[x] for x in ind]
        distances = [dist[x] for x in ind]
        freq = dict()
        mindist = dict()
        for no, val in enumerate(values):
            if val in freq:
                freq[val] += 1
                if mindist[val] > distances[no]:
                    mindist[val] = distances[no]
            else:
                freq[val] = 1
                mindist[val] = distances[no]
        max_count = max(freq.values())
        prediction = ''
        for val, count in freq.items():
            if count == max_count:
                if prediction == '':
                    prediction = val
                elif mindist[val] < mindist[prediction]:
                    prediction = val
                elif mindist[val] == mindist[prediction]:
                    prediction = random.choice([val, prediction])
        preds.append(prediction)
        # print(prediction, '\t', actual[i],'\t',prediction == actual[i])
        # accurate += (prediction == actual[i])

    # print(accurate, ' correctly predicted out of ', len(test), ' test samples')
    # print('Accuracy = ', (100 * accurate / len(test)), '%')
    return preds


def calculate_from_confusion_matrix(y_test, y_pred):
    cm=confusion_matrix(y_test,y_pred)
    cc,ic=0,0
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            if i==j:
                cc+=cm[i][j]
            else:
                ic+=cm[i][j]
    return cc,ic


if __name__ == '__main__':
    file=open('results_proj.txt','r')
    choices_k = [1, 3, 5, 7, 9, 11, 13, 15]
    sps = [0.5,0.4,0.25]
    # choices_k = [15]
    # data = datasets.load_iris()
    data = datasets.load_breast_cancer()
    # data = pandas.read_csv('IRIS.csv')
    file.write('IRIS DATASET\n')

    X, y = data.data, data.target
    '''X, y = [], []
    for i in range(len(data)):
        row = []
        result = ''
        for j, col in enumerate(data):
            if j < (data.shape[1] - 1):
                row.append(float(data[col][i]))
            else:
                result = data[col][i]
        X.append(row)
        y.append(result)'''
    # print(X, y)
    for sp in sps:
        file.write('TRAIN-TEST SPLIT : ' + str((1 - sp) * 100) + '-' + str(sp * 100) + '\n')
        accK = []
        precK = []
        recK = []
        f1K = []
        aucK = []
        mccK = []
        cohenkappaK = []
        ccK = []
        icK = []
        '''fnK = []
        tnK = []
        tnrK = []'''
        for K in choices_k:
            file.write('K = '+str(K)+'\n')
            avgacc = [0.0 for _ in range(19)]
            stddevacc = [0.0 for _ in range(19)]
            avgprec = [0.0 for _ in range(19)]
            stddevprec = [0.0 for _ in range(19)]
            avgrec = [0.0 for _ in range(19)]
            stddevrec = [0.0 for _ in range(19)]
            avgf1 = [0.0 for _ in range(19)]
            stddevf1 = [0.0 for _ in range(19)]
            avgmcc = [0.0 for _ in range(19)]
            stddevmcc = [0.0 for _ in range(19)]
            avgauc = [0.0 for _ in range(19)]
            stddevauc = [0.0 for _ in range(19)]
            avgck = [0.0 for _ in range(19)]
            stddevck = [0.0 for _ in range(19)]
            avgcc = [0.0 for _ in range(19)]
            stddevcc = [0.0 for _ in range(19)]
            avgic = [0.0 for _ in range(19)]
            stddevic = [0.0 for _ in range(19)]
            '''avgtn = [0.0 for _ in range(19)]
            stddevtn = [0.0 for _ in range(19)]
            avgfn = [0.0 for _ in range(19)]
            stddevfn = [0.0 for _ in range(19)]
            avgtnr = [0.0 for _ in range(19)]
            stddevtnr = [0.0 for _ in range(19)]'''
            l=0
            for z in range(10):
                X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), test_size=sp)
                n_samples, n_features = X_train.shape
                # n_class_labels = len(np.unique(y_train))
                features = np.random.choice(n_features, n_features, replace=False)
                weights = dict()
                gain_ratios = dict()
                for feat in features:
                    X_feat = X_train[:, feat]
                    thresholds = np.unique(X_feat)
                    split = {'score': - 1, 'feat': None, 'thresh': None}
                    for thresh in thresholds:
                        score = information_gain(X_feat, y_train, thresh)
                        if score > split['score']:
                            split['score'] = score
                            split['feat'] = feat
                            split['thresh'] = thresh
                    # print(split['feat'],": ",split['score'])
                    weights[split['feat']] = split['score']  # case 1: just using information gain values as weights
                min_max_scaled_weights = dict()
                weights_divided_by_min = dict()
                weights_divided_by_min_timesk = dict()
                baobaominru=dict()
                minm, maxm = 10, 0
                for k in weights.keys():
                    if weights[k] < minm:
                        minm = weights[k]
                    if weights[k] > maxm:
                        maxm = weights[k]
                for k in weights.keys():
                    min_max_scaled_weights[k] = (weights[k] - minm) / (maxm - minm)
                    weights_divided_by_min[k] = round(weights[k] / minm)
                    weights_divided_by_min_timesk[k] = round(k * weights[k] / minm)
                    baobaominru[k]=weights[k]*exp(weights[k])/sum(weights.values())
                    # print(k,': ',weights_divided_by_min[k])
                gini_indexes = dict()
                gini_recs = dict()
                for feat in features:
                    X_feat = X_train[:, feat]
                    thresholds = np.unique(X_feat)
                    split = {'score': - 1, 'feat': None, 'thresh': None}
                    gini_split = {'score': 2, 'feat': None, 'thresh': None}
                    for thresh in thresholds:
                        score = gain_ratio(X_feat, y_train, thresh)
                        gini_score = gini(X_feat, y_train, thresh)
                        if score > split['score']:
                            split['score'] = score
                            split['feat'] = feat
                            split['thresh'] = thresh
                        if gini_score < gini_split['score']:
                            gini_split['score'] = gini_score
                            gini_split['feat'] = feat
                            gini_split['thresh'] = thresh
                    # print(split['feat'],": ",split['score'],", ",gini_split['score'])
                    gain_ratios[split['feat']] = split['score']  # just using gain ratio values as weights
                    gini_indexes[split['feat']] = 1 - gini_split['score']  # using 1-gini as weights
                    gini_recs[split['feat']] = 1 / gini_split['score']  # using 1/gini as weights
                min_max_scaled_ginis = dict()
                ginis_divided_by_min = dict()
                minm, maxm = 2, 0
                for k in gini_indexes.keys():
                    if gini_indexes[k] < minm:
                        minm = gini_indexes[k]
                    if gini_indexes[k] > maxm:
                        maxm = gini_indexes[k]
                for k in gini_indexes.keys():
                    min_max_scaled_ginis[k] = (gini_indexes[k] - minm) / (maxm - minm)
                    ginis_divided_by_min[k] = round(gini_indexes[k] / minm)

                min_max_scaled_ginirecs = dict()
                ginirecs_divided_by_min = dict()
                ginirecs_divided_by_min_timesk = dict()
                minm, maxm = -1, 0
                for k in gini_recs.keys():
                    if gini_recs[k] < minm or minm == -1:
                       minm = gini_recs[k]
                    if gini_recs[k] > maxm:
                       maxm = gini_recs[k]
                for key in gini_recs.keys():
                    min_max_scaled_ginirecs[key] = (gini_recs[key] - minm) / (maxm - minm)
                    ginirecs_divided_by_min[key] = round(gini_recs[key] / minm)
                    ginirecs_divided_by_min_timesk[key] = round(K * gini_recs[key] / minm)

                min_max_scaled_grs = dict()
                grs_divided_by_min = dict()
                minm, maxm = 10, 0
                for k in gain_ratios.keys():
                    if gain_ratios[k] < minm:
                        minm = gain_ratios[k]
                    if gain_ratios[k] > maxm:
                        maxm = gain_ratios[k]
                for k in gain_ratios.keys():
                    min_max_scaled_grs[k] = (gain_ratios[k] - minm) / (maxm - minm)
                    grs_divided_by_min[k] = round(gain_ratios[k] / minm)

                preds=[]
                # print('Results from own algorithm (only information gain as it is):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights, K))
                # print('Results from own algorithm (min-max scaled information gain):')
                preds.append(knn(X_train, y_train, X_test, y_test, min_max_scaled_weights, K))
                # print('Results from own algorithm (information gain divided by min ig):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_divided_by_min, K))
                # print('Results from algorithm (baobao minru):')
                preds.append(knn(X_train, y_train, X_test, y_test, baobaominru, K))
                '''print('Results from own algorithm (information gain divided by min ig times k):') # no use
                acc18 = knn(X_train, y_train, X_test, y_test, weights_divided_by_min_timesk, k)'''
                '''print('Results from own algorithm (gain ratio):')  # giving same as ig
                acc4 = knn(X_train, y_train, X_test, y_test, gain_ratios, k)
                print('Results from own algorithm (min-max scaled gain ratio):')
                acc5 = knn(X_train, y_train, X_test, y_test, min_max_scaled_grs, k)'''
                # print('Results from own algorithm (1-gini index):')  # gives less accuracy
                preds.append(knn(X_train, y_train, X_test, y_test, gini_indexes, K))
                # print('Results from own algorithm (min-max scaled 1-gini index):')
                preds.append(knn(X_train, y_train, X_test, y_test, min_max_scaled_ginis, K))
                # print('Results from own algorithm (1-gini index divided by min):')
                preds.append(knn(X_train, y_train, X_test, y_test, ginis_divided_by_min, K))
                # print('Results from own algorithm (1/gini index):')
                preds.append(knn(X_train, y_train, X_test, y_test, gini_recs, K))
                # print('Results from own algorithm (min-max scaled 1/gini index):')
                preds.append(knn(X_train, y_train, X_test, y_test, min_max_scaled_ginirecs, K))
                # print('Results from own algorithm (1/gini index divided by min):')
                preds.append(knn(X_train, y_train, X_test, y_test, ginirecs_divided_by_min, K))
                '''print('Results from own algorithm (1/gini index divided by min times k):') # no use
                acc12 = knn(X_train, y_train, X_test, y_test, ginirecs_divided_by_min_timesk, k)'''
                '''print('Results from own algorithm (gain ratio divided by min gr):') # no use same as weights_divided_by_min
                acc6 = knn(X_train, y_train, X_test, y_test, grs_divided_by_min, k)'''
                # combining two of them to get some new weight
                weights_ig_neggini = dict()
                weights_ig_neggini_minmax = dict()
                weights_ig_neggini_divbymin = dict()
                weights_ig_recgini = dict()
                weights_ig_recgini_minmax = dict()
                weights_ig_recgini_divbymin = dict()
                minm, maxm = -1, -1
                for key in weights.keys():
                    weights_ig_neggini[key] = weights[key] * gini_indexes[key]  # custom weight 1
                    weights_ig_neggini_minmax[key] = min_max_scaled_weights[key] * min_max_scaled_ginis[key]  # custom weight 2
                    weights_ig_neggini_divbymin[key] = weights_divided_by_min[key] * ginis_divided_by_min[key]  # custom weight 3
                    weights_ig_recgini[key] = weights[key] * gini_recs[key]  # custom weight 4
                    if weights_ig_recgini[key] < minm or minm == -1:
                        minm = weights_ig_recgini[key]
                    if weights_ig_recgini[key] > maxm or maxm == -1:
                        maxm = weights_ig_recgini[key]
                    weights_ig_recgini_minmax[key] = min_max_scaled_weights[key] * min_max_scaled_ginirecs[key]  # custom weight 5
                    weights_ig_recgini_divbymin[key] = weights_divided_by_min[key] * ginirecs_divided_by_min[key]  # custom weight 6
                min_max_scaled_customwt4 = dict()
                customwt4_divbymin = dict()
                for key in weights.keys():
                    min_max_scaled_customwt4[key] = (weights_ig_recgini[key] - minm) / (maxm - minm)  # custom weight 7
                    customwt4_divbymin[key] = round(weights_ig_recgini[key] / minm)  # custom weight 8
                #print('Results from own algorithm (custom weight 1):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_ig_neggini, K))
                # print('Results from own algorithm (custom weight 2):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_ig_neggini_minmax, K))
                # print('Results from own algorithm (custom weight 3):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_ig_neggini_divbymin, K))
                #print('Results from own algorithm (custom weight 4):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_ig_recgini, K))
                #print('Results from own algorithm (custom weight 5):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_ig_recgini_minmax, K))
                #print('Results from own algorithm (custom weight 6):')
                preds.append(knn(X_train, y_train, X_test, y_test, weights_ig_recgini_divbymin, K))
                #print('Results from own algorithm (custom weight 7):')
                preds.append(knn(X_train, y_train, X_test, y_test, min_max_scaled_customwt4, K))
                #print('Results from own algorithm (custom weight 8):')
                preds.append(knn(X_train, y_train, X_test, y_test, customwt4_divbymin, K))
                #print('Results from scikit-learn knn algorithm:')
                knnsc = KNeighborsClassifier(n_neighbors=K)
                knnsc.fit(X_train, y_train)
                preds.append(knnsc.predict(X_test))

                for i in range(len(preds)):
                    accsc=(100*accuracy_score(y_test,preds[i]))
                    precsc=(100*precision_score(y_test,preds[i],average='macro'))
                    recsc=(100*recall_score(y_test,preds[i],average='macro'))
                    f1sc=(100*f1_score(y_test,preds[i],average='macro'))
                    mccsc=(matthews_corrcoef(y_test,preds[i]))
                    auprcsc=auc(recsc,precsc)
                    # aucsc=(100*roc_auc_score(y_test,preds[i],average='macro')) # eta ekhono .. ache
                    cksc=cohen_kappa_score(y_test,preds[i])
                    cc,ic = calculate_from_confusion_matrix(y_test,preds[i])
                    if z==0:
                        avgacc[i]=accsc
                        avgprec[i]=precsc
                        avgrec[i]=recsc
                        avgf1[i]=f1sc
                        avgmcc[i]=mccsc
                        avgck[i]=cksc
                        avgcc[i]=cc
                        avgic[i] = ic
                        '''avgtn[i] = tn
                        avgfn[i] = fn
                        avgtnr[i] = tnr'''
                    else:
                        accdelta=accsc-avgacc[i]
                        avgacc[i]=((z-1)*avgacc[i]+accsc)/z
                        stddevacc[i]+=accdelta*(accsc-avgacc[i])
                        precdelta = precsc - avgprec[i]
                        avgprec[i] = ((z - 1) * avgprec[i] + precsc) / z
                        stddevprec[i] += precdelta * (precsc - avgprec[i])
                        recdelta = recsc - avgrec[i]
                        avgrec[i] = ((z - 1) * avgrec[i] + recsc) / z
                        stddevrec[i] += recdelta * (recsc - avgrec[i])
                        f1delta = f1sc - avgf1[i]
                        avgf1[i] = ((z - 1) * avgf1[i] + f1sc) / z
                        stddevf1[i] += f1delta * (f1sc - avgf1[i])
                        mccdelta = mccsc - avgmcc[i]
                        avgmcc[i] = ((z - 1) * avgmcc[i] + mccsc) / z
                        stddevmcc[i] += mccdelta * (mccsc - avgmcc[i])
                        ckdelta = cksc - avgck[i]
                        avgck[i] = ((z - 1) * avgck[i] + cksc) / z
                        stddevck[i] += ckdelta * (cksc - avgck[i])
                        ccdelta = cc - avgcc[i]
                        avgcc[i] = ((z - 1) * avgcc[i] + cc) / z
                        stddevcc[i] += ccdelta * (cc - avgcc[i])
                        '''tndelta = tn - avgtn[i]
                        avgtn[i] = ((z - 1) * avgtn[i] + tn) / z
                        stddevtn[i] += tndelta * (tn - avgtn[i])'''
                        icdelta = ic - avgic[i]
                        avgic[i] = ((z - 1) * avgic[i] + ic) / z
                        stddevic[i] += icdelta * (ic - avgic[i])
                        '''fndelta = fn - avgfn[i]
                        avgfn[i] = ((z - 1) * avgfn[i] + fn) / z
                        stddevfn[i] += fndelta * (fn - avgfn[i])
                        tnrdelta = tnr - avgtnr[i]
                        avgtnr[i] = ((z - 1) * avgtnr[i] + tnr) / z
                        stddevtnr[i] += tnrdelta * (tnr - avgtnr[i])'''

            stddevacc=[sqrt(val/10) for val in stddevacc]
            stddevprec = [sqrt(val / 10) for val in stddevprec]
            stddevrec = [sqrt(val / 10) for val in stddevrec]
            stddevf1 = [sqrt(val / 10) for val in stddevf1]
            stddevmcc = [sqrt(val / 10) for val in stddevmcc]
            stddevck=[sqrt(val/10) for val in stddevck]
            stddevcc = [sqrt(val / 10) for val in stddevcc]
            stddevic = [sqrt(val / 10) for val in stddevic]
            '''stddevtn = [sqrt(val / 10) for val in stddevtn]
            stddevfn = [sqrt(val / 10) for val in stddevfn]
            stddevtnr = [sqrt(val / 10) for val in stddevtnr]'''

            for ss in range(len(avgacc)):
                accK.append((avgacc[ss],stddevacc[ss]))
                precK.append((avgprec[ss], stddevprec[ss]))
                recK.append((avgrec[ss], stddevrec[ss]))
                f1K.append((avgf1[ss],stddevf1[ss]))
                mccK.append((avgmcc[ss], stddevmcc[ss]))
                cohenkappaK.append((avgck[ss], stddevck[ss]))
                ccK.append((avgcc[ss], stddevcc[ss]))
                # tnK.append((avgtn[ss], stddevtn[ss]))
                icK.append((avgic[ss], stddevic[ss]))
                '''fnK.append((avgfn[ss], stddevfn[ss]))
                tnrK.append((avgtnr[ss], stddevtnr[ss]))'''

            file.write('Accuracy\n')
            file.write(str(accK)+'\n')
            file.write('Precision\n')
            file.write(str(precK)+'\n')
            file.write('Recall\n')
            file.write(str(recK) + '\n')
            file.write('F1 Score\n')
            file.write(str(f1K) + '\n')
            file.write('MCC\n')
            file.write(str(mccK) + '\n')
            file.write("Cohen's Kappa\n")
            file.write(str(cohenkappaK) + '\n')
            file.write('Correctly Classified Instances\n')
            file.write(str(ccK) + '\n')
            file.write('Incorrectly Classified Instances\n')
            file.write(str(icK) + '\n')
            '''file.write('TN\n')
            file.write(str(tnK) + '\n')
            file.write('FN\n')
            file.write(str(fnK) + '\n')
            file.write('True Negative Rate\n')
            file.write(str(tnrK) + '\n')'''
    file.close()
