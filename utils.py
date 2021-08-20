import numpy as np
from scipy.stats import chi2_contingency
from sklearn.preprocessing import MinMaxScaler

def create_table(X, Y, features):
    values = [None]*len(features)
    counts = [None]*len(features)
    class_values = np.sort(Y.unique())

    for ind, feature in enumerate(features):
        values[ind] = X[feature].unique()
        counts[ind] = np.zeros((len(values[ind]), len(class_values)))

        for i, val in enumerate(values[ind]):
            for j, label in enumerate(class_values):
                counts[ind][i][j] = sum((X[feature] == val) & (Y == label))
        counts[ind] = counts[ind].astype(int)

    return values, counts

def print_counts(features, counts):
    for i in range(len(features)):
        print(features[i])
        print(counts[i])

def create_data(features, counts):
    data = []

    for ind, feature in enumerate(features):
        shape = counts[ind].shape
        cnt = 1
        temp = np.array([], dtype=int)
        for i in range(shape[0]):
            for j in range(shape[1]):
                temp = np.concatenate((temp, np.full(counts[ind][i,j], cnt)))
                cnt = cnt + 1
        data.append(temp)
    return data
    
def fix_zeros(counts, value):
    for i in range(len(counts)):
        idx = np.where(counts[i] == 0)
        idx = list(zip(*idx))
        for j in range(len(idx)):
            counts[i][idx[j]] = value

def kendall_tau_distance(values1, values2):
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), 
                                np.logical_and(a[i] > a[j], b[i] < b[j])).sum()

    return ndisordered / (n * (n - 1))

def calc_kendall_tau(data1, data2):
    order1 = np.flip(np.argsort(data1))
    order2 = np.flip(np.argsort(data2))
    return kendall_tau_distance(order1, order2)
    
def smooth(x, window_size):
    out = np.convolve(x, np.ones(window_size,dtype=int), 'valid')/window_size    
    r = np.arange(1, window_size-1, 2)
    start = np.cumsum(x[:window_size-1])[::2]/r
    stop = (np.cumsum(x[:-window_size:-1])[::2]/r)[::-1]
    return np.concatenate((start, out, stop))
    

def information_gain(table):
    total = sum(sum(table))
    p_classes = sum(table)
    p_features = np.sum(table, axis=1)
    H_c = 0
    H_ca = 0
    for j in range(table.shape[1]):
        prob = p_classes[j]/total
        H_c = H_c - (prob*np.log2(prob))
    
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            prob = table[i,j]/p_features[i]
            H_ca = H_ca - (p_features[i]/total*prob*np.log2(prob))
    
    return H_c - H_ca
    
def calc_information_gain(counts):
    infogain = np.zeros((len(counts)))
    for i in range(len(counts)):
        infogain[i] = information_gain(counts[i])
    return infogain 
    
def calc_chisquare(counts):
    chi2 = np.zeros(len(counts))
    p = np.zeros(len(counts))
    for i in range(len(counts)):
        chi2[i], p[i], df, _ = chi2_contingency(counts[i])
    return chi2, p
    
def standardize(x):
    scaler = MinMaxScaler([0,1])
    output = dict()

    for key in x.keys():
        output[key] = np.concatenate(scaler.fit_transform(x[key].reshape(-1,1)), axis=0)

    return output