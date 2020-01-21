# load the NN functions from previous examples
from Base10_NN import *

# load sklearn data and funcs needed
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

print('Loading and prep data ...')
digits = load_digits()
X_scale = StandardScaler()
X = X_scale.fit_transform(digits.data)
y = digits.target
print('ready.')

# setup the parameter selection function
def select_parameters(hidden_size, alpha, lamb, X, y):
    X_train, X_holdover, y_train, y_holdover = train_test_split(X, y, test_size=0.4)
    X_valid, X_test, y_valid, y_test = train_test_split(X_holdover, y_holdover, 
                                       test_size=0.5)
    # convert the targets (scalars) to vectors
    yv_train = convert_y_to_vect(y_train)
    yv_valid = convert_y_to_vect(y_valid)
    results = np.zeros((len(hidden_size)*len(alpha)*len(lamb), 4))
    cnt = 0
    for hs in hidden_size:
        for al in alpha:
            for l in lamb:
                nn_structure = [64, hs, 10]
                n_iterations = 300
                print("\nTesting params: {} @ {} iterations ...".format([hidden_size, alpha, lamb], n_iterations))
                start = time.time()
                W, b, avg_cost = train_nn(nn_structure, X_train, yv_train, 
                                    iter_num=n_iterations, alpha=al, lamb=l)
                y_pred = predict_y(W, b, X_valid, 3)
                accuracy = accuracy_score(y_valid, y_pred) * 100
                end = time.time()
                print("Accuracy is {}% for {}, {}, {}".format( round(accuracy,2), hs, al, l))
                print('Time elapsed (sec) = '+str(round(end - start,2)))

                # store the data
                results[cnt, 0] = accuracy
                results[cnt, 1] = hs
                results[cnt, 2] = al
                results[cnt, 3] = l
                cnt += 1
    # get the index of the best accuracy
    best_idx = np.argmax(results[:, 0])
    return results, results[best_idx, :]

##############################################################################

# main thread

#  Parameters

# 1. The number of hidden nodes
hidden_size = [10, 25, 50, 60]

# 2. The learning rate
alpha = [0.05, 0.1, 0.25, 0.5]

# 3. The regularisation constant
lamb = [0.0001, 0.0005, 0.001, 0.01]


p = input('Ready to search for params (y/n)?').upper()
if p[0] == 'Y':
    r, br = select_parameters(hidden_size, alpha, lamb, X, y)
    print('\nBest case: '+str(br))
print('End.')