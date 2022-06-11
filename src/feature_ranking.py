from copyreg import pickle
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from utils import data_to_4D, get_data, get_reference, reshape_reference, category_cmap, category_norm, average_data, reference_forest, reference_buildings, reduce_and_undersample
import numpy as np
import matplotlib.pyplot as plt
from pixel_classifier import pixel_classifier
import pickle
import reliefe
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

def reference_target(ref):
    # return reference_forest(ref)
    return reference_buildings(ref)

# EDINA delujoča implementacija Reliefa za python!!!! IJS STRONG!
train = [2, 3, 4,  7, 8, 9,  12, 13, 14,  17, 19]
#train = [2]
#test = [1]
test = [1, 6, 11, 16]
X_train = np.concatenate([get_data(i) for i in train])
y_train = np.concatenate(
    [reference_target(get_reference(i)) for i in train])
X_test = np.concatenate([get_data(i) for i in test])
y_test = np.concatenate([reference_target(get_reference(i)) for i in test])
X_train_avr_full = np.concatenate(
    [average_data(get_data(i), [i for i in range(12)]) for i in train])
X_train_avr_spring = np.concatenate(
    [average_data(get_data(i), [i for i in range(12)]) for i in train])
X_test_avr_spring = np.concatenate(
    [average_data(get_data(i), [i for i in range(12)]) for i in test])


if __name__ == "__main__":
    with open('../img/eval-buildings-average/obj_svc', 'rb') as f:
        model = pickle.load(f)


    X, y = reduce_and_undersample(X_test_avr_spring, y_test, size=0.0001)
    print('permutation importances..')
    perm_importance = permutation_importance(model, X, y)

    feature_names = [i for i in range(15)]
    features = np.array(feature_names)

    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")

    # output: 
    # sorted_idx = [ 2, 10,  3,  0,  8,  7,  1,  4,  9, 14,  5, 12, 11,  6, 13]
    # perm_importance.importances_mean = 
    # array([0.01551157, 0.0486112 , 0.00062046, 0.01087051, 0.05190586,
    #   0.07708217, 0.20309818, 0.042177  , 0.04015015, 0.05303096,
    #   0.00544973, 0.17597361, 0.10466588, 0.35304957, 0.06302662])