# iskanje optimalnega modela na pixklih za nove featurje
from pixel_classifier import pixel_classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from utils import average_data, data_to_4D, get_data, get_reference, reshape_reference, category_cmap, category_norm, reference_forest, reference_buildings
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
from new_features import monte_carlo_shiti_method

def reference_target(ref):
    # return reference_forest(ref)
    return reference_buildings(ref)


# klasična Patricija
train = [2, 3, 4,  7, 8, 9,  12, 13, 14,  17, 19]
test = [1, 6, 11, 16]

# novi featurji
#new_features = [(-13, -12, 12, 12), (-12, -7, 0, 8), (-11, -2, 2, 8), (-12, -10, -7, 7, 11), (-10, -8, 8, 11, 12), (7, 14), (9, 11), (-13, -12, -10, -2, 10), (-7, 0), (-11, -1, 2, 9), (-2, 2), (-11, 0), (-11, 0, 7, 9, 12), (-8,), (12,), (-12, -7, -5, 14), (-9, -8, -7, -1, 5), (-10, -9, -7, 12, 13), (-12, -11, -5, -1, 5), (1,), (-2,), (-14,), (2,), (-6, -2, 0, 4), (-11, -9, 9), (-11, -6, 2, 12), (-12,), (5,), (-11, -8, 5), (-12, -4), (3, 13), (8, 12), (-5, -4, 1), (-12, -10, -2, 2, 7), (-4,), (-4, 13), (-6, -2, 9), (-10, -6, -6, -5, 11), (-14, -12, -11, -2, 0), (2, 7), (-9, -6, -2), (-10, 6), (-5, 2), (-7, 7, 10, 12), (11,), (5, 7, 10), (3,),
#            (10,), (-2, 12), (-3,), (-14, -5, -5, 3), (-11, -7, 3), (4,), (-9, 2, 5, 9), (-8, -2), (13,), (9,), (-12, -9), (-11, -11, 0, 12, 12), (-3, 12), (8,), (-7, 2), (-10, -8, -2, 7), (-10, 2, 5, 7), (-10, -2, 2, 3, 4), (-2, 7, 10, 12), (-14, -13, -11, 1, 12), (7,), (-5, 3, 8), (-9,), (-6, -5, -3, 2, 11), (-13, 5), (-13, 0, 8), (14,), (-4, 7, 9, 10, 12), (-10, -9, 2, 6), (-2, 0, 7, 12), (0, 14), (-13, -11, 0, 7, 9), (-12, -7, 10, 10), (-10, 2, 7, 10, 11), (0,), (-10, -7, -7, 2, 10), (6,), (-10, -2, 0, 2), (-2, 0, 7, 9), (-9, -8, -2), (-10, -1, 12), (-13, -11, -6, 0, 7), (-11, -10, -9), (-11, -8, 2, 14), (-8, 5), (-2, 12, 12), (7, 10), (0, 2, 14)]
#new_features = [(i,) for i in range(15)]
# new_features = monte_carlo_shiti_method([i for i in range(15)], [1/15 for i in range(15)], n_iter=20, depth=2)
new_features = [ 1,  4,  5,  6,  7,  8,  9, 11, 12, 13, 14] # le še 8 spektralnih kanalov
new_features = [(i,) for i in new_features]
#prvih 8 se sliš ok, to je 12 dimenzij mnj
features = new_features #[:10]
# to je zdj X s priloženimi 12*15 featurji
X = np.concatenate([get_data(i) for i in train + test])
INF = np.max(X)*1000
BigX = []
l = X.shape[0]
print('Prepearing data..')
for month in tqdm(range(12), total=12):
    for feature in features:
        if len(feature) == 1 and feature[0] < 0:
            continue
        #print(feature)
        if feature[0] >= 0:
            vector = X[...,feature[0]+month*15].reshape(l,1)
        else:
            vector = 1/X[...,feature[0]+month*15].reshape(l,1)
        #primnožim/zdelim še ostale stolpce
        for f in feature[1:]:
            if f >= 0:
                vector *= X[..., f+month*15].reshape(l,1)
            else:
                t = X[..., -f+month*15].reshape(l,1)
                t[t == [0]] =   0.001    
                vector /= X[..., -f+month*15].reshape(l,1)
        # print(vector.shape)
        BigX.append(vector)
BigX = np.concatenate(BigX, axis=1)
# popravi inf vrednosti na nekej velikega
BigX[ np.isfinite(BigX) == False] = INF

# patricijajajajaaj rada fafafa fantaziara programira
# poznov noč pa fiks ne
X_train = BigX[:len(train)*300**2]
X_test = BigX[-len(test)*300**2:]
y_train = np.concatenate(
    [reference_target(get_reference(i)) for i in train])
y_test = np.concatenate([reference_target(get_reference(i)) for i in test])


models = {
    # 'svc': GridSearchCV(SVC(), param_grid={'C': [2, 4, 6, 8], 'kernel': ['poly', 'rbf', ], 'degree': [5], 'probability': [True]}, scoring='accuracy'),
    'svc': GridSearchCV(SVC(), param_grid={'C': [2*i for i in range(2, 8)], 'kernel': ['rbf'],  'probability': [True]}, scoring='accuracy'),
  #  'tree': GridSearchCV(DecisionTreeClassifier(), param_grid={'min_samples_split': [2, 4, 8],
    #                                                           'min_samples_leaf': [1, 2, 4],
    #                                                           'ccp_alpha': [0.0, 0.1, 0.2, 0.4, 0.5]}, scoring='accuracy'),
  #  'rf': GridSearchCV(RandomForestClassifier(), param_grid={'max_depth': [50, 100, 120], 'ccp_alpha': [0.0, 0.1, 0.2, 0.4, 0.5]}, scoring='accuracy'),
    'knn': GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [3, 5, 8, 10]}, scoring='accuracy')
}
func_names = {'tree': DecisionTreeClassifier,
                  'rf': RandomForestClassifier, 'svc': SVC, 'knn': KNeighborsClassifier}
results = {}
for name, base_model in models.items():
    #   if name == 'tree' or name == 'rf':
    #        continue
    print(f'Fitting model {name}..')
    model = pixel_classifier(X_train, y_train, base_model, func_names[name], undersampling=0.01)#00005
    #model = pixel_classifier(X_train_avr_spring, y_train, base_model, func_names[name], undersampling=0.006)
    # napaka
    prob_predictions = model.predict_proba(X_test)[:,1]
    print('-----------', X_test.shape)
    #prob_predictions = model.predict_proba(X_test_avr_spring)[:, 1]
    #predictions = model.predict(X_test)
    #mse = mean_squared_error(y_test, prob_predictions*2)
    mse = mean_squared_error(y_test, prob_predictions*8)
    print(f'Model {name} has MSE={mse}')
    results[name] = mse

    for indexj, j in enumerate(test):
        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        axs[0].imshow(data_to_4D(get_data(j))[..., 8, [3, 2, 1]] * 3)
        test_data = X_test[300**2*indexj : 300**2*(indexj+1)]
        print('-----------xTest', X_test.shape)
        print('-----------test data', test_data.shape)
        axs[1].imshow(reference_target(reshape_reference(model.predict(test_data))), cmap=category_cmap, norm=category_norm)
        #axs[1].imshow(reference_target(reshape_reference(model.predict(
            #   average_data(get_data(j), [i for i in range(0,12)])))), cmap=category_cmap, norm=category_norm)
        axs[2].imshow(reference_target(reshape_reference(get_reference(j))),
                        cmap=category_cmap, norm=category_norm)
        fig.suptitle(f'Evaluation - {name}, mse={mse}')
        for ax, title in zip(axs, ("Slika", "Napoved", "Referenca")):
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("auto")
        plt.savefig(f'../img/eval-features/{name}_{j}')
    print('saving model info..')
    with open(f'../img/eval-features/{name}.txt', 'w') as f:
        f.write(str(model.get_params()))
    with open(f'../img/eval-features/obj_{name}', 'wb') as f:
        pickle.dump(model, f)

print(results)