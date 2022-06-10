from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from utils import data_to_4D, get_data, get_reference, reshape_reference, category_cmap, category_norm
import numpy as np
import matplotlib.pyplot as plt
RANDOM_STATE = 420


def pixel_classifier(X_train, y_train, param_grid, undersampling=0.001, verbose=True):
    '''
    Vrne DecisionTreeClassifier, ki zaznava gozd, natreniran na predelanih podatkih.
    Najprej se izbere najboljše parametre za model na undersamplanih podatkih (z random samplerjem), nato pa se 
    model natrenira na oversamplanih (s SMOTE). 

    Parametri
    -----------
     - X_train - podatki. Morjo bit istih dimenzij kot y_train 
     - undersampling - velikost manjše množice glede na večjo
    '''
    # UNDERSAMPLING - optimalne parametre poiščem na manjši množici, ker bo sicer trajal sto let
    # najprej izberem manjšo podmnožico
    if verbose:
        print('Undersampling..')
    X_small, _, y_small, _ = train_test_split(
        X_train, y_train, train_size=undersampling, random_state=RANDOM_STATE, stratify=y_train)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    X_undersampled, y_undersampled = rus.fit_resample(X_small, y_small)

    # ISKANJE OPTIMALNIH PARAMETROV
    if verbose:
        print('Grid search..')
    tree = DecisionTreeClassifier()
    # accuracy je ok metrika, ker smo uporabili undersampling
    final_tree = GridSearchCV(estimator=tree, param_grid=param_grid,
                              cv=10, verbose=3,  n_jobs=-1, scoring='accuracy')
    final_tree.fit(X_small, y_small)

    # predelani podatki
    if verbose:
        print('Oversampling..')
    # smote = SMOTE(random_state=RANDOM_STATE)
    # X_res, y_res = smote.fit_resample(X_train, y_train)

    X_res, y_res = rus.fit_resample(X_train, y_train)
    params = final_tree.best_params_
    # naredimo novo drevo s starimi parametri
    final_tree = DecisionTreeClassifier(**params)
    # novo drevo naučimo na vseh podatkih
    if verbose:
        print('Final fit..')
    final_tree.fit(X_res, y_res)

    return final_tree


if __name__ == "__main__":
    train = [2, 3, 4,  7, 8, 9,  12, 13, 14,  17, 19]
    # train = [2, 4,  8,  19]
    test = [1, 6, 11, 16]
    X_train = np.concatenate([get_data(i) for i in train])
    y_train = np.concatenate([get_reference(i) for i in train])
    X_test = np.concatenate([get_data(i) for i in test])
    y_test = np.concatenate([get_reference(i) for i in test])

    model = pixel_classifier(X_train, y_train, param_grid={'min_samples_split': [2, 4, 8],
                                                           'min_samples_leaf': [1, 2, 4],
                                                           'ccp_alpha': [0.0, 0.1, 0.2, 0.5]
                                                           })
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    axs[0].imshow(data_to_4D(get_data(1))[..., 8, [3, 2, 1]] * 3.5)
    axs[1].imshow(reshape_reference(model.predict(get_data(1))),
                  cmap=category_cmap, norm=category_norm)
    axs[2].imshow(reshape_reference(get_reference(1)),
                  cmap=category_cmap, norm=category_norm)

    for ax, title in zip(axs, ("Slika", "Napoved", "Referenca")):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")
