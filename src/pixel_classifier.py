from unittest import result
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
RANDOM_STATE = 420


def pixel_classifier(X_train, y_train, grid, model, grid_search=True, undersampling=0.001, verbose=True):
    '''
    Vrne DecisionTreeClassifier,
    Najprej se izbere najboljše parametre za model na undersamplanih podatkih (z random samplerjem), nato pa se 
    model natrenira na oversamplanih (s SMOTE). 

    Parametri
    -----------
     - X_train - podatki. Morjo bit istih dimenzij kot y_train 
     - undersampling - velikost manjše množice glede na večjo
    '''
    # UNDERSAMPLING - optimalne parametre poiščem na manjši množici, ker bo sicer trajal sto let
    # najprej izberem manjšo podmnožico
    if grid_search:
        if verbose:
            print('Sampling subset')
        X_small, _, y_small, _ = train_test_split(
            X_train, y_train, train_size=undersampling, random_state=RANDOM_STATE, stratify=y_train)
        rus = RandomUnderSampler(random_state=RANDOM_STATE)
        X_undersampled, y_undersampled = rus.fit_resample(X_small, y_small)

        # ISKANJE OPTIMALNIH PARAMETROV
        if verbose:
            print('Grid search..')
        tree = DecisionTreeClassifier()
        grid.fit(X_undersampled, y_undersampled)

        # predelani podatki
        if verbose:
            print('Undersampling..')
        # SVC so prepočasni
        #X_train, _, y_train, _ = train_test_split(X_train, y_train, train_size=undersampling*15, stratify=y_train)
        if model == SVC:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=0.01, stratify=y_train)
        if model == RandomForestClassifier or model == KNeighborsClassifier:
            X_train, _, y_train, _ = train_test_split(
                X_train, y_train, train_size=0.01, stratify=y_train)
        X_res, y_res = rus.fit_resample(X_train, y_train)
        params = grid.best_params_
        # naredimo novo drevo s starimi parametri
        model = model(**params)
    # novo drevo naučimo na vseh podatkih
    if verbose:
        print('Fitting on all training data..')
    model.fit(X_res, y_res)

    return model


def pixel_classifier_tree(X_train, y_train, param_grid, undersampling=0.0007, verbose=True):
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
    def reference_target(ref):
        # return reference_forest(ref)
        return reference_buildings(ref)

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

    

    models = {
        #'svc': GridSearchCV(SVC(), param_grid={'C': [2, 4, 6, 8], 'kernel': ['poly', 'rbf', ], 'degree': [5], 'probability': [True]}, scoring='accuracy'),
           'svc': GridSearchCV(SVC(), param_grid={'C': [2*i for i in range(2, 8)], 'kernel': ['rbf'],  'probability' : [True]}, scoring='accuracy'),
           'tree': GridSearchCV(DecisionTreeClassifier(), param_grid={'min_samples_split': [2, 4, 8],
                                                                      'min_samples_leaf': [1, 2, 4],
                                                                      'ccp_alpha': [0.0, 0.1, 0.2, 0.4, 0.5]}, scoring='accuracy'),
           'rf': GridSearchCV(RandomForestClassifier(), param_grid={'max_depth': [50, 100, 120], 'ccp_alpha': [0.0, 0.1, 0.2, 0.4, 0.5]}, scoring='accuracy'),
        'knn': GridSearchCV(KNeighborsClassifier(), param_grid={'n_neighbors': [3, 5, 8, 10]}, scoring='accuracy')
    }
    func_names = {'tree': DecisionTreeClassifier,
                  'rf': RandomForestClassifier, 'svc': SVC, 'knn': KNeighborsClassifier}
    results = {}
    for name, base_model in models.items():
        #   if name == 'tree' or name == 'rf':
        #        continue
        print(f'Fitting model {name}..')
        model = pixel_classifier(X_train, y_train, base_model, func_names[name], undersampling=0.006)
        #model = pixel_classifier(X_train_avr_spring, y_train, base_model, func_names[name], undersampling=0.006)
        # napaka
        prob_predictions = model.predict_proba(X_test)[:,1]
        #prob_predictions = model.predict_proba(X_test_avr_spring)[:, 1]
        #predictions = model.predict(X_test)
        #mse = mean_squared_error(y_test, prob_predictions*2)
        mse = mean_squared_error(y_test, prob_predictions*8)
        print(f'Model {name} has MSE={mse}')
        results[name] = mse

        for j in test:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

            axs[0].imshow(data_to_4D(get_data(j))[..., 8, [3, 2, 1]] * 3)
            axs[1].imshow(reference_target(reshape_reference(model.predict(get_data(j)))), cmap=category_cmap, norm=category_norm)
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
            plt.savefig(f'../img/eval-buildings-big/{name}_{j}')
        print('saving model info..')
        with open(f'../img/eval-buildings-big/{name}.txt', 'w') as f:
            f.write(str(model.get_params()))
        with open(f'../img/eval-buildings-big/obj_{name}', 'wb') as f:
            pickle.dump(model, f)

    print(results)
