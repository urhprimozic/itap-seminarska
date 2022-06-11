# pobrano iz repozitorija od Ljupča
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

WIDTH, HEIGHT = 300, 300
TIME, FEATURES = 12, 15
INPUT_FOLDER = os.path.join(".", "../data")  # pot do mape s podatki
references = [0,5,10,15,20,21,22,23,24]
colors = {
    0: "#ffffff",  # No Data
    1: "#ffff00",  # Cultivated Land
    2: "#054907",  # Forest
    3: "#ffa500",  # Grassland
    4: "#806000",  # Shrubland
    5: "#069af3",  # Water
    6: "#95d0fc",  # Wetlands
    7: "#967bb6",  # Tundra
    8: "#dc143c",  # Artificial Surface
    9: "#a6a6a6",  # Bareland
    10: "#000000",  # Snow and Ice
}
category_cmap = ListedColormap(list(colors.values()), name="category_cmap")
category_norm = BoundaryNorm([x - 0.5 for x in range(len(colors) + 1)], category_cmap.N)
GOZD = 2
POZIDANO = 8
SNEGLED = 10

def get_reference(N:int):
    '''Vrne tabelo referenc polja N'''
    if N in references:
        raise Exception('O teh poljih nimaš podatkov brrt')
    with open(os.path.join(INPUT_FOLDER, f'reference_{N}.npy'), 'rb') as filehandle:
        array = np.load(filehandle)
    return array

def reference_forest(reference):
    '''
    Spremeni vse, kar ni gozd(2) v 0 (no data)
    '''
    reference[reference != GOZD] = 0
    return reference

def reference_buildings(reference):
    '''
    Spremeni vse, kar ni pozidano(2) v 0 (no data)
    '''
    reference[reference != POZIDANO] = 0
    return reference


def get_data(N: int, reference=False):
    '''Vrne tabelo oblike 90.000 x 180, ki predstavlja polje številka N. Če je reference=True, vrne reference
    '''
    if reference:
        return get_reference(N)
    with open(os.path.join(INPUT_FOLDER, f'data_{N}.npy'), 'rb') as filehandle:
        array = np.load(filehandle)
    return array
# Datoteke vsebujejo 90.000 vrstic in 180 (data.npy) ali 1 (reference.npy) stolpec.
# Za grafični prikaz je lažje, če obliko spremenimo v 300x300, pri podatkih pa še 180 stolpcev pretvorimo v 12x15
def data_to_4D(array, height=HEIGHT, width=WIDTH, time=TIME, features=FEATURES):
    '''Pretvori 2D podatke v 4D prikaz. Indeksi so:
        1. vrstica (300)
        2. stolpec (300)
        3. mesec (12)
        4. valovna dolžina (oz. NDI) (15)'''
    return array.reshape(height, width, time, features)

def average_data(data, meseci):
    '''
    Vzame povprečje vseh značilk čez izbrane mesece
    '''
    if not data.shape == (90000, 180):
        raise NotImplementedError('Shape should be (90000, 180)')
    new_data = np.zeros((90000, 15))
    for i in range(15):
        # vsako izmed 15tih značil izpovprečiš po vseh 12 mesecih
        # features = np.concatenate([data[...,i+j*15] for j in range(0,12)], axis=1)
        indexes = [i+j*15 for j in meseci]
        features = data[..., indexes]
        new_data[...,i] = np.average(features, axis=1)
    return new_data


# Pri referenci zgolj spremenimo 90.000x1 obliko v 300x300 obliko
def reshape_reference(array, height=HEIGHT, width=WIDTH):
    '''Pretvori referenčne podatke v obliko primerno za prikazovanje.'''
    return array.reshape(height, width)

def plot_reference(N : int):
    plt.figure(figsize=(8, 8))
    plt.imshow(reshape_reference(get_reference(N)), cmap=category_cmap, norm=category_norm)
    plt.show()


def plot_data(N : int, mesec, kanali=[3, 2, 1], lighting=3.5, reference=False):
    '''Izriše poljše številka N v c določenem mesecu. 
    '''
    if reference:
        plot_reference(N)
    array_4d = data_to_4D(get_data(N))
    plt.figure(figsize=(8, 8))
    plt.imshow(array_4d[..., mesec, kanali] * lighting)
    plt.show()

def plot_data_reference(N : int, mesec, kanali=[3, 2, 1], lighting=3.5, reference=False):
    array_4d = data_to_4D(get_data(N))
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    axs[0].imshow(array_4d[..., mesec, kanali] * lighting)
    axs[1].imshow(reshape_reference(get_reference(N)), cmap=category_cmap, norm=category_norm)
    plt.show()