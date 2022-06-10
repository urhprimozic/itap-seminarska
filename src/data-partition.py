#------------------- IDEJA ----------
# js delam pixel recognition za vsak pixel, nato pa za en pixel pogledam še sosede na kvadratu 10x10
# zato enostavna particija na pixle odpade, ampak rabim delit po kosih. 
# IDEJA: razrežem mapo na manjše kvadrate in ene vzamem za test, ene pa za train. Tako imam zagotovilo,
# da bom celotno testiranje delal na neznanih pixlih.
# Osnovna delitev na kvadrate 300x300 se mi zdi mal velika. Mogoče bi blo bolš delit 100x100.
# Podatke razrežem na samostojne enote CHUNK_SIZExCHUNK_SIZE. Te razdelim na train + test set

import numpy as np
from sklearn.model_selection import train_test_split

# indeksi podatkov, ki jih bomo uporabili (tisti brez referenc gredo stran in 18 gre stran, ker ma letališče*)
# *TODO če bo slučajno premalo podatkov, spemeni 18 v travo iz pozidanega
valid_data = [1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 19]

      

def data_index_to_coordinates(i):
    '''
    Vrne tuple (vrstica, stolpec) pixla levo zgoraj v podatki_i.npy'''
    return  (4 - i % 5)*300, (i//5) * 300

def train_test_chunks(CHUNK_SIZE, train_size):
    '''
    Razdeli podatke na kose CHUNK_SIZExCHUNK_SIZE in te razdeli na učne in testne. 

    Parametri
    --------
        - CHUNK-SIZE - velikost posameznih kosov
        - test_size - delež učnih primerov v delitvi
    Vrne
    ------
    Tuple (X_train, X_test, y_train, y_test), kjer je vsak element tabela pixlov.
    '''
    if not 300 % CHUNK_SIZE == 0:
        raise NotImplementedError('CHUNK_SIZE naj deli 300, da so lahko vsi kvadrati enako veliki')

    # vrstica, stolpec pixla levo zgoraj
    for i in valid_data:
        # absolutne koordinate pixla levo zgoraj v podatki_i na celotni shemi vsega kar mam, torej 1500*1500
        vrstica = (4 - i % 5)*300
        stolpec = (i//5) * 300
    raise NotImplementedError

