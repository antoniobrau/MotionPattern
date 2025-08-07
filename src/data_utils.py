from pathlib import Path
import pandas as pd
import numpy as np
import os
import ast
import matplotlib.pyplot as plt
import cv2
from IPython.display import display, Image, clear_output
import ipywidgets as widgets
from itertools import product


class MaskedDataFrame(pd.DataFrame):
    """
    Classe che contiene un Dataframe dei dati con la possibilità di creare una maschera ai dati, in modo da selezionare patterns.
    Il dataframe deve avere un formato specifico:
        Columns:
        "Pattern"       --> Binary string representing the pattern "[1,0,..,1,0]"
        "p"             --> Stimed probability of the pattern.
        "Trust"         --> proportional to the quality of "p" estimation.  
        "SpaceFilter    --> Filter dimension. 
        "TimeFilter     --> Filter dimension.
        "Entropy"       --> just a columns with -plogp.
        "EntropyRatio"  --> plogp / sum(plogp) considering only Patterns with "Trust" > 7 in the sum.
        "PatternValue   --> Binary rapresentation of "Pattern" (int)
        "Mask"          --> Boolean Mask (all True by default)
    
    Arguments:
        pd.Dataframe data: dataframe imported with ImportData
    """
    def __init__(self, data):
        if isinstance(data, MaskedDataFrame):
            # Se è già MaskedDataFrame, crea una copia profonda
            data = data.copy(deep=True)
        super().__init__(data)
    
    def get_N(self):
        """
        Get number of patterns selected by the Mask.

        return : Number of Patterns
        """
        return len(self[self["Mask"]])

    def get_BandWidth(self):
        """
        Get Band width of patterns selected by the Mask.

        return : BandWidth
        """
        return self.loc[self["Mask"], "p"].sum()

    def get_Entropy(self):
        """
        Get Entropy of patterns selected by the Mask.

        return : Entropy of selected patterns
        """
        return self.loc[self["Mask"], "Entropy"].sum()

    def get_EntropyRatio(self):
        """
        Get Entropy of patterns selected by the Mask.

        return : Entropy ratio of selected patterns
        """
        return self.loc[self["Mask"], "EntropyRatio"].sum()
    def get_PatternsValue(self):
        """
        Restituisce la time series dei pattern selezionati in formato intero.

        Returns:
            pd.Series: Serie temporale di interi dove ogni elemento rappresenta un pattern,
                    filtrato dalla maschera.
        """
        return self.loc[self["Mask"]].index
    def get_PatternList(self):
        """
        Restituisce la lista dei pattern in formato numpy array.

        Ogni pattern viene convertito in un array tridimensionale con la seguente struttura:
        - Dimensione 0 → Filtro temporale (TimeFilter)
        - Dimensione 1 → Filtro spaziale in altezza (SpaceFilter)
        - Dimensione 2 → Filtro spaziale in larghezza (SpaceFilter)

        Il valore dei pattern è scalato a 255 e rappresentato come array di tipo uint8.

        Returns:
            list[np.ndarray]: Lista di array numpy con shape (TimeFilter, SpaceFilter, SpaceFilter),
                            dove ogni elemento rappresenta un pattern.
        """
        # Converte la stringa di ogni pattern in lista di interi usando ast.literal_eval
        lista = [ast.literal_eval(stringa) for stringa in self.loc[self["Mask"],"Pattern"]]

        # Recupera le dimensioni dei filtri temporale e spaziale dal DataFrame
        time_filter = int(self["TimeFilter"].iloc[0])
        space_filter = int(self["SpaceFilter"].iloc[0])

        # Converte ogni pattern in un array numpy e ne modifica la forma
        # Moltiplica per 255 per ottenere valori nell'intervallo [0, 255] e usa il tipo uint8
        lista = [255 * np.array(x, dtype=np.uint8).reshape((time_filter, space_filter, space_filter)) for x in lista]

        return lista

    def add_1_distance_columns(self):
        """
        Aggiunge al dataframe le colonne relative ai pattern con distanza 1 da quelli selezionati.
        """
        # Recupera i parametri dal DataFrame
        space_filter = int(self["SpaceFilter"].iloc[0])  
        time_filter = int(self["TimeFilter"].iloc[0])
        length = space_filter * space_filter * time_filter  # Correzione di `lenght`

        array = []
        # Aggiunge i pattern con distanza 1
        for ind in self.index:
            distance_1_patterns = np.array(Generate_Pattern_with_distance(ind, length, 1),dtype = np.uint64)
            array.append(distance_1_patterns)

        array = np.array(array)
        for i in range(length):  
            self[f"d1_{i}"] = array[:,i] 

    def print_info(self):
        """Stampa un riepilogo delle metriche chiave dei pattern selezionati."""
        print(f"Numero di pattern selezionati: {self.get_N()}")
        print(f"Larghezza di banda: {self.get_BandWidth():.4f}")
        print(f"Entropia totale: {self.get_Entropy():.4f}")
        print(f"Rapporto di entropia: {self.get_EntropyRatio():.4f}")


def ImportData(path, p : float  = 0.9, trust : float = 0.7):
    """
    L'algoritmo Top-k salva le simulazioni su un file.csv con le seguenti colonne :"Pattern", "Sum_level", "Count_level".
    Queste colonne rappresentano rispettivamente il livello associato al patterne e da quanti pool è stato rilevato.
    Da queste informazioni creo un dataframe con informazioni accessibili.
    La prima riga è un commento che contiene informazione di base come POOL_SIZE,SPACE_FILTER, TIME_FILTER, TOTAL_ITERATIONS
    
    Arguments:
        string path: Directory of the .csv file
        float p: geometric distribution parameter
        int trust: Threshold for the trust column. Patterns with trust < trust are not considered.
    
    Returns:
        MaskedDataFrame : higher level dataframe with columns :
        "Pattern"       --> Binary string representing the pattern "[1,0,..,1,0]"
        "p"             --> Stimed probability of the pattern.
        "Trust"         --> proportional to the quality of "p" estimation.  
        "SpaceFilter    --> Filter dimension. 
        "TimeFilter     --> Filter dimension.
        "Entropy"       --> just a columns with -plogp.
        "EntropyRatio"  --> plogp / sum(plogp) considering only Patterns with "Trust" > 7 in the sum.
        "PatternValue   --> Binary rapresentation of "Pattern" (int)
        "Mask"          --> Boolean Mask (all True by default)
    """

    # Se path è una stringa, lo converte in Path
    if not isinstance(path, Path):
        path = Path(path)

    #Leggo le informazioni di base dal file che mi serviranno per costruire le probabilità dei pattern
    with path.open('r') as f:
        line = f.readline()
        line = f.readline()[1:]  
        line = line.strip()  
        lista = line.split(";")[:-1]

    TotalIterations,SpaceFilter, TimeFilter, PoolSize, DimMultiLevel = list(map(float,lista))
    
    data = pd.read_csv(path, sep=';', comment="#")
    
    data['Trust'] = data["Count_level"]/(DimMultiLevel*PoolSize)
    data = data[data["Trust"] > trust]

    data['p_old'] = PoolSize * ((p + 1) / (4 * p) * p**(-data["Sum_level"]/data["Count_level"]) - 0.5) / TotalIterations
    data["p"] = PoolSize* np.exp(data["Sum_level"]/data["Count_level"]*(1 - p) - np.euler_gamma) / TotalIterations
    data["Entropy"] = -data["p"]*np.log(data["p"])
    data["EntropyRatio"] = data["Entropy"] / data.loc[data["Trust"] > trust,"Entropy"].sum()
    data['SpaceFilter'] = SpaceFilter
    data['TimeFilter'] = TimeFilter
    data['TotalIterations'] = TotalIterations
    data['Mask'] = True


    binary = lambda s : int(''.join(map(str, [int(x) for x in s.strip('[]').split(', ')])), 2)
    data["PatternValue"] = data["Pattern"]#.apply(binary)
    # data["PatternValue"] = data["Pattern"].apply(binary)


    columns = ["Pattern",'p','p_old', 'Trust','SpaceFilter', "Entropy","EntropyRatio", 'TimeFilter',"PatternValue","Mask", "TotalIterations"]
    data = data[columns]

    data.set_index("PatternValue", inplace = True)

    return MaskedDataFrame(data)


def ImportData_real_counting(path):

    # Se path è una stringa, lo converte in Path
    if not isinstance(path, Path):
        path = Path(path)

    
    
    data = pd.read_csv(path, sep=';', comment="#")
    data = pd.read_csv(r"C:\Users\braua\Documents\TesiMagistrale\Dataset\MOMENTS_IN_TIME\occorrenze_filtrati.csv", comment="#", sep=";")
    data["p"] = data["Occurrences"]


    data["Entropy"] = -data["p"]*np.log(data["p"])
    data["EntropyRatio"] = data["Entropy"] / data.sum()
    data['SpaceFilter'] = 3
    data['TimeFilter'] = 3
    data['Mask'] = True


    # data["PatternValue"] = data["Pattern"].apply(binary)


    columns = ["Pattern",'p','SpaceFilter', "Entropy","EntropyRatio", 'TimeFilter',"Mask"]
    data = data[columns]

    data.set_index("Pattern", inplace = True)

    return MaskedDataFrame(data)


def Generate_Pattern_with_distance(pattern: int, length: int, distance: int):
    combinazioni = product(range(length), repeat=distance) 
    patterns = []  
    
    for tupla in combinazioni:
        mask = 0
        for i in range(distance):
            mask |= 1 << tupla[i]  
        patterns.append(pattern ^ mask)  
    return patterns # Restituiamo una lista, se necessario

class Euristica(MaskedDataFrame):
    """
    Applicazione Euristica ai dati.
    Crea un altra maschera booleana per selezionare i pattern in base all'euristica.

    Args:
        data (MaskedDataFrame): Dataframe importato con ImportData.
        Max_W (float): Massima larghezza di banda dei pattern selezionati dall'euristica.
        Max_N (int): Numero massimo di pattern selezionati dall'euristica.
    """
    _metadata = ["Max_N","Max_W"]

    def __init__(self, data : MaskedDataFrame, Max_W : float, Max_N : int ):
        self.Max_W = Max_W
        self.Max_N = Max_N

        # Inizializza il DataFrame
        super().__init__(data)

        # Calcolo euristica
        self["Euristica"] = self["Entropy"] / np.maximum(np.ones(self["p"].shape) / self.Max_N, self["p"] / self.Max_W)

        # Ordina in base all'euristica
        data_temp = self.sort_values(by='Euristica', ascending=False)

        # Selezione dei pattern
        elements = self.Max_N
        while data_temp["p"][:elements].sum() > self.Max_W and elements > 0:
            elements -= 1

        # Crea la maschera booleana
        selected_indices = data_temp.index[:elements]
        self["Mask"] = self.index.isin(selected_indices)


def low_frequency_mask(data : Euristica, begin : int = 0):
    """
    Crea un maskedDataFrame selezionando tutti i pattern (a partire dalle basse frequenze) fino a fare un matching di informazione con 
    i pattern selezionati dall'euristica del dataset in input.
    Args:
        data (Euristica): Dataframe importato con ImportData e processato con l'euristica.
        begin (int): Indice di inizio per la selezione dei pattern.
    Returns:    
        MaskedDataFrame : higher level dataframe with columns :
    """
    low_data = MaskedDataFrame(data).sort_values(by = "p", ascending = True)
    low_data["Mask"] = False
    info = data.get_Entropy()
    elements = 1
    while low_data["Entropy"].iloc[begin : begin + elements].sum() < info:
        elements += 1
    elements -= 1
    low_data.loc[low_data.index[begin: begin + elements], "Mask"] = True

    return MaskedDataFrame(low_data)

def high_frequency_mask(data : Euristica, begin : int = 0):
    """
    Crea un maskedDataFrame selezionando tutti i pattern (a partire dalle alte frequenze) fino a fare un matching di informazione con 
    i pattern selezionati dall'euristica del dataset in input.
    Args:
        data (Euristica): Dataframe importato con ImportData e processato con l'euristica.
        begin (int): Indice di inizio per la selezione dei pattern.
    Returns:    
        MaskedDataFrame : higher level dataframe with columns :
    """
    high_data = MaskedDataFrame(data).sort_values(by = "p", ascending = False)
    high_data["Mask"] = False
    info = data.get_Entropy()
    elements = 1
    while high_data["Entropy"].iloc[begin : begin + elements].sum() < info:
        elements += 1
    elements -= 1
    high_data.loc[high_data.index[begin: begin + elements], "Mask"] = True

    return MaskedDataFrame(high_data)


    
  