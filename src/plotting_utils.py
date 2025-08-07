from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Optional

def ScatterPlot(x: pd.DataFrame, y: pd.DataFrame, x_label: Optional[str] = None, y_label: Optional[str] = None, save_fig: Optional[Path] = None):
    """
    Scatter plot of two DataFrames merged on 'PatternValue'.

    Arguments:
        x (pd.DataFrame): First DataFrame.
        y (pd.DataFrame): Second DataFrame.
        x_label (Optional[str]): Label for x-axis.
        y_label (Optional[str]): Label for y-axis.
        save_fig (Optional[Path]): Path to save the figure.
    """
    # Merge based on 'PatternValue'
    combined = pd.merge(x, y, on='PatternValue', how='inner')
    print(len(combined))

    # Plot
    plt.figure(figsize=(8, 6))
    plt.title("Scatter plot")
    if x_label and y_label:
        plt.suptitle(f"{x_label} vs {y_label}")

    # Punti più piccoli e semi-trasparenti
    plt.scatter(combined["p_x"], combined["p_y"], s=5, alpha=0.7)  # s controlla la dimensione dei punti

    # Scale logaritmiche
    plt.xscale("log")
    plt.yscale("log")
    
    # Etichette
    plt.xlabel(f"p {x_label}" if x_label else "p")
    plt.ylabel(f"p {y_label}" if y_label else "p")
    
    # Griglia meno fitta
    plt.grid(True, which="major", linestyle='--', linewidth=0.5)

    # Disegna la bisettrice
    min_val = min(combined["p_x"].min(), combined["p_y"].min())
    max_val = max(combined["p_x"].max(), combined["p_y"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1, label='y = x')


    # Save or Show
    if save_fig is not None:
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig)
        print(f"Figura salvata in: {save_fig}")
    else:
        plt.show()

def ScatterPlotEuristica(x: pd.DataFrame, y: pd.DataFrame, x_label: Optional[str] = None, y_label: Optional[str] = None, save_fig: Optional[Path] = None):
    """
    Scatter plot of two DataFrames merged on 'PatternValue'.

    Arguments:
        x (pd.DataFrame): First DataFrame.
        y (pd.DataFrame): Second DataFrame.
        x_label (Optional[str]): Label for x-axis.
        y_label (Optional[str]): Label for y-axis.
        save_fig (Optional[Path]): Path to save the figure.
    """
    # Merge based on 'PatternValue'
    combined = pd.merge(x, y, on='PatternValue', how='inner')

    # Crea maschere per i quattro gruppi
    mask1 = combined["Mask_x"] == True
    mask2 = combined["Mask_y"] == True
    mask = (combined["Mask_x"] == True) & (combined["Mask_y"] == True)
    pattern_comuni = len(combined[(combined["Mask_x"] == True) & (combined["Mask_y"] == True)])
    pattern1 = len(combined[(combined["Mask_x"] == True)])
    pattern2 = len(combined[(combined["Mask_y"] == True)])
    
    # Crea lo scatter plot con pallini più piccoli e colori diversi
    plt.scatter(combined["p_x"],combined["p_y"], s=6, marker='o', c='gray')
    plt.scatter(combined[mask1]["p_x"], combined[mask1]["p_y"], s=6, marker='o', c='blue', alpha=0.2, label=f'Pattern selezionati {x_label}')
    plt.scatter(combined[mask2]["p_x"], combined[mask2]["p_y"], s=6, marker='o', c='yellow', alpha=0.2,label=f'Pattern selezionati {y_label}')

    
    # Imposta la scala logaritmica per entrambi gli assi
    plt.xscale("log")
    plt.yscale("log")
    
    # Aggiungi etichette agli assi
    plt.xlabel(f'p {x_label}', fontsize=14)
    plt.ylabel(f'p {y_label}', fontsize=14)
    
    # Disegna la bisettrice
    min_val = min(combined["p_x"].min(), combined["p_y"].min())
    max_val = max(combined["p_x"].max(), combined["p_y"].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1)
    
    # Aggiungi titolo
    plt.title(f'Scatter plot of p : {x_label} vs {y_label}', fontsize=16)
    plt.suptitle(f"{x_label} : N = {pattern1}; {y_label} : N = {pattern2}; Comuni = {pattern_comuni}")
    
    # Aggiungi legenda
    plt.legend(fontsize=12)
    
    # Aggiungi griglia
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


    # Save or Show
    if save_fig is not None:
        save_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig)
        print(f"Figura salvata in: {save_fig}")
    else:
        plt.show()


def Hist( data : pd.DataFrame, title : str, save_fig: Optional[Path] = None):
    """
    Crea un istogramma dei pattern selezionati dall'euristica.

    Args:
        data (pd.DataFrame): DataFrame contenente i pattern selezionati dall'euristica.
        title (str): Titolo dell'istogramma.
        save_fig (Optional[Path]): Percorso in cui salvare la figura.
    """
    min_log_p = np.log10(data["p"].min())
    bins = np.logspace(min_log_p, 0, 250)

    SpaceFilter =int( data["SpaceFilter"].iloc[0])
    TimeFilter =int( data["TimeFilter"].iloc[0])
    # Crea la figura e l'asse
    fig, ax = plt.subplots(figsize=(12, 8))

    # Istogramma
    ax.hist(data["p"], bins=bins, alpha=1, color='gray', edgecolor='black')
    try:
        ax.hist(data.loc[data["Mask"],"p"], bins = bins, alpha = 1, label = f"Selected Patterns (N={data.get_N()}, W={data.get_BandWidth():.3f}, Entropy ratio = {data.get_EntropyRatio():.3f})", color = "green", edgecolor='black')
    except:
        ax.hist(data.loc[data["Mask"],"p"], bins = bins, alpha = 1, color = "green", edgecolor='black')
    # Etichette e titolo
    ax.set_xlabel('p', fontsize=16)
    ax.set_ylabel('Frequency', fontsize=16)
    try:
        ax.set_title(f'Histogram of p: Max W = {data.Max_W},  max N = {data.Max_N}', fontsize=20)
    except:
        ax.set_title(f'Histogram of p', fontsize=20)
    fig.suptitle(title, fontsize=20)
    ax.set_xscale('log')

    # Personalizzazione della griglia
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    
    # Personalizzazione dei tick
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Legenda
    ax.legend(fontsize=14)

    plt.tight_layout()

        # Salva la figura se il percorso di salvataggio è specificato
    if save_fig:
        fig.savefig(save_fig)

    else :
        plt.show()


def Plot_Img(patterns : list, SpaceFilter  : int = 3, TimeFilter  : int = 3, title_list : list = None):
    if title_list is not None and len(title_list) != len(patterns):
        raise ValueError("The number of titles must be equal to the number of patterns")

    lista = [ np.array(list(map(int,np.binary_repr(p, width = 27)))).reshape(TimeFilter,SpaceFilter,SpaceFilter) for p in patterns]
    
    # Cicla attraverso le immagini e mostra ciascuna in una figura separata
    for idx,image in enumerate(lista):
        gray_image = np.ones((SpaceFilter, SpaceFilter*TimeFilter + (TimeFilter - 1)))*0.5
        for i,box in enumerate(image):
            gray_image[:,i*SpaceFilter + i:(i+1)*SpaceFilter + i] = box
        plt.figure(figsize=(5, 5), facecolor='gray')  # Imposta la dimensione della figura
        plt.title(f"Pattern {patterns[idx]}" if title_list is None else title_list[idx])
        plt.imshow(gray_image, cmap='gray', vmin=0, vmax=1)  # Mostra l'immagine in scala di grigi
        plt.axis('off')  # Rimuove gli assi per una visualizzazione più pulita
        plt.show()
