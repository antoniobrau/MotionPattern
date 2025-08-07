import cv2
import numpy as np
import os
import sys
from pathlib import Path
from IPython.display import display, Image, clear_output
import ipywidgets as widgets
from typing import List, Optional
from collections import deque
from itertools import product
from pymediainfo import MediaInfo


def Generate_Pattern_with_distance(pattern: int, length: int, distance: int):
    combinazioni = product(range(length), repeat=distance) 
    patterns = []  
    
    for tupla in combinazioni:
        mask = 0
        for i in range(distance):
            mask |= 1 << tupla[i]  
        patterns.append(pattern ^ mask)  
    return patterns # Restituiamo una lista, se necessario

class SketchGenerator:
    """
    Classe per la ricostruzione di video basata su pattern selezionati.

    Questa classe permette di ricostruire un video applicando filtri spaziali e temporali 
    ai frame originali e selezionando solo i pattern specificati. Il risultato è un video 
    "sketch" che evidenzia le regioni di interesse secondo i pattern forniti.

    Args:
        path_in (str): Percorso del video di input da elaborare.
        path_out (str): Percorso dove salvare il video ricostruito.
        list_patterns (list[int]): Lista di pattern (in formato intero) da utilizzare per la ricostruzione.
        TimeFilter (int): Filtro temporale che definisce la finestra di frame considerati.
        SpaceFilter (int): Filtro spaziale che definisce l'area su cui viene calcolato ogni pattern.
        frame_rate (int, optional): Frequenza dei frame del video di output. Se None, viene mantenuto quello originale.
        max_frame (int, optional): Numero massimo di frame da processare. Se None, vengono considerati tutti i frame.

    Attributi:
        Patterns_set (set): Insieme dei pattern unici utilizzati per il matching.
        TimeFilter (int): Finestra temporale utilizzata per l'analisi dei pattern.
        SpaceFilter (int): Filtro spaziale applicato al video.
        pattern_grid (np.ndarray): Matrice che memorizza i pattern estratti dai frame.
        frames (np.ndarray): Frame originali memorizzati in memoria.
        cap (cv2.VideoCapture): Oggetto per leggere i frame dal video di input.
        frame_rate (int): Frequenza dei frame del video di output.
        max_frames (int): Limite massimo di frame da processare.

    Esempio d'uso:
    -------------
    >>> df = DU.Euristica(data, 0.01, 500)
    >>> lista = df.get_PatternsValue()
    >>> prova_dir = Path("C:/Users/braua/Documents/TesiMagistrale") / "prova1.avi"
    >>> sk = SG.Sketch(DIRECTORY_VIDEO / "grano.avi", prova_dir, lista, TimeFilter=3, SpaceFilter=3)
    >>> sk.generate_sketch()

    Descrizione:
    -----------
    In questo esempio, si crea un oggetto `Sketch` che prende un video di input ("grano.avi"), 
    lo elabora con una finestra temporale e spaziale di dimensione 3, e lo salva come un video 
    ricostruito nel percorso indicato in `prova_dir`. La lista di pattern da usare nella ricostruzione 
    viene ottenuta attraverso un'euristica definita in `DU.Euristica`.
    """

    def __init__(self, 
                 path_in: str, 
                 path_out: str, 
                 list_patterns: List[int], 
                 TimeFilter: int, 
                 SpaceFilter: int, 
                 frame_rate: Optional[int] = None, 
                 max_frame: Optional[int] = None) -> None:
        """
        Inizializza la classe Sketch per la ricostruzione di video basata su pattern selezionati.

        Args:
            path_in (str): Percorso del video di input.
            path_out (str): Percorso per salvare il video di output.
            list_patterns (List[int]): Lista di pattern (interi) da usare per la ricostruzione.
            TimeFilter (int): Finestra temporale per l'analisi dei pattern.
            SpaceFilter (int): Dimensione del filtro spaziale applicato.
            frame_rate (Optional[int], optional): FPS del video di output. Se None, usa quello originale.
            max_frame (Optional[int], optional): Numero massimo di frame da processare. Se None, usa tutti i frame.
        """
        self.Patterns_set: set[int] = set(list_patterns)
        self.max_frames: Optional[int] = max_frame
        self.frame_rate: Optional[int] = frame_rate

        if len(self.Patterns_set) == 0:
            print("No patterns found")
            return None

        self.TimeFilter: int = TimeFilter
        self.SpaceFilter: int = SpaceFilter
        self.len_box: int = SpaceFilter * SpaceFilter
        self.path: str = path_out
        self.cap: cv2.VideoCapture = cv2.VideoCapture(path_in)

        if self.SpaceFilter > 9:
            print("SpaceFilter troppo grande per creare MetaData")

        # Inizializza le matrici per memorizzare frame e metadati
        frame_height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_length: int = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if self.max_frames is not None:
            self.max_frames = min(video_length, self.max_frames)
        else :
            self.max_frames = video_length

        self.patterns_for_frame = (frame_height - self.SpaceFilter + 1) * (frame_width - self.SpaceFilter + 1)
        self.frames = deque(maxlen=self.TimeFilter)
        self.pattern_grid = deque(maxlen=self.TimeFilter)

        self.dimensione_Pre_Set = 500000
        self.Pre_Set = np.zeros(self.dimensione_Pre_Set, dtype = np.uint8)
        for _pattern in list_patterns:
            self.Pre_Set[_pattern % self.dimensione_Pre_Set] = 1

    def generate(self, lossless: bool = False, verbose: int = 1) -> float:
        """
        Generate a sketch video from the input video based on selected patterns.

        This method processes the input video, applies spatial and temporal filters, 
        and generates a sketch video that highlights regions matching the specified patterns. 
        The resulting video can be saved in a lossless or compressed format based on the parameter.

        Args:
            lossless (bool, optional): If True, saves the video using a lossless codec (e.g., 'FFV1') 
                                    ensuring no quality loss but larger file size. 
                                    If False, uses a compressed codec (e.g., 'mp4v') for smaller file size.
                                    Defaults to False.
            verbose (int, optional): Controls the level of logging output during processing.

        Returns:
            float: The ratio of accepted patterns to total patterns processed, 
                indicating the density of matched patterns in the final video.

        Raises:
            ValueError: If no patterns are found or if the input video cannot be processed.

        Example:
            >>> sketch = Sketch("input.avi", "output_sketch.avi", patterns_list, TimeFilter=3, SpaceFilter=3)
            >>> ratio = sketch.generate(lossless=True, verbose=1)
            >>> print(f"Pattern Match Ratio: {ratio:.2%}")
        """

        # Controlla se ci sono pattern definiti
        if len(self.Patterns_set) == 0:
            print("No patterns found")
            return

        # Inizializza il video writer per scrivere il video di output
        fourcc = cv2.VideoWriter_fourcc(*'FFV1') if lossless else cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Larghezza frame
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Altezza frame

        # Ottieni il frame rate dal video se non è già definito
        if self.frame_rate is None:
            self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))

        # Crea il video di output in scala di grigi
        out = cv2.VideoWriter(self.path, fourcc, self.frame_rate, (frame_width, frame_height), isColor=False)

        try:
            # Carica i primi 'TimeFilter' frame per inizializzare
            for i in range(self.TimeFilter):
                ret, frame = self.cap.read()
                if not ret:
                    # Se il video ha meno frame del previsto, termina
                    print(f"{i} max frames, TimeFilter must be less than the number of frames in the video")
                    return None

                # Converte il frame in scala di grigi e normalizza i valori
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = (frame // 255).astype(np.uint32)  # Normalizzazione a 0 o 1

                # Inizializza la griglia dei pattern per il frame corrente
                pattern_grid_current = np.zeros((frame.shape[0] - (self.SpaceFilter - 1), 
                                                frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint32)

                # Applica il filtro spaziale scorrendo una finestra sul frame
                for row in range(self.SpaceFilter):
                    for col in range(self.SpaceFilter):
                        shift = (( 2  - row) * 3) + ( 2  - col) # Calcola lo shift
                        # Applica uno shift a sinistra e somma al pattern corrente
                        pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - self.SpaceFilter, 
                                                                        col:frame.shape[1] + col + 1 - self.SpaceFilter], shift)

                # Salva i frame e le griglie dei pattern in una deque
                self.frames.appendleft(frame)
                self.pattern_grid.appendleft(pattern_grid_current)

            # Inizializza la coda dei frame del video sketch
            sketch_frames = deque(maxlen=self.TimeFilter)
            filled_frame = np.full(frame.shape, 128, dtype=np.uint8)  # Frame grigio neutro
            for _ in range(self.TimeFilter):
                sketch_frames.appendleft(filled_frame.copy())

            # Variabili per tracciare il conteggio dei pattern
            frame_count = 0
            pattern_totali = 0
            pattern_accettati = 0

            # Loop principale per processare il video
            while True:
                ret, frame = self.cap.read()
                if verbose > 1 and frame_count % 10 == 0:
                    print("Frame: ", frame_count)  # Messaggio ogni 10 frame

                # Termina se non ci sono più frame o si raggiunge il limite massimo
                if not ret or (self.max_frames is not None and frame_count >= self.max_frames):
                    break

                # Converte il frame in scala di grigi e normalizza
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = (frame // 255).astype(np.uint32)

                # Crea la griglia dei pattern per il frame corrente
                pattern_grid_current = np.zeros((frame.shape[0] - (self.SpaceFilter - 1), 
                                                frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint32)

                # Applica il filtro spaziale al nuovo frame
                for row in range(self.SpaceFilter):
                    for col in range(self.SpaceFilter):
                        shift = (( 2  - row) * 3) + ( 2  - col)
                        pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - self.SpaceFilter, 
                                                                        col:frame.shape[1] + col + 1 - self.SpaceFilter], shift, dtype=np.uint16)

                # Scrive il frame più vecchio nel video di output
                out.write(sketch_frames[-1])

                # Aggiorna le code con il nuovo frame e la sua griglia di pattern
                self.pattern_grid.appendleft(pattern_grid_current)
                self.frames.appendleft(frame)
                sketch_frames.appendleft(filled_frame.copy())

                # Combina i pattern nel box considerando il filtro temporale
                box = np.zeros(self.pattern_grid[0].shape, dtype=np.uint32)
                for t in range(self.TimeFilter):
                    box += np.left_shift(self.pattern_grid[t], t * self.len_box, dtype=np.uint32)

                # Aggiorna i contatori dei pattern
                pattern_totali += self.patterns_for_frame

                # Trova i pattern che corrispondono a quelli predefiniti
                rows, cols = np.where(self.Pre_Set[box % self.dimensione_Pre_Set])
                for i, j in zip(rows, cols):
                    if box[i, j] in self.Patterns_set:
                        pattern_accettati += 1
                        # Copia i pixel del pattern nei frame dello sketch
                        for t in range(self.TimeFilter):
                            sketch_frames[t][i:i+self.SpaceFilter, j:j+self.SpaceFilter] = self.frames[t][i:i+self.SpaceFilter, j:j+self.SpaceFilter]

                frame_count += 1

            # Scrive i frame rimanenti nello sketch finale
            for fr in reversed(sketch_frames):
                out.write(fr)

            ratio = pattern_accettati / pattern_totali
            if verbose > 0:
                print("-------------------")
                print(f"Elaborazione completata, video salvato in : {self.path}.\n {frame_count} frame processati, banda : {ratio:.3}.")
            # Restituisce il rapporto tra pattern accettati e totali
            return ratio

        except:
            self.cap.release()
            out.release()
            raise "Errore"
            
        finally:
            # Rilascia le risorse video
            self.cap.release()
            out.release()

    def misura_banda(self, verbose : int = 1) -> float:
        """
        Generate a sketch video from the input video based on selected patterns.

        This method processes the input video, applies spatial and temporal filters, 
        and generates a sketch video that highlights regions matching the specified patterns. 
        The resulting video can be saved in a lossless or compressed format based on the parameter.

        Args:
            lossless (bool, optional): If True, saves the video using a lossless codec (e.g., 'FFV1') 
                                    ensuring no quality loss but larger file size. 
                                    If False, uses a compressed codec (e.g., 'mp4v') for smaller file size.
                                    Defaults to False.
            verbose (int, optional): Controls the level of logging output during processing.

        Returns:
            float: The ratio of accepted patterns to total patterns processed, 
                indicating the density of matched patterns in the final video.

        Raises:
            ValueError: If no patterns are found or if the input video cannot be processed.

        Example:
            >>> sketch = Sketch("input.avi", "output_sketch.avi", patterns_list, TimeFilter=3, SpaceFilter=3)
            >>> ratio = sketch.generate(lossless=True, verbose=1)
            >>> print(f"Pattern Match Ratio: {ratio:.2%}")
        """

        # Controlla se ci sono pattern definiti
        if len(self.Patterns_set) == 0:
            print("No patterns found")
            return

        try:
            # Carica i primi 'TimeFilter' frame per inizializzare
            for i in range(self.TimeFilter):
                ret, frame = self.cap.read()
                if not ret:
                    # Se il video ha meno frame del previsto, termina
                    print(f"{i} max frames, TimeFilter must be less than the number of frames in the video")
                    return None

                # Converte il frame in scala di grigi e normalizza i valori
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = (frame // 255).astype(np.uint32)  # Normalizzazione a 0 o 1

                # Inizializza la griglia dei pattern per il frame corrente
                pattern_grid_current = np.zeros((frame.shape[0] - (self.SpaceFilter - 1), 
                                                frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint32)

                # Applica il filtro spaziale scorrendo una finestra sul frame
                for row in range(self.SpaceFilter):
                    for col in range(self.SpaceFilter):
                        shift = (( 2  - row) * 3) + ( 2  - col) # Calcola lo shift
                        # Applica uno shift a sinistra e somma al pattern corrente
                        pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - self.SpaceFilter, 
                                                                        col:frame.shape[1] + col + 1 - self.SpaceFilter], shift)

                self.pattern_grid.appendleft(pattern_grid_current)


            # Variabili per tracciare il conteggio dei pattern
            frame_count = 0
            pattern_totali = 0
            pattern_accettati = 0

            # Loop principale per processare il video
            while True:
                ret, frame = self.cap.read()
                if verbose > 1 and frame_count % 10 == 0:
                    print("Frame: ", frame_count)  # Messaggio ogni 10 frame

                # Termina se non ci sono più frame o si raggiunge il limite massimo
                if not ret or (self.max_frames is not None and frame_count >= self.max_frames):
                    break

                # Converte il frame in scala di grigi e normalizza
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = (frame // 255).astype(np.uint32)

                # Crea la griglia dei pattern per il frame corrente
                pattern_grid_current = np.zeros((frame.shape[0] - (self.SpaceFilter - 1), 
                                                frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint32)

                # Applica il filtro spaziale al nuovo frame
                for row in range(self.SpaceFilter):
                    for col in range(self.SpaceFilter):
                        shift = (( 2  - row) * 3) + ( 2  - col)
                        pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - self.SpaceFilter, 
                                                                        col:frame.shape[1] + col + 1 - self.SpaceFilter], shift, dtype=np.uint16)


                # Aggiorna le code con il nuovo frame e la sua griglia di pattern
                self.pattern_grid.appendleft(pattern_grid_current)

                # Combina i pattern nel box considerando il filtro temporale
                box = np.zeros(self.pattern_grid[0].shape, dtype=np.uint32)
                for t in range(self.TimeFilter):
                    box += np.left_shift(self.pattern_grid[t], t * self.len_box, dtype=np.uint32)

                # Aggiorna i contatori dei pattern
                pattern_totali += self.patterns_for_frame

                # Trova i pattern che corrispondono a quelli predefiniti
                rows, cols = np.where(self.Pre_Set[box % self.dimensione_Pre_Set])
                for i, j in zip(rows, cols):
                    if box[i, j] in self.Patterns_set:
                        pattern_accettati += 1
                frame_count += 1
                break

            
            ratio = pattern_accettati / pattern_totali
            if verbose > 0:
                print("-------------------")
                print(f"Elaborazione completata, video salvato in : {self.path}.\n {frame_count} frame processati, banda : {ratio:.3}.")
            # Restituisce il rapporto tra pattern accettati e totali
            return ratio

        finally:
            # Rilascia le risorse video
            self.cap.release()


    def togli_pattern_buoni(self, lossless: bool = False, verbose: int = 1) -> float:
            """
            Generate a sketch video from the input video based on selected patterns.

            This method processes the input video, applies spatial and temporal filters, 
            and generates a sketch video that highlights regions matching the specified patterns. 
            The resulting video can be saved in a lossless or compressed format based on the parameter.

            Args:
                lossless (bool, optional): If True, saves the video using a lossless codec (e.g., 'FFV1') 
                                        ensuring no quality loss but larger file size. 
                                        If False, uses a compressed codec (e.g., 'mp4v') for smaller file size.
                                        Defaults to False.
                verbose (int, optional): Controls the level of logging output during processing.

            Returns:
                float: The ratio of accepted patterns to total patterns processed, 
                    indicating the density of matched patterns in the final video.

            Raises:
                ValueError: If no patterns are found or if the input video cannot be processed.

            Example:
                >>> sketch = Sketch("input.avi", "output_sketch.avi", patterns_list, TimeFilter=3, SpaceFilter=3)
                >>> ratio = sketch.generate(lossless=True, verbose=1)
                >>> print(f"Pattern Match Ratio: {ratio:.2%}")
            """

            # Controlla se ci sono pattern definiti
            if len(self.Patterns_set) == 0:
                print("No patterns found")
                return

            # Inizializza il video writer per scrivere il video di output
            fourcc = cv2.VideoWriter_fourcc(*'FFV1') if lossless else cv2.VideoWriter_fourcc(*'mp4v')
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # Larghezza frame
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Altezza frame

            # Ottieni il frame rate dal video se non è già definito
            if self.frame_rate is None:
                self.frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))

            # Crea il video di output in scala di grigi
            out = cv2.VideoWriter(self.path, fourcc, self.frame_rate, (frame_width, frame_height), isColor=False)
            maskera_patten_buoni = deque(maxlen=self.TimeFilter)
            sketch_frames = deque(maxlen=self.TimeFilter)

            try:
                # Carica i primi 'TimeFilter' frame per inizializzare
                for i in range(self.TimeFilter):
                    ret, frame = self.cap.read()
                    if not ret:
                        # Se il video ha meno frame del previsto, termina
                        print(f"{i} max frames, TimeFilter must be less than the number of frames in the video")
                        return None

                    # Converte il frame in scala di grigi e normalizza i valori
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mask = frame == 128
                    sketch_frames.appendleft(frame.copy())
                    gray_frame = (frame // 255).astype(np.uint32)  # Normalizzazione a 0 o 1

                    # Inizializza la griglia dei pattern per il frame corrente
                    pattern_grid_current = np.zeros((frame.shape[0] - (self.SpaceFilter - 1), 
                                                    frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint32)
                    shifted_mask = np.zeros((frame.shape[0] - (self.SpaceFilter - 1),
                                            frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint8)
                    # Applica il filtro spaziale scorrendo una finestra sul frame
                    for row in range(self.SpaceFilter):
                        for col in range(self.SpaceFilter):
                            shift = (row * self.SpaceFilter + col)  # Calcola lo shift
                            # Applica uno shift a sinistra e somma al pattern corrente
                            shifted_mask += mask[row:frame.shape[0] + row + 1 - self.SpaceFilter, col:frame.shape[1] + col + 1 - self.SpaceFilter]
                            pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - self.SpaceFilter, 
                                                                            col:frame.shape[1] + col + 1 - self.SpaceFilter], shift)

                    # Salva i frame e le griglie dei pattern in una deque
                    self.frames.appendleft(frame)
                    maskera_patten_buoni.appendleft(shifted_mask)
                    self.pattern_grid.appendleft(pattern_grid_current)


                # Variabili per tracciare il conteggio dei pattern
                frame_count = 0
                pattern_totali = 0
                pattern_accettati = 0

                # Loop principale per processare il video
                while True:
                    ret, frame = self.cap.read()
                    if verbose > 1 and frame_count % 10 == 0:
                        print("Frame: ", frame_count)  # Messaggio ogni 10 frame

                    # Termina se non ci sono più frame o si raggiunge il limite massimo
                    if not ret or (self.max_frames is not None and frame_count >= self.max_frames):
                        break

                    # Converte il frame in scala di grigi e normalizza
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    mask = frame == 128
                    gray_frame = (frame // 255).astype(np.uint8)  # Normalizzazione a 0 o 1

                    # Inizializza la griglia dei pattern per il frame corrente
                    pattern_grid_current = np.zeros((frame.shape[0] - (self.SpaceFilter - 1), 
                                                    frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint32)
                    shifted_mask = np.zeros((frame.shape[0] - (self.SpaceFilter - 1),
                                            frame.shape[1] - (self.SpaceFilter - 1)), dtype=np.uint8)

                    # Applica il filtro spaziale al nuovo frame
                    for row in range(self.SpaceFilter):
                        for col in range(self.SpaceFilter):
                            shifted_mask += mask[row:frame.shape[0] + row + 1 - self.SpaceFilter, col:frame.shape[1] + col + 1 - self.SpaceFilter]
                            shift = (row * self.SpaceFilter + col)
                            pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - self.SpaceFilter, 
                                                                            col:frame.shape[1] + col + 1 - self.SpaceFilter], shift, dtype=np.uint16)

                    # Scrive il frame più vecchio nel video di output
                    out.write(sketch_frames[-1])

                    # Aggiorna le code con il nuovo frame e la sua griglia di pattern
                    self.pattern_grid.appendleft(pattern_grid_current)
                    self.frames.appendleft(frame)
                    maskera_patten_buoni.appendleft(shifted_mask)
                    sketch_frames.appendleft(frame.copy())

                    # Combina i pattern nel box considerando il filtro temporale
                    box = np.zeros(self.pattern_grid[0].shape, dtype=np.uint32)
                    for t in range(self.TimeFilter):
                        box += np.left_shift(self.pattern_grid[self.TimeFilter - 1 - t], t * self.len_box, dtype=np.uint32)

                    # Aggiorna i contatori dei pattern
                    pattern_totali += self.patterns_for_frame

                    gray_box = np.ones((self.SpaceFilter,self.SpaceFilter), dtype = np.uint8) * 128
                    # Trova i pattern che corrispondono a quelli predefiniti
                    rows, cols = np.where(self.Pre_Set[box % self.dimensione_Pre_Set])
                    for i, j in zip(rows, cols):
                        flag = any(maskera_patten_buoni[t][i, j]  for t in range(self.TimeFilter))
                        if not flag and box[i, j] in self.Patterns_set:
                            pattern_accettati += 1
                            # Copia i pixel del pattern nei frame dello sketch
                            for t in range(self.TimeFilter):
                                sketch_frames[t][i:i+self.SpaceFilter, j:j+self.SpaceFilter] = gray_box

                    frame_count += 1

                # Scrive i frame rimanenti nello sketch finale
                for fr in reversed(sketch_frames):
                    out.write(fr)
                
                ratio = pattern_accettati / pattern_totali
                if verbose > 0:
                    print("-------------------")
                    print(f"Elaborazione completata, video salvato in : {self.path}.\n {frame_count} frame processati, banda : {ratio:.3}.")
                # Restituisce il rapporto tra pattern accettati e totali
                return ratio

            finally:
                # Rilascia le risorse video
                self.cap.release()
                out.release()







def split_video_by_camera_cut(input_video_path: Path, output_directory: Path, difference_threshold: float, lossless: bool = True, verbose : int = 1) -> None:
    """
    Funzione per suddividere un video in più file, in base al cambio di inquadratura.
    Un nuovo file video viene creato quando la differenza media tra il frame corrente e quello precedente 
    supera una soglia definita dall'utente.

    Args:
        input_video_path (Path): Percorso del video di input da elaborare.
        output_directory (Path): Percorso della directory dove salvare i video suddivisi.
        difference_threshold (float): Soglia di differenza media tra i frame per decidere se creare un nuovo video.
        lossless (bool, optional): Se True, utilizza il codec FFV1 per la compressione senza perdita. Default: True.
        verbose (int, optional): Livello di dettaglio dei messaggi di output. Default: 1.

    Returns:
        None: La funzione salva i video separati nella directory di output.
    """

    # Apre il video di input
    video_capture = cv2.VideoCapture(str(input_video_path))
    
    if not video_capture.isOpened():
        print("Errore nell'aprire il file video.")
        return

    # Variabili di inizializzazione
    file_number = 0  # Numero del file di output
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') if lossless else cv2.VideoWriter_fourcc(*'mp4v')  # Codec per il file di output
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))  # Frame per secondo del video di input
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))  # Larghezza del frame
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Altezza del frame

    # Crea il primo file di output
    output_file_path = output_directory / f"{input_video_path.stem}_SPLIT_{file_number}.avi"
    video_writer = cv2.VideoWriter(str(output_file_path), fourcc, fps, (frame_width, frame_height), isColor=False)

    # Legge il primo frame del video per iniziare l'elaborazione
    while True:
        ret, previous_frame = video_capture.read()
        if not ret:
            video_capture.release()
            video_writer.release()
            return
        previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        if np.any(previous_frame):  # Assicurati che il frame non sia vuoto
            break

    frame_count = 0  # Conta il numero di frame elaborati nel video corrente

    # Inizia a leggere e scrivere i frame nel video di output
    while True:
        ret, current_frame = video_capture.read()
        if not ret:
            video_writer.release()  # Rilascia il video writer quando non ci sono più frame
            break
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Calcola la differenza media tra il frame corrente e quello precedente
        if np.mean(current_frame != previous_frame) > difference_threshold:
            # Se la differenza supera la soglia, crea un nuovo file video
            video_writer.release()
            if frame_count > 10:  # Evita di creare file troppo piccoli
                file_number += 1
            output_file_path = output_directory / f"{input_video_path.stem}_SPLIT_{file_number}.avi"
            frame_count = 0
            if verbose > 1:
                print(f"Creazione file video: {output_file_path.name}")
            video_writer = cv2.VideoWriter(str(output_file_path), fourcc, fps, (frame_width, frame_height), isColor=False)

        # Scrive il frame corrente nel file di output
        video_writer.write(current_frame)
        previous_frame = current_frame.copy()
        frame_count += 1  # Incrementa il contatore dei frame

    # Rilascia tutte le risorse
    video_capture.release()
    video_writer.release()
    if verbose > 0:
        print(f"Elaborazione completata, tutte le risorse sono state rilasciate.\n {file_number} file video creati nella directory : {output_directory}.")

def get_video_format(video_path: Path) -> tuple:
    """
    Ottiene le informazioni sul formato del video (width, height, fps, frame_count).

    Args:
        video_path (Path): Percorso del file video da analizzare.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Errore nell'aprire il file video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    media_info = MediaInfo.parse(video_path)


    for track in media_info.tracks:
        if track.track_type == "Video":
            return width, height, fps, frame_count, track.bit_rate, track.maximum_bit_rate
        
    return width, height, fps, frame_count, None, None

def binary_video_converter(input_path: str, output_path: str, output_width: int = None, output_height: int = None, init_frame: int = 0, max_frame: int = -1, lossless : bool = True, verbose : int = 1, adaptive_tresh : bool = False, dim_blocco : int = None, c : int = None) -> bool:
    """
    Processa un video convertendolo in una versione binarizzata (bianco/nero) basata sulla soglia della mediana dei pixel.

    Il video viene ridimensionato se specificato e convertito in scala di grigi. Ogni frame viene quindi binarizzato 
    utilizzando la mediana come soglia. Il risultato viene salvato in un nuovo file video.

    Args:
        input_path (str): Percorso del file video di input.
        output_path (str): Percorso del file video di output.
        output_width (int, optional): Larghezza desiderata del video di output. Se None, mantiene la larghezza originale.
        output_height (int, optional): Altezza desiderata del video di output. Se None, mantiene l'altezza originale.
        init_frame (int, optional): Numero di frame iniziali da saltare. Default è 0.
        max_frame (int, optional): Numero massimo di frame da processare. Se -1, processa tutti i frame disponibili.
        lossless (bool, optional): Se True, utilizza il codec FFV1 per la compressione senza perdita. Default: True.
        verbose (int, optional): Livello di dettaglio dei messaggi di output. Default: 1.

    Returns:
        bool: 
            - True se il video è stato processato con successo.
            - False in caso di errore o se il video non è valido (es. orientamento verticale).

    Raises:
        SystemExit: Se il file di input non esiste o non può essere aperto.
    """

    # Verifica che il file di input esista
    if not os.path.exists(input_path):
        print(f"Errore: il file '{input_path}' non esiste.")
        sys.exit(1)

    # Apertura del video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Errore: impossibile aprire il video '{input_path}'.")
        sys.exit(1)

    # Ottieni proprietà del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {input_path}")
    print(f"Risoluzione: {width}x{height}, FPS: {fps}, Frame totali: {frame_count}")

    # Verifica se l'output esiste già
    if os.path.exists(output_path):
        print(f"Attenzione: il file '{output_path}' esiste già. Verrà sovrascritto.")

    # Esclude video in orientamento verticale
    if height > width:
        print("Errore: il video ha orientamento verticale. Operazione annullata.")
        return False

    # Imposta dimensioni di output se non specificate
    if output_width is None:
        output_width = width
    if output_height is None:
        output_height = height

    # Configura il video writer per il file di output (lossless se usato FFV1)
    fourcc = cv2.VideoWriter_fourcc(*'FFV1') if lossless else cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height), isColor=False)

    # Salta i frame iniziali se richiesto
    for _ in range(init_frame):
        ret, _ = cap.read()
        if not ret:
            print("Errore: raggiunto fine video durante il salto dei frame iniziali.")
            return False

    frame_index = 0  # Contatore dei frame processati

    # Inizio ciclo di lettura e processamento dei frame
    while True:
        ret, frame = cap.read()
        if not ret or (0 <= max_frame == frame_index):
            break  # Fine del video o raggiunto limite di frame

        resized_frame = frame
        # Ridimensiona il frame alle dimensioni desiderate
        # frame = cv2.GaussianBlur(frame,(5,5),0)
        resized_frame =  frame.reshape(frame.shape[0]//2, 2, frame.shape[1]//2, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
        resized_frame = cv2.resize(frame, (output_width, output_height))

        # Converti in scala di grigi
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)

        binary_frame = gray_frame

        if adaptive_tresh:

            binary_frame = cv2.adaptiveThreshold(
                gray_frame, 255,  # Valore massimo (bianco)
                cv2.ADAPTIVE_THRESH_MEAN_C,  # Metodo: Media o Gaussiana
                cv2.THRESH_BINARY,  # Binarizzazione normale
                dim_blocco,  # Dimensione del blocco (deve essere dispari)
                c  # Costante sottratta alla soglia locale (aiuta con il contrasto)
            )
            
            # _,binary_frame = cv2.threshold(blur,median_value,255, cv2.THRESH_BINARY)
        else:
            # Calcola la mediana del frame per usarla come soglia
            median_value = np.median(gray_frame)

            # Applica la soglia binaria usando la mediana
            _, binary_frame = cv2.threshold(gray_frame, median_value, 255, cv2.THRESH_BINARY)

        #  Scrivi il frame processato nel file di output
        out.write(binary_frame )

        frame_index += 1

        # Aggiorna lo stato ogni 10 frame
        if verbose > 1 and frame_index % 100 == 0:
            print(f"Processati {frame_index}/{frame_count} frame...")

    # Rilascia le risorse
    cap.release()
    out.release()
    if verbose > 0:
        print(f"Video processato salvato in: {output_path}")
    return True



def ContaPatterns(path_in: str,  
                list_patterns: List[int], 
                TimeFilter: int, 
                SpaceFilter: int, 
                max_frame: Optional[int] = None,
                init_frame: Optional[int] = 0) -> None:
    """
    Inizializza la classe Sketch per la ricostruzione di video basata su pattern selezionati.

    Args:
        path_in (str): Percorso del video di input.
        path_out (str): Percorso per salvare il video di output.
        list_patterns (List[int]): Lista di pattern (interi) da usare per la ricostruzione.
        TimeFilter (int): Finestra temporale per l'analisi dei pattern.
        SpaceFilter (int): Dimensione del filtro spaziale applicato.
        frame_rate (Optional[int], optional): FPS del video di output. Se None, usa quello originale.
        max_frame (Optional[int], optional): Numero massimo di frame da processare. Se None, usa tutti i frame.
    """
    Pattern_dic = { _pattern : i for i, _pattern in enumerate(list_patterns)}
    array_patterns = np.zeros(len(list_patterns), dtype = np.uint64)
    max_frames: Optional[int] = max_frame

    if len(Pattern_dic) == 0:
        print("No patterns found")
        return None

    TimeFilter: int = TimeFilter
    SpaceFilter: int = SpaceFilter
    len_box: int = SpaceFilter * SpaceFilter
    cap: cv2.VideoCapture = cv2.VideoCapture(path_in)

    if SpaceFilter > 9:
        print("SpaceFilter troppo grande per creare MetaData")

    # Inizializza le matrici per memorizzare frame e metadati
    frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_length: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if max_frames is not None:
        max_frames = min(video_length, max_frames)
    else :
        max_frames = video_length

    patterns_for_frame = (frame_height - SpaceFilter + 1) * (frame_width - SpaceFilter + 1)
    pattern_grid = deque(maxlen=TimeFilter)

    dimensione_Pre_Set = 500000
    Pre_Set = np.zeros(dimensione_Pre_Set, dtype = np.uint8)
    for _pattern in list_patterns:
        Pre_Set[_pattern % dimensione_Pre_Set] = 1


    try:

        # Carica i primi 'TimeFilter' frame per inizializzare
        for i in range(max(TimeFilter, init_frame)):
            ret, frame = cap.read()
            if not ret:
                # Se il video ha meno frame del previsto, termina
                print(f"{i} max frames, TimeFilter must be less than the number of frames in the video")
                return None

            # Converte il frame in scala di grigi e normalizza i valori
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = (frame // 255).astype(np.uint32)  # Normalizzazione a 0 o 1

            # Inizializza la griglia dei pattern per il frame corrente
            pattern_grid_current = np.zeros((frame.shape[0] - (SpaceFilter - 1), 
                                            frame.shape[1] - (SpaceFilter - 1)), dtype=np.uint32)

            # Applica il filtro spaziale scorrendo una finestra sul frame
            for row in range(SpaceFilter):
                for col in range(SpaceFilter):
                    shift = (( 2  - row) * 3) + ( 2  - col) # Calcola lo shift
                    # Applica uno shift a sinistra e somma al pattern corrente
                    pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - SpaceFilter, 
                                                                    col:frame.shape[1] + col + 1 - SpaceFilter], shift)

            pattern_grid.appendleft(pattern_grid_current)


        # Variabili per tracciare il conteggio dei pattern
        frame_count = 0
        pattern_totali = 0

        # Loop principale per processare il video
        while True:
            ret, frame = cap.read()

            # Termina se non ci sono più frame o si raggiunge il limite massimo
            if not ret or (max_frames is not None and frame_count >= max_frames):
                break

            # Converte il frame in scala di grigi e normalizza
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = (frame // 255).astype(np.uint32)

            # Crea la griglia dei pattern per il frame corrente
            pattern_grid_current = np.zeros((frame.shape[0] - (SpaceFilter - 1), 
                                            frame.shape[1] - (SpaceFilter - 1)), dtype=np.uint32)

            # Applica il filtro spaziale al nuovo frame
            for row in range(SpaceFilter):
                for col in range(SpaceFilter):
                    shift = (( 2  - row) * 3) + ( 2  - col)
                    pattern_grid_current += np.left_shift(gray_frame[row:frame.shape[0] + row + 1 - SpaceFilter, 
                                                                    col:frame.shape[1] + col + 1 - SpaceFilter], shift, dtype=np.uint16)

            # Aggiorna le code con il nuovo frame e la sua griglia di pattern
            pattern_grid.appendleft(pattern_grid_current)

            # Combina i pattern nel box considerando il filtro temporale
            box = np.zeros(pattern_grid[0].shape, dtype=np.uint32)
            for t in range(TimeFilter):
                box += np.left_shift(pattern_grid[t], t * len_box, dtype=np.uint32)

            # Aggiorna i contatori dei pattern
            pattern_totali += patterns_for_frame

            # Trova i pattern che corrispondono a quelli predefiniti
            rows, cols = np.where(Pre_Set[box % dimensione_Pre_Set])
            for i, j in zip(rows, cols):
                if box[i, j] in Pattern_dic:
                    # Copia i pixel del pattern nei frame dello sketch
                    array_patterns[Pattern_dic[box[i, j]]] += 1

            frame_count += 1

        # Restituisce il rapporto tra pattern accettati e totali
        return array_patterns/pattern_totali
    finally:
        # Rilascia le risorse video
        cap.release()