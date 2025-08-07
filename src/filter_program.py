import cv2
import pandas as pd
from pathlib import Path

class Scarta:
    def __init__(self, path_video: Path, directory_dataframe: Path):
        if not path_video.exists() or not path_video.is_dir():
            raise FileNotFoundError(f"La cartella {path_video} non esiste o non è una directory.")

        self.path_video = path_video
        self.dataframe_path = directory_dataframe / "scarta.csv"

        if self.dataframe_path.exists():
            self.df = pd.read_csv(self.dataframe_path)
        else:
            self.df = self._crea_dataframe(path_video)
            self.salva_dataframe()

    def _crea_dataframe(self, path_video: Path):
        files = [file.name for file in path_video.iterdir() if file.is_file()]
        df = pd.DataFrame({'nome_file': files, 'scartare': 0})
        return df

    def salva_dataframe(self):
        self.df.set_index("nome_file", inplace=True)
        self.df.to_csv(self.dataframe_path)
        self.df.reset_index(inplace=True)

    def riproduci_video(self, video_path):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Impossibile aprire il video {video_path}")
            return None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Fine del video

            cv2.imshow("Video", frame)
            key = cv2.waitKey(15)  # 30ms per frame ≈ 33 FPS
            if key == 113:  # Q per uscire
                cap.release()
                cv2.destroyAllWindows()
                return "exit"
            if key == 8:  # backspace per indietro
                cap.release()
                cv2.destroyAllWindows()
                return "indietro"
            elif key == 13:  # ENTER → Segna come scartato
                cap.release()
                cv2.destroyAllWindows()
                return "scartare"
            elif key == 32:  # SPACE → Segna come salvato
                cap.release()
                cv2.destroyAllWindows()
                return "salvare"

        cap.release()
        cv2.destroyAllWindows()
        return "da_capo"

    def run(self, elementi=100):
        to_check = self.df[self.df["scartare"] == 0
                           ].head(elementi)
        indici = to_check.index.tolist()
        if to_check.empty:
            print("Nessun video da valutare.")
            return
        indice = 0
        while indice < len(indici):
            idx = indici[indice]
            video_path = self.path_video / to_check.loc[idx, "nome_file"]
            
            result = self.riproduci_video(video_path)
            if result == "exit":
                break
            elif result == "scartare":
                self.df.at[idx, "scartare"] = 1
            elif result == "salvare":

                self.df.at[idx, "scartare"] = -1
            elif result == "da_capo":
                continue
            elif result == "indietro":
        
                indice -= 2

            print(idx)
            
            indice += 1


        self.salva_dataframe()
        print("Aggiornamento completato.")

df_path = Path(r"C:\Users\braua\Documents\TesiMagistrale\Dataset")
path_video = Path(r"C:\Users\braua\Downloads\Inter4K\60fps\UHD")
scarta = Scarta(path_video, df_path)
scarta.run(500)
 
