# Video Sketcher

Tool per l’analisi e la ricostruzione video basata su pattern frequenti.  
Contiene codice per la selezione euristica, visualizzazione e generazione di video "sketch".

## Struttura

- `src/data_utils.py` → importazione e mascheratura dati
- `src/video_utils.py` → funzioni per elaborare e ricostruire video
- `src/plotting_utils.py` → grafici e visualizzazioni
- `src/scarta_video.py` → sistema di classificazione interattivo dei video

## Requisiti

```bash
pip install -r requirements.txt