# N-Suite: test di tutti i nodi

Apri `N-Suite-all-nodes-test.json` in ComfyUI. Il workflow contiene tutti i 14 tipi di nodo N-Suite presenti in questa versione, con anteprime dei risultati e tre prove di salvataggio video.

Prima di premere **Queue Prompt**:

1. Scegli una tua immagine nel nodo **LoadImage** della sezione 01. Il loader GPT usa Moondream, già selezionato nel workflow.
2. Copia un MP4 breve nella cartella `ComfyUI/input/n-suite`, ricarica la pagina e selezionalo nel nodo **LoadVideo** della sezione 04. Un video di pochi secondi riduce il tempo necessario per RIFE.
3. Metti almeno due immagini PNG della stessa dimensione, con nomi numerati come `0001.png` e `0002.png`, nella cartella `ComfyUI/input/n-suite/test_frames`. Il nodo **String Variable** della sezione 05 contiene il percorso visto dal container: `/workspace/ComfyUI/input/n-suite/test_frames`.
4. Premi **Queue Prompt**. Il nodo CLIP usa `clip_l.safetensors`, già disponibile nell'installazione per cui è stato creato il workflow.

La risposta Moondream e i condizionamenti CLIP compaiono nei nodi **Preview as Text**. Le immagini e la maschera compaiono nelle anteprime. I video vengono scritti in `ComfyUI/output/n-suite/videos` con prefissi `n_suite_test_*`. Se un ramo fallisce, ComfyUI evidenzia il nodo che ha generato l'errore.

Il file è generato da `generate_test_workflow.py` usando gli schemi `/object_info` di ComfyUI. Su un'installazione diversa, rigeneralo con `python examples/generate_test_workflow.py http://127.0.0.1:8188` dalla cartella del repository.
