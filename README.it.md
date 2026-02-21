<div align="center">

# ComfyUI Rife TensorRT ⚡

[![python](https://img.shields.io/badge/python-3.12-green)](https://www.python.org/downloads/)
[![cuda](https://img.shields.io/badge/cuda-13.0-green)](https://developer.nvidia.com/cuda-13-0-2-download-archive)
[![trt](https://img.shields.io/badge/TRT-10.15.1.29-green)](https://developer.nvidia.com/tensorrt)
[![by-nc-sa/4.0](https://img.shields.io/badge/license-CC--BY--NC--SA--4.0-lightgrey)](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en)
[![english](https://img.shields.io/badge/lang-🇬🇧%20english-blue)](README.md)

![node](https://github.com/user-attachments/assets/5fd6d529-300c-42a5-b9cf-46e031f0bcb5)

</div>

Questo progetto fornisce un'implementazione [TensorRT](https://github.com/NVIDIA/TensorRT) di [RIFE](https://github.com/hzwer/ECCV2022-RIFE) per l'interpolazione di frame ultra veloce all'interno di ComfyUI

Questo progetto è licenziato sotto [CC BY-NC-SA](https://creativecommons.org/licenses/by-nc-sa/4.0/), tutti sono LIBERI di accedere, usare, modificare e ridistribuire con la stessa licenza.

Se ti piace il progetto, lascia una stella! ⭐

---

## ⏱️ Performance

_Nota: I seguenti risultati sono stati testati su motori FP16 all'interno di ComfyUI, utilizzando 2000 frame composti da 2 frame alternati simili, mediati 2-3 volte_

| Dispositivo | Motore Rife | Risoluzione| Moltiplicatore | FPS |
| :----: | :-: | :-: | :-: | :-: |
|  H100  | rife49_ensemble_True_scale_1_sim | 512 x 512  | 2 | 45 |
|  H100  | rife49_ensemble_True_scale_1_sim | 512 x 512  | 4 | 57 |
|  H100  | rife49_ensemble_True_scale_1_sim | 1280 x 1280  | 2 | 21 |

## 🚀 Installazione

Naviga nella directory `/custom_nodes` di ComfyUI

```bash
git clone https://github.com/huchukato/ComfyUI-RIFE-TensorRT-Auto
cd ./ComfyUI-RIFE-TensorRT-Auto
```

### 🎯 Installazione Completamente Automatica (Raccomandata)

**Questo nodo presenta rilevamento automatico CUDA e installazione TensorRT!**

Quando ComfyUI carica il nodo per la prima volta, eseguirà:
1. **Rilevamento automatico della versione CUDA** (12 o 13)
2. **Installazione automatica dei pacchetti TensorRT appropriati**
3. **Configurazione di tutto** per un funzionamento senza interruzioni

**Nessun passaggio manuale richiesto!** Basta clonare la repo e riavviare ComfyUI.

### 📦 Opzioni di Installazione Manuale

Se preferisci l'installazione manuale o riscontri problemi:

**Script di auto-installazione:**
```bash
# Linux/macOS
./install.sh

# Windows  
install.bat

# Python (cross-platform)
python install.py
```

**File requirements manuali:**
```bash
# Per CUDA 13 (RTX 50 series)
pip install -r requirements.txt

# Per CUDA 12 (RTX 30/40 series) - METODO LEGACY
pip install -r requirements_cu12.txt
```

> **💡 Nota:** Il `requirements_cu12.txt` è fornito come metodo legacy di fallback. L'installazione automatica è fortemente raccomandata in quanto gestisce il rilevamento CUDA e l'installazione dei pacchetti senza problemi.

### 📦 CUDA Toolkit Richiesto

Il nodo rileva automaticamente la tua installazione CUDA tramite le variabili d'ambiente `CUDA_PATH` o `CUDA_HOME`.

Se CUDA non viene rilevato, scarica da: https://developer.nvidia.com/cuda-13-0-2-download-archive

### 🎯 Limiti di Risoluzione

**Importante**: Il motore TensorRT supporta diversi range di risoluzione in base al profilo selezionato:
- **profilo small**: 384-1080px 
- **profilo medium**: 672-1312px
- **profilo large**: 720-1920px (perfetto per 1440x960 e risoluzioni superiori)

Per immagini più grandi del massimo del profilo selezionato, ridimensionale prima di usare il nodo RIFE o seleziona un profilo superiore.

### 🎯 Profili di Risoluzione

Il nodo supporta profili di risoluzione per ottimizzare l'uso della VRAM:
- **small**: 384-1080px (raccomandato per la maggior parte della generazione video)
- **medium**: 672-1312px (per video ad alta risoluzione)
- **large**: 720-1920px (per contenuti 4K e ad alta risoluzione)
- **custom**: Collega un nodo "RIFE Custom Resolution Config" per controllo manuale

I seguenti modelli RIFE sono supportati e verranno scaricati e compilati automaticamente:
   - **rife49_ensemble_True_scale_1_sim** (default) - Il più recente e accurato
   - **rife48_ensemble_True_scale_1_sim** - Buon equilibrio tra velocità e qualità
   - **rife47_ensemble_True_scale_1_sim** - Opzione più veloce

I modelli vengono scaricati automaticamente da [HuggingFace](https://huggingface.co/yuvraj108c/rife-onnx) e i motori TensorRT vengono compilati al primo utilizzo.

## ☀️ Utilizzo

1. **Carica Modello**: Inserisci `Right Click -> Add Node -> tensorrt -> Load Rife Tensorrt Model`
   - Scegli il tuo modello RIFE preferito (rife47, rife48, o rife49)
   - Seleziona la precisione (fp16 raccomandato per velocità, fp32 per massima accuratezza)
   - Seleziona il profilo di risoluzione (small, medium, large, o custom)
   - Il modello verrà scaricato automaticamente e il motore TensorRT compilato al primo utilizzo

2. **Processa Frame**: Inserisci `Right Click -> Add Node -> tensorrt -> Rife Tensorrt`
   - Collega il modello caricato dal passaggio 1
   - Inserisci i tuoi frame video
   - Configura le impostazioni di interpolazione (moltiplicatore, CUDA graph, ecc.)

## 🤖 Ambiente Testato

- Windows 11, CUDA 13.0, TensorRT 10.15.1.29, Python 3.12, RTX 5090
- WSL Ubuntu 24.04.03 LTS, CUDA 12.9, TensorRT 10.13.3.9, Python 3.12.11, RTX 5080

## 🚨 Aggiornamenti

### Febbraio 2026
- **Installazione Completamente Automatica**: Rilevamento CUDA e installazione TensorRT al caricamento del nodo
- **Correzione Dipendenze**: Aggiornato TensorRT a 10.15.1.29 per risolvere conflitti di installazione
- **Supporto RTX 5090**: Testato e confermata compatibilità con RTX 5090
- **Documentazione Risoluzione**: Aggiunta guida chiara sui limiti di risoluzione e preprocessing
- **Profilo Large**: Aggiunto supporto per risoluzioni fino a 1920px

### Gennaio 2026
- **CUDA 13 Default**: Aggiornato a CUDA 13.0 e TensorRT 10.14.1.48
- **Rilevamento CUDA Automatico**: Rileva automaticamente CUDA toolkit e percorsi DLL
- **Profili di Risoluzione**: Aggiunti profili small/medium/custom per ridurre l'uso della VRAM

### Dicembre 2025
- **Gestione Modelli Automatica**: Niente più download manuali! I modelli vengono scaricati automaticamente da HuggingFace e i motori TensorRT vengono compilati su richiesta
- **Workflow Migliorato**: Nuovo sistema a due nodi con `Load Rife Tensorrt Model` + `Rife Tensorrt` per migliore organizzazione
- **Dipendenze Aggiornate**: TensorRT aggiornato a 10.13.3.9 per migliori performance e compatibilità

## 👏 Crediti

- https://github.com/styler00dollar/VSGAN-tensorrt-docker
- https://github.com/Fannovel16/ComfyUI-Frame-Interpolation
- https://github.com/hzwer/ECCV2022-RIFE

## Licenza

[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

---

[📖 Read this in English](README.md)
