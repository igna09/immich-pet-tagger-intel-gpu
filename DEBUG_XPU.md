## Debugging Intel XPU Support

Hai aggiunto supporto Intel iGPU, ma l'app usa ancora CPU. Seguire questi step per diagnosticare il problema.

### 1. Verificare cosa è installato nel container

Dopo aver ricostruito l'immagine con `docker compose build --build-arg XPU=true`:

```bash
docker compose run --rm immich-pet-tagger python debug_device.py
```

Questo mostra:
- Se torch è correttamente installato e la versione
- Se `torch.xpu` è disponibile
- Se `intel_extension_for_pytorch` è installato
- Quale device viene selezionato

### 2. Controllare i log dell'app

Una volta avviato il container:

```bash
docker compose up -d
docker compose logs -f immich-pet-tagger
```

Cerca i messaggi che iniziano con:
- `[device]` - Debug del device detection
- `[detector]` - YOLO worker device
- `[embedder]` - CLIP worker device
- `[main]` - Poller info

Se vedi messaggi di errore tipo:
- `CLIP worker X failed to load on xpu: ...` - il modello non riesce a caricare su XPU
- `torch.xpu: NOT installed` - manca intel_extension_for_pytorch

### 3. Problemi comuni

#### `torch.xpu module not available`

**Causa**: `intel_extension_for_pytorch` non è installato o la versione di torch non è compatibile.

**Soluzione**: 
- Verificare che il Dockerfile installi `intel-extension-for-pytorch` prima di torch
- Assicurarsi che l'indice PyPI sia corretto: `https://pytorch-extension.intel.com/release-whl/stable/xpu/`
- Buildare di nuovo: `docker compose build --build-arg XPU=true --no-cache`

#### `torch.xpu.is_available() = False`

**Causa**: I driver Intel GPU non sono installati sul host o non sono corretti.

**Soluzione**:
1. Verificare i driver Intel GPU sul host:
   ```bash
   # Su Linux
   ls -la /dev/dri/by-path/
   
   # O installare Level Zero:
   # Ubuntu: sudo apt install level-zero-loader intel-opencl-icd
   # Fedora: sudo dnf install level-zero intel-opencl
   ```

2. Assicurarsi che il container abbia accesso a `/dev/dri`:
   ```yaml
   devices:
     - /dev/dri:/dev/dri
   group_add:
     - video
   ```

3. Verificare i driver nel container:
   ```bash
   docker compose exec immich-pet-tagger ls -la /dev/dri/
   docker compose exec immich-pet-tagger clinfo 2>/dev/null || echo "OpenCL not available"
   ```

#### `Module 'torch' has no attribute 'xpu'`

**Causa**: torch non è compilato con supporto XPU.

**Soluzione**: Ricostruire il container con `--no-cache` per ignorare le cache di layer precedenti:
```bash
docker compose build --build-arg XPU=true --no-cache
```

### 4. Verificare il device selection logic

Il codice seleziona il device in questo ordine:
1. Se `torch.cuda` è disponibile e funziona → usa `cuda`
2. Se `torch.xpu` è disponibile e funziona → usa `xpu`
3. Altrimenti → usa `cpu`

Se vedi "Using CPU device" nei log:
- Significa che sia CUDA che XPU non sono disponibili
- Controlla il step 1 e 2 di questa sezione

### 5. Log di fallback

Se vedi messaggi come:
- `YOLO worker 0 failed to load on xpu: ...`
- `CLIP worker 0 failed to load on xpu: ...`

Significa che il device è stato rilevato (`torch.xpu` esiste e ritorna True) ma il caricamento del modello fallisce.

Questi errori sono loggati con `exc_info=True`, quindi vedrai lo stack trace completo che ti aiuterà a identificare il problema specifico.

### 6. Fare il rebuild corretto

Se sei passato da una build precedente (CPU/CUDA):

```bash
# Pulire completamente
docker compose down -v
docker system prune -a --volumes

# Ricostruire
docker compose build --build-arg XPU=true --no-cache
docker compose up -d
```
