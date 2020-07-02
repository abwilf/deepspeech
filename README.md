# Deepspeech Tutorial and Wrapper

## Usage
### CLI
```
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.scorer
python3 convert.py --wav_dir /z/abwilf/wavs_test --out_path /z/abwilf/hey.pk --graph_path /z/abwilf/deepspeech/deepspeech-0.7.4-models.pbmm --scorer_path /z/abwilf/deepspeech/deepspeech-0.7.4-models.scorer
```

### API
```
from convert import *
wav_dir = '/z/abwilf/wavs_test'
out_path = '/z/abwilf/hey.pk'
graph_path = '/z/abwilf/deepspeech/deepspeech-0.7.4-models.pbmm'
scorer_path = '/z/abwilf/deepspeech/deepspeech-0.7.4-models.scorer'
get_transcripts(wav_dir, graph_path, scorer_path, out_path)
```
