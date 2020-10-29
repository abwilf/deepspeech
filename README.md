# DeepSpeech Wrapper
DeepSpeech is a powerful opensource transcription tool.  This repository acts as a wrapper over the python `deepspeech` module, enabling it to perform well with arbitrary audio lengths (the model has difficulty with segments longer than 30 seconds). This project builds on the documentation by parallelizing transcription and providing word level timestamps. For more information, see the [deepspeech documentation](https://github.com/mozilla/DeepSpeech-examples/tree/r0.7/vad_transcriber). 

## Usage
### CLI
```
pip install numpy tensorflow==2.3.0 deepspeech-gpu==0.8.2 scipy tqdm webrtcvad
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.8.2/deepspeech-0.8.2-models.scorer
```

Modify `GRAPH_PATH` and `SCORER_PATH` in `convert.py` to match your filepath locations.

```
python3 convert.py --wav_dir /path/to/wavs --out_path /path/to/transcripts.pk
```

### Python
```
from convert import convert_wavs
wav_dir = '/path/to/wavs'
out_path = '/path/to/transcripts.pk'
aggressivness = 2
convert_wavs(wav_dir, out_path, aggressivness)
```
