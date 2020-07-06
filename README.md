# DeepSpeech Wrapper
DeepSpeech is a powerful opensource transcription tool.  This repository implements a wrapper over the python `deepspeech` module enabling it to perform well with arbitrary audio lengths (the model has difficulty with audio sequences longer than 30 seconds).  This wrapper draws heavily on the [deepspeech documentation](https://github.com/mozilla/DeepSpeech-examples/tree/r0.7/vad_transcriber), but also yields word level timestamps and is parallelized across 6 threads, which the documentation does not offer.

## Usage
### CLI
```
pip install numpy tensorflow-gpu==2.0.0 deepspeech==0.7.4 scipy tqdm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.pbmm
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.7.4/deepspeech-0.7.4-models.scorer
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
convert_wavs(wav_dir, out_path)
```
