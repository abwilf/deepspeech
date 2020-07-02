import scipy.io.wavfile as wav
from deepspeech import Model
import sys, argparse
import numpy as np
import pickle
import glob
import multiprocessing.dummy as mp
import os
from tensorflow.keras.utils import Progbar

transcripts = {}
model = None
p = None
out_filepath = None

def save_pk(file_stub, pk, protocol=None):
    filename = file_stub if '.pk' in file_stub else f'{file_stub}.pk'
    with open(filename, 'wb') as f:
        pickle.dump(pk, f, protocol=protocol)
    
def ar(a):
    return np.array(a)

def lzip(*keys):
    return list(zip(*keys))

def arzip(*keys):
    return [ar(elt) for elt in lzip(*keys)]
    
def load_model(graph_path, scorer):
    ds = Model(graph_path)
    ds.enableExternalScorer(scorer)
    return ds

def transcribe(audio_path, model):
    sample_rate, audio = wav.read(audio_path)
    assert sample_rate == 16000

    output = model.sttWithMetadata(audio)

    a = list(map(lambda elt: (elt.text, elt.start_time), output.transcripts[0].tokens))
    chars, intervals = arzip(*a)
    split_pts = np.concatenate([[0], np.where(ar(chars) == ' ')[0]])
    coords = lzip(split_pts, np.concatenate([split_pts[1:], [-1]]))

    words = []
    final_intervals = []
    for start, end in coords:
        if end == -1: # last
            interval = intervals[start], intervals[start] + (intervals[start] - intervals[start-1])
            word = ''.join(chars[start:]).strip()

        else:
            interval = intervals[start], intervals[end]
            word = ''.join(chars[start:end]).strip()
        
        words.append(word)
        final_intervals.append(interval)

    last_start = start

    words = ar(words)
    final_intervals = ar(final_intervals)
    print(words, final_intervals)

    return words, final_intervals

def transcribe_wrapper(audio_path):
    '''For multiprocessing'''
    id = audio_path.split('/')[-1].split('.')[0]
    transcripts[id] = transcribe(audio_path, model)
    p.add(1)
    if np.random.random() < .8:
        save_pk(out_filepath, transcripts)

def get_transcripts(wav_dir, graph_path, scorer_path, out_path):
    num_workers = 6
    pool = mp.Pool(num_workers)

    global model
    model = load_model(graph_path, scorer_path)
    audio_paths = glob.glob(os.path.join(wav_dir, '*'))
    global p
    p = Progbar(len(audio_paths))
    global out_filepath
    out_filepath = out_path

    # for audio_path in audio_paths:
    #     transcribe_wrapper(audio_path)

    pool.imap_unordered(transcribe_wrapper, audio_paths)
    pool.close()
    pool.join()

    save_pk(out_path, transcripts)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use deepspeech to transcribe audio')
    parser.add_argument('--wav_dir', type=str, help='Absolute path indicating where the wavs are. Each wav should be of the form /path/to/id.wav')
    parser.add_argument('--graph_path', type=str, help='Absolute path to graph file (ends in .pbmm)')
    parser.add_argument('--scorer_path', type=str, help='Absolute path to scorer file (ends in .scorer)')
    parser.add_argument('--out_path', type=str, help='Absolute path to the .pk file where the transcripts will be stored')
    args = parser.parse_args()

    get_transcripts(args.wav_dir, args.graph_path, args.scorer_path, args.out_path)

