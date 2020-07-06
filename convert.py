import scipy.io.wavfile as wav
from deepspeech import Model
import glob, argparse
import multiprocessing.dummy as mp
from tensorflow.keras.utils import Progbar
from utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GRAPH_PATH = '/z/abwilf/deepspeech/deepspeech-0.7.4-models.pbmm'
SCORER_PATH = '/z/abwilf/deepspeech/deepspeech-0.7.4-models.scorer'
DS_SEP = '__-__'

# globals for shared multithreading memory
transcripts = {}
model = None
p = None
out_filepath = None
seconds_per_batch_global = None
orig_wav_dir = None
temp_wav_dir = None

def load_deepspeech_model(graph_path=GRAPH_PATH, scorer_path=SCORER_PATH):
    ds = Model(graph_path)
    ds.enableExternalScorer(scorer_path)
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
    return words, final_intervals

def transcribe_wrapper(audio_path):
    '''For multiprocessing'''
    id = audio_path.split('/')[-1].split('.')[0]
    features, intervals = transcribe(audio_path, model)
    transcripts[id] = {
        'features': features,
        'intervals': intervals
    }
    p.add(1)
    if np.random.random() < .2:
        save_pk(out_filepath, transcripts)

def get_transcripts(wav_dir, graph_path, scorer_path, out_path):
    num_workers = 6
    pool = mp.Pool(num_workers)

    global model, p, out_filepath
    model = load_deepspeech_model(graph_path, scorer_path)
    audio_paths = glob.glob(os.path.join(wav_dir, '*'))
    p = Progbar(len(audio_paths))
    out_filepath = out_path

    # for audio_path in audio_paths: # if not multiprocessing capable machine
    #     transcribe_wrapper(audio_path)

    pool.imap_unordered(transcribe_wrapper, audio_paths)
    pool.close()
    pool.join()
    print('\n')
    save_pk(out_path, transcripts)

def batch_split(arr, batch_size):
    num_batches = np.ceil(arr.shape[0] / batch_size)
    return [arr[batch_idx*batch_size:(batch_idx+1)*batch_size] for batch_idx in range(int(num_batches))]

def split_wavs_helper(audio_path):
    sample_rate, audio = wav.read(audio_path)
    audio = batch_split(audio, sample_rate*seconds_per_batch_global)
    id = audio_path.split('/')[-1].split('.')[0]
    for idx, audio_segment in enumerate(audio):
        temp_path = join(temp_wav_dir, f'{id}{DS_SEP}{idx}.wav')
        wav.write(temp_path, sample_rate, audio_segment)
    p.add(1)

def split_wavs(wav_dir, seconds_per_batch=30):
    print('Splitting wavs into 30 second chunks for transcription...')
    global seconds_per_batch_global, orig_wav_dir, temp_wav_dir, p
    temp_wav_dir = join(wav_dir, 'temp')
    orig_wav_dir = wav_dir
    seconds_per_batch_global = seconds_per_batch

    rmtree(temp_wav_dir)
    audio_paths = glob.glob(os.path.join(wav_dir, '*'))
    rm_mkdirp(temp_wav_dir, overwrite=True, quiet=True)

    # testing
    # audio_paths = list(filter(lambda elt: 'Xa086gxLJ3Y' in elt, audio_paths))
    # print(audio_paths)

    # for audio_path in tqdm(audio_paths): # if not multiprocessing capable machine
    #     split_wavs_helper(audio_path)

    p = Progbar(len(audio_paths))
    num_workers = 6
    pool = mp.Pool(num_workers)

    pool.imap_unordered(split_wavs_helper, audio_paths)
    pool.close()
    pool.join()
    print('\n')
    
    return temp_wav_dir

def recombine_partial_transcripts(out_path):
    print('\nRecombining partial transcripts...')
    transcripts = load_pk(out_path)
    full_dict = {}

    for key, idx in tqdm(lmap(lambda elt: elt.split(DS_SEP), sorted(lkeys(transcripts)))):
        assembled = key+DS_SEP+idx

        if key not in full_dict:
            full_dict[key] = transcripts[assembled]
        else:
            full_dict[key]['features'] = np.concatenate([full_dict[key]['features'], transcripts[assembled]['features']])
            full_dict[key]['intervals'] = np.concatenate([full_dict[key]['intervals'], transcripts[assembled]['intervals'] + full_dict[key]['intervals'].max()])
    
    save_pk(out_path, full_dict)


def convert_wavs(wav_dir, out_path, graph_path=GRAPH_PATH, scorer_path=SCORER_PATH, seconds_per_batch=30):
    temp_wav_dir = split_wavs(wav_dir, seconds_per_batch=seconds_per_batch)
    get_transcripts(temp_wav_dir, graph_path, scorer_path, out_path)
    recombine_partial_transcripts(out_path)
    rmtree(temp_wav_dir)

    print('\nSuccessfully finished transcribing.')

if __name__ == '__main__':
    '''
    Usage:
        python3 convert.py --wav_dir /z/abwilf/mosi/full/mosei_wavs --out_path /z/abwilf/transcripts.pk --graph_path /z/abwilf/deepspeech/deepspeech-0.7.4-models.pbmm --scorer_path /z/abwilf/deepspeech/deepspeech-0.7.4-models.scorer
    '''
    parser = argparse.ArgumentParser(description='Use deepspeech to transcribe audio')
    parser.add_argument('--wav_dir', type=str, help='Absolute path indicating where the wavs are. Each wav should be of the form /path/to/id.wav')
    parser.add_argument('--out_path', type=str, help='Absolute path to the .pk file where the transcripts will be stored')
    args = parser.parse_args()

    convert_wavs(args.wav_dir, args.out_path)

