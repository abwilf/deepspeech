import scipy.io.wavfile as wav
from deepspeech import Model
import glob, argparse
import multiprocessing.dummy as mp
from tensorflow.keras.utils import Progbar
from ds_utils import *
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

GRAPH_PATH = '/z/abwilf/deepspeech/deepspeech-0.7.4-models.pbmm'
SCORER_PATH = '/z/abwilf/deepspeech/deepspeech-0.7.4-models.scorer'
DS_SEP = '__-__'

# globals for shared multithreading memory
transcripts = {}
model = None
p = None
out_filepath = None
orig_wav_dir = None
temp_wav_dir = None
global_aggressiveness = None

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
    print('\nTranscribing...')
    num_workers = 6
    pool = mp.Pool(num_workers)

    global model, p, out_filepath
    model = load_deepspeech_model(graph_path, scorer_path)
    audio_paths = glob.glob(os.path.join(wav_dir, '*'))
    p = Progbar(len(audio_paths))
    out_filepath = out_path

    # for audio_path in tqdm(audio_paths): # if not multiprocessing capable machine or if error not being caught in mp
    #     transcribe_wrapper(audio_path)

    pool.imap_unordered(transcribe_wrapper, audio_paths)
    pool.close()
    pool.join()
    print('\n')
    save_pk(out_path, transcripts)

def split_wavs_helper(audio_path):
    sample_rate, audio = wav.read(audio_path)
    id = audio_path.split('/')[-1].split('.')[0]
    segments, sample_rate, audio_length = vad_segment_generator(audio_path, aggressiveness=global_aggressiveness)
    for idx, segment in enumerate(segments):
        audio_segment = np.frombuffer(segment, dtype=np.int16)
        temp_path = join(temp_wav_dir, f'{id}{DS_SEP}{idx}.wav')
        wav.write(temp_path, sample_rate, audio_segment)
    p.add(1)

def split_wavs(wav_dir):
    print('Splitting wavs into VAD chunks for transcription...')
    global orig_wav_dir, temp_wav_dir, p
    temp_wav_dir = join(wav_dir, 'temp')
    orig_wav_dir = wav_dir

    rmtree(temp_wav_dir)
    audio_paths = glob.glob(os.path.join(wav_dir, '*'))
    rm_mkdirp(temp_wav_dir, overwrite=True, quiet=True)
    p = Progbar(len(audio_paths))

    # for audio_path in tqdm(audio_paths): # if not multiprocessing capable machine or an error
    #     split_wavs_helper(audio_path)

    num_workers = 6
    pool = mp.Pool(num_workers)

    x = pool.imap_unordered(split_wavs_helper, audio_paths)
    [elt for elt in x] # forces errors to propagate
    pool.close()
    pool.join()
    
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

def convert_wavs(wav_dir, out_path, aggressiveness, graph_path=GRAPH_PATH, scorer_path=SCORER_PATH, overwrite=False):
    if os.path.exists(out_path):
        if overwrite:
            rm_file(out_path)
        else:
            print(f'Transcripts exist in {out_path} and overwrite=False.  Skipping...')
            return

    print(f'#### Using deepspeech to convert wavs in {wav_dir} to {out_path} ####')
    global global_aggressiveness
    global_aggressiveness = aggressiveness

    temp_wav_dir = split_wavs(wav_dir)
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
    parser.add_argument('--aggressiveness', type=int, default=2, help='How aggressive the VAD algorithm should be.  In range(4). 0 is least, 4 is greatest.')
    args = parser.parse_args()

    convert_wavs(args.wav_dir, args.out_path, args.aggressiveness)

