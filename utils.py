import shutil, os, pathlib, pickle, sys, math, importlib, json.tool
import numpy as np
from os.path import join, exists
from tqdm import tqdm
from itertools import product

def obj_to_grid(a):
    '''get all objects corresponding to hyperparamter grid search
    a = {
        'hey': [1,2],
        'there': [3,4],
        'people': 5
    }
    ->
    {'hey': 1, 'there': 3, 'people': 5}
    {'hey': 1, 'there': 4, 'people': 5}
    {'hey': 2, 'there': 3, 'people': 5}
    {'hey': 2, 'there': 4, 'people': 5}
    '''

    for k,v in list(a.items()):
        if type(v) != list:
            a[k] = [v]

    to_ret = []
    for values in list(product(*list(a.values()))):
        to_ret.append({k:v for k,v in zip(a.keys(), values)})
    return to_ret

def ar(a):
    return np.array(a)

def rmtree(dir_path):
    print(f'Removing {dir_path}')
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    # else:
        # print(f'{dir_path} is not a directory, so cannot remove')

def npr(x, decimals=2):
    return np.round(x, decimals=decimals)
    
def int_to_str(*keys):
    return [list(map(lambda elt: str(elt), key)) for key in keys]
    
def rm_mkdirp(dir_path, overwrite, quiet=False):
    if os.path.isdir(dir_path):
        if overwrite:
            if not quiet:
                print('Removing ' + dir_path)
            shutil.rmtree(dir_path, ignore_errors=True)

        else:
            print('Directory ' + dir_path + ' exists and overwrite flag not set to true.  Exiting.')
            exit(1)
    if not quiet:
        print('Creating ' + dir_path)
    pathlib.Path(dir_path).mkdir(parents=True)

def lists_to_2d_arr(list_in, max_len=None):
    '''2d list in, but where sub lists may have differing lengths, one big padded 2d arr out'''
    max_len = max([len(elt) for elt in list_in]) if max_len is None else max_len
    new_arr = np.zeros((len(list_in), max_len))
    for i,elt in enumerate(list_in):
        if len(elt) < max_len:
            new_arr[i,:len(elt)] = elt
        else:
            new_arr[i,:] = elt[:max_len]
    return new_arr


def mkdirp(dir_path):
    if not os.path.isdir(dir_path):
        pathlib.Path(dir_path).mkdir(parents=True)

def rm_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)
    
def rglob(dir_path, pattern):
    return list(map(lambda elt: str(elt), pathlib.Path(dir_path).rglob(pattern)))

def move_matching_files(dir_path, pattern, new_dir, overwrite):
    rm_mkdirp(new_dir, True, overwrite)
    for elt in rglob(dir_path, pattern):
        shutil.move(elt, new_dir)
    
def subset(a, b):
    return np.min([elt in b for elt in a]) > 0

def list_gpus():
    return tf.config.experimental.list_physical_devices('GPU')


def save_pk(file_stub, pk, protocol=None):
    filename = file_stub if '.pk' in file_stub else f'{file_stub}.pk'
    rm_file(filename)
    with open(filename, 'wb') as f:
        pickle.dump(pk, f, protocol=protocol)
    
def load_pk(file_stub):
    filename = file_stub
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'rb') as f:
        obj = pickle.load(f)
        return obj

def load_pk_old(filename):
    with open(filename, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        p = u.load()
        return p

def get_ints(*keys):
    return [int(key) for key in keys]

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_json(file_stub, obj):
    filename = file_stub
    with open(filename, 'w') as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=4)

def load_json(file_stub):
    filename = file_stub
    with open(filename) as json_file:
        return json.load(json_file)

def lfilter(fn, iterable):
    return list(filter(fn, iterable))

def lkeys(obj):
    return list(obj.keys())

def lvals(obj):
    return list(obj.values())

def lmap(fn, iterable):
    return list(map(fn, iterable))

def sort_dict(d, reverse=False):
    return {k: v for k,v in sorted(d.items(), key=lambda elt: elt[1], reverse=reverse)}

def csv_path(sym):
    return join('csvs', f'{sym}.csv')

def is_unique(a):
    return len(np.unique(a)) == len(a)

def lists_equal(a,b):
    return np.all([elt in b for elt in a]) and np.all([elt in a for elt in b])
    
def split_arr(cond, arr):
    return lfilter(cond, arr), lfilter(lambda elt: not cond(elt), arr)

def lzip(*keys):
    return list(zip(*keys))

def arzip(*keys):
    return [ar(elt) for elt in lzip(*keys)]
    
def dilation_pad(max_len, max_dilation_rate):
    to_ret = math.ceil(max_len/max_dilation_rate)*max_dilation_rate
    assert (to_ret % max_dilation_rate) == 0
    return to_ret

def zero_pad_to_length(data, length):
    padAm = length - data.shape[0]
    if padAm == 0:
        return data
    else:
        return np.pad(data, ((0,padAm), (0,0)), 'constant')

def paths_to_mfbs(paths, max_len):
    '''Get normalized & padded mfbs from paths'''
    # normalize
    mfbs = None
    for file_name in paths:
        if mfbs is None:
            mfbs = np.array(np.load(file_name))
        else:
            mfbs = np.concatenate([mfbs, np.load(file_name)], axis=0)
    mean_vec = np.mean(mfbs, axis=0)
    std_vec  = np.std(mfbs, axis=0)

    # concat & pad
    mfbs = None
    for file_name in paths:
        mfb = (np.load(file_name) - mean_vec) / (std_vec + np.ones_like(std_vec)*1e-3)
        if mfbs is None:
            mfbs = np.array([zero_pad_to_length(mfb, max_len)])
        else:
            mfbs = np.concatenate([mfbs, [zero_pad_to_length(mfb, max_len)]], axis=0)
    return tf.cast(mfbs, tf.float64)

def destring(y, width=3):
    ''' y is an array with elements in the format '.5;.5;0.'.  Need to turn into nx3 arr'''
    y = np.array(y)
    y_new = np.zeros((len(y), width))
    for i in range(len(y)):
        if '[0.333' in y[i]:
            y_new[i] = [.333, .333, .333]
            continue
        assert ';' in y[i] or ' ' in y[i]
        char = ';' if ';' in y[i] else None
        y_new[i] = list(map(lambda elt: float(elt), y[i].split(char)))
    return y_new

def get_batch(arr, batch_idx, batch_size):
    return arr[batch_idx * batch_size:(batch_idx + 1) * batch_size]

def sample_batch(x, y, batch_size):
    start = np.random.randint(x.shape[0]-batch_size)
    x_batch = x[start:start+batch_size]
    y_batch = y[start:start+batch_size]
    return x_batch, y_batch

def get_mfbs(paths, lengths_dict, max_dilation_rate):
    max_len = max([lengths_dict[file_name] for file_name in paths])
    max_len = dilation_pad(max_len, max_dilation_rate)
    return paths_to_mfbs(paths, max_len)

def shuffle_data(*arrs):
    rnd_state = np.random.get_state()
    for arr in arrs:
        np.random.shuffle(arr)
        np.random.set_state(rnd_state)

def get_class_weights(arr):
    '''pass in dummies'''
    class_weights = np.nansum(arr, axis=0)
    return np.sum(class_weights) / (class_weights*len(class_weights))

def get_class_weights_ds(arr):
    '''do not pass in dummies'''
    arr = np.stack(np.unique(np.array(arr), return_counts=True), axis=1)
    return (np.sum(arr[:,1]) - arr[:,1]) / arr[:,1]

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)

def path_to_func(path, func_name):
    '''From path, import function from module at that path'''
    import importlib.util
    spec = importlib.util.spec_from_file_location("module.name", path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    return getattr(foo, func_name)