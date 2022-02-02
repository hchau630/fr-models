import pathlib
import os
import pickle
import json

from fr_models import exceptions

def assign_dict(d, keys, value):
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value

def flatten_dict(d, depth=-1):
    """
    depth=0 returns the dictionary unchanged, depth=-1 returns a fully flattened dictionary
    depth=n means the first n layers of the dictionary is flattened, so a depth k dict becomes a depth k-1 dict.
    for example: let d = {'a': {'b': {'c': d}}}
    then flatten_dict(d, depth=1) = {('a','b'): {'c': d}}
    """
    flattened_dict = {}
    for k, v in d.items():
        if depth == 0 or not isinstance(v, dict) or v == {}: # if depth == 0 or v is leaf
            flattened_dict[k] = v
        else:
            for new_k, new_v in flatten_dict(v, depth=depth-1).items():
                flattened_dict[(k, *new_k)] = new_v
    return flattened_dict

def save(path, data, extension, depth=-1, overwrite=False):
    """
    Saves data at path.
    If path has a suffix, this must match the extension, and the data 
    is stored as a single file with the specified extension at path.
    The depth parameter is ignored in this case.
    If path does not have suffix, then path is the directory in which
    the data is stored. The data must be a dictionary.
    The depth parameter controls the depth of the directory.
    If depth=0, then path will become a depth 0 directory, and
    if depth=-1, then path will be a directory as deep as the data dictionary.
    """
    path = pathlib.Path(path)
    
    if extension == 'pkl':
        def save_func(filename, dat):
            with open(filename, 'wb') as f:
                pickle.dump(dat, f)
    elif extension == 'json':
        def save_func(filename, dat):
            with open(filename, 'w') as f:
                json.dump(dat, f, indent=4)
    else:
        raise NotImplementedError(f"Extension {extension} is not yet implemented")
    
    if path.suffix != '': # path is a filename
        suffix = path.suffix[1:]
        if suffix != extension:
            raise ValueError(f"path suffix must match extension if suffix is present. suffix: {suffix}, extension: {extension}.")
        if not overwrite and path.is_file():
            raise exceptions.PathAlreadyExists(f"The file {str(path)} already exists.")
        path.parent.mkdir(parents=True, exist_ok=True)
        save_func(path, data)
        
    else: # path is a directory
        for key, val in flatten_dict(data, depth=depth).items():
            filename = path / f"{'/'.join(key)}.{extension}"
            filename.parent.mkdir(parents=True, exist_ok=True)
            if not overwrite and filename.is_file():
                raise exceptions.PathAlreadyExists(f"The file {str(filename)} already exists.")
            save_func(filename, val)
        
def save_data(path, data_dict, **kwargs):
    save(path, data_dict, 'pkl', **kwargs)
    
def save_config(path, config, **kwargs):
    save(path, config, 'json', **kwargs)
    
def load(path, extension):
    path = pathlib.Path(path)
    if not path.exists():
        raise exceptions.PathNotFound(f"The path {path} does not exist.")
        
    if extension == 'pkl':
        def load_func(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
    elif extension == 'json':
        def load_func(filename):
            with open(filename, 'r') as f:
                return json.load(f)
    else:
        raise NotImplementedError(f"Extension {extension} is not yet implemented")
        
    if path.is_file():
        data = load_func(path)
    elif path.is_dir():
        data = {}
        for cur_path, dirnames, filenames in os.walk(path):
            if '.ipynb_checkpoints' not in cur_path:
                for filename in filenames:
                    filename = pathlib.Path(filename)
                    if filename.suffix == f'.{extension}':
                        # with open(os.path.join(cur_path,filename), 'r') as f:
                        #     print(f.read())
                        # print(f"Done reading file {os.path.join(cur_path,filename)}")
                        cur_path_rel = pathlib.Path(cur_path).relative_to(path)
                        assign_dict(data, [*cur_path_rel.parts,filename.stem], load_func(os.path.join(cur_path,filename)))
    else:
        raise IOError("Path {path} is neither file nor directory")
        
    return data
        
def load_data(path):
    return load(path, 'pkl')

def load_config(path):
    return load(path, 'json')