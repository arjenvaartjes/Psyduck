# patch for scipy's shitty loadmat code
# Author: mergen
# https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries

import scipy.io as spio
import pathlib

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

class DictToObj:
    """
    Recursively convert a dictionary into a nested object,
    where each key becomes an attribute of the object.
    """
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            # Recursively convert value if it's a dict
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            elif isinstance(value, list):
                obj_list = []
                for item in list:
                    obj_list.append(DictToObj(item))
                setattr(self, key, obj_list)
            else:
                setattr(self, key, value)

def dict_to_obj(dictionary):
    """
    Convenience function that wraps the DictToObj class,
    in case you prefer to call a function rather than a constructor.
    """
    return DictToObj(dictionary)

def LoadMewtwoData(path):
    filePath = pathlib.Path(path)
    data = loadmat(filePath.joinpath(filePath.name+'_organised.mat'))['dataset']
    return dict_to_obj(data)