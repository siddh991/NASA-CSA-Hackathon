import pandas as pd


def remove_small_fires(data: pd.DataFrame, scn_trck='scan', size=1):
    other = 'track'
    if scn_trck == 'track':
        other = 'scan'
    keep_data = []
    for index, row in data[scn_trck].items():
        keep = True
        if row <= size:
            keep = False
        keep_data.append(keep)
    data = data[keep_data].reset_index().drop(columns=['index', other])
    return data


def load_fire_data(files: [list, str]):
    if type(files) == str:
        data = pd.read_csv(files)
    elif type(files) == list:
        data = pd.DataFrame([])
        for file in files:
            data = data.append(pd.read_csv(file))
    else:
        raise ValueError('files must be a list or a string')
    data = data[data['type'] == 0.0]
    data = data.reset_index().drop(columns=['index', 'version', 'instrument', 'daynight', 'confidence', 'type', 'bright_t31'])
    data = remove_small_fires(data, size=4)
    return data


def convert_distance_lat_long(lat, long, x, y):
