# %load training_functions.py
import pandas as pd
import os
import numpy as np
from datetime import datetime
import json
from os import listdir
from os.path import isfile, join


def read_csv_power_file(file_path, filename):
    csv_path = os.path.join(file_path, filename)
    df = pd.read_csv(csv_path)

    # Drop columns that we don't need
    df.drop('lat', axis=1, inplace=True)
    df.drop('lon', axis=1, inplace=True)
    df.drop('nm', axis=1, inplace=True)
    df.drop('hhb', axis=1, inplace=True)
    df.drop('o2hb', axis=1, inplace=True)
    df.drop('thb', axis=1, inplace=True)
    df.drop('smo2', axis=1, inplace=True)
    df.drop('rps', axis=1, inplace=True)
    df.drop('lps', axis=1, inplace=True)
    df.drop('rte', axis=1, inplace=True)
    df.drop('lte', axis=1, inplace=True)
    df.drop('headwind', axis=1, inplace=True)
    df.drop('slope', axis=1, inplace=True)

    # Replace 0's in columns like cadence
    df['cad'].replace(0, value=np.NaN, inplace=True)
    df['kph'].replace(0, value=np.NaN, inplace=True)
    df['hr'].replace(0, value=np.NaN, inplace=True)

    return df


def calculate_height_gain(hd):
    old = hd[0]
    height_gain = 0

    for v in hd:
        if v > old:
            height_gain += v - old
        old = v

    return height_gain


def normalized_power(pd):
    return np.power((pd ** 4).sum() / len(pd), 0.25)


def hr_drift(hrd, pd):
    l = len(hrd) / 2

    first_half = hrd[:l].mean() / pd[:l].mean()
    second_half = hrd[l:].mean() / pd[l:].mean()

    return 1 - (first_half / second_half)


def kilojoules(pd):
    return (pd.mean() * len(pd)) / 1000


def do_aggregations(d, filename, FTP):
    epoch_day = get_epoch_day(filename)
    df = pd.DataFrame(columns=('filename', 'epoch_day', 'time', 'cad', 'hr', 'hr_min', 'hr_max', 'hr_drift',
                               'km', 'kph', 'kilojoules', 'watts', 'watts_max', 'watts_std', 'watts_25',
                               'watts_50', 'watts_75', 'np', 'alt', 'temp', 'vi', 'tss', 'if'))

    HR = d.get('HR', np.full([len(d)], np.nan))
    KM = d.get('KM', np.full([len(d)], np.nan))
    KPH = d.get('KPH', np.full([len(d)], np.nan))
    WATTS = d.get('WATTS', np.full([len(d)], np.nan))
    CAD = d.get('CAD', np.full([len(d)], np.nan))

    alt = calculate_height_gain(d.get('ALT', np.full([len(d)], np.nan)))
    int_fac = normalized_power(WATTS) / FTP
    minutes = -1
    minutes = len(HR) / 60
    tss = (minutes / 60.0) * 100 * int_fac

    w_25 = 0
    w_50 = 0
    w_75 = 0

    if d.get('WATTS') is not None:
        watts = d.get('WATTS')
        w_25 = WATTS.quantile(q=0.25)
        w_50 = WATTS.quantile(q=0.50)
        w_75 = WATTS.quantile(q=0.75)

    v_index = normalized_power(WATTS) / WATTS.mean()
    vi = normalized_power(WATTS) / WATTS.mean()

    df.loc[0] = [filename, epoch_day, minutes, CAD.mean(), HR.mean(), HR.min(),
                 HR.max(), hr_drift(HR, WATTS), KM.max(), KPH.mean(), kilojoules(WATTS),
                 WATTS.mean(), WATTS.max(), WATTS.std(), w_25, w_50, w_75, normalized_power(WATTS),
                 alt, d.TEMP.mean(), v_index, tss, int_fac]

    df.filename.apply(str)

    return df

def do_aggregations_json(d, filename, FTP):
    i = do_aggregations(d, filename, FTP)
    i['training_type'] = tag_classifier_by_power(i['watts'].mean(), FTP)

    return i

# def do_aggregations_json(d, epoch_day, filename, FTP):
#     df = pd.DataFrame(columns=('filename', 'epoch_day', 'time', 'cad', 'hr', 'hr_min', 'hr_max', 'hr_drift',
#                                'km', 'kph', 'kilojoules', 'watts', 'watts_max', 'watts_std', 'watts_25',
#                                'watts_50', 'watts_75', 'np', 'alt', 'temp', 'vi', 'tss', 'if', 'training_type'))
#
#     alt = calculate_height_gain(d['ALT'].values)
#     hrdrift = hr_drift(d['HR'], d['WATTS'])
#     v_index = normalized_power(d['WATTS']) / d['WATTS'].mean()
#     minutes = len(d['HR']) / 60
#     int_fac = normalized_power(d['WATTS']) / FTP
#     tss = (minutes / 60.0) * 100 * int_fac
#
#     df.loc[0] = [filename, epoch_day, minutes, d['CAD'].mean(), d['HR'].mean(), d['HR'].min(),
#                  d['HR'].max(), hrdrift, d['KM'].max(), d['KPH'].mean(), kilojoules(d['WATTS']),
#                  d['WATTS'].mean(), d['WATTS'].max(), d['WATTS'].std(), d['WATTS'].quantile(q=0.25),
#                  d['WATTS'].quantile(q=0.50), d['WATTS'].quantile(q=0.75), normalized_power(d['WATTS']),
#                  alt, d['TEMP'].mean(), v_index, tss, int_fac, tag_classifier_by_power(d['WATTS'].mean(), FTP)]
#
#     df.filename.apply(str)
#
#     df['kph'].fillna(0, inplace=True)
#
#     return df


def get_epoch_day(date_s):
    day = date_s.split('.')[0]
    return (datetime.strptime(day, '%Y_%m_%d_%H_%M_%S') - datetime(1970, 1, 1)).days


def read_json_file(file_path, filename):
    return json.loads(open(os.path.join(file_path, filename)).read())


def tag_classifier_by_power(interval, FTP):
    p = interval

    if p <= FTP * 0.55:
        return 'Active recovery'
    elif p <= FTP * 0.75:
        return 'Endurance'
    elif p <= FTP * 0.87:
        return 'Tempo'
    elif p <= FTP * 0.91:
        return 'SS'
    elif p <= FTP * 1.05:
        return 'FTP'
    elif p <= FTP * 1.2:
        return 'VO2'
    elif p <= FTP * 1.5:
        return 'Anaerobic'
    elif p > FTP * 1.5:
        return 'Neuromuscular power'
    else:
        return 'NA'


def get_intervals_from_json_map(j, filename, FTP):
    ride = j.get('RIDE')
    intervals = ride.get('INTERVALS')
    samples = ride.get('SAMPLES')

    d1 = [pd.DataFrame(
        samples[interval.get('START'):interval.get('STOP')])
              .drop(['LRBALANCE', 'LAT', 'LON', 'SLOPE'], axis=1, errors='ignore')
          for interval in intervals]

    d1 = filter(lambda x: len(x), d1)

    if len(d1) == 0:
        print('Omitting file: %s' % (filename))
        return pd.DataFrame()

    return pd.concat([
        do_aggregations_json(
            interval,
            filename,
            FTP)
        for interval in d1])


def files_in_folder(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]


def read_intervals(path, FTP):
    dfs = [get_intervals_from_json_map(read_json_file(path, filename), filename, FTP)
           for filename in files_in_folder(path)
           if filename.split('.')[1] == 'utf8']

    t_data = pd.concat([f for f in dfs])
    t_data.sort_values('epoch_day', axis=0, inplace=True)

    return t_data


def read_rides(path, FTP):
    files = files_in_folder(path)

    loaded_files = [do_aggregations(
        pd.DataFrame(read_json_file(path, filename)
                     .get('RIDE').get('SAMPLES')), filename, FTP)
                        .drop(['SLOPE', 'LAT', 'LON', 'LRBALANCE'], axis=1, errors='ignore')
                    for filename in files
                    if filename.split('.')[1] == 'utf8']

    rides = pd.concat(loaded_files).sort_values('epoch_day', axis=0)

    rides['prev_1_day'] = rides.epoch_day.shift(periods=1)
    rides['prev_2_day'] = rides.epoch_day.shift(periods=2)
    rides['prev_3_day'] = rides.epoch_day.shift(periods=3)
    rides['prev_4_day'] = rides.epoch_day.shift(periods=4)
    rides['prev_5_day'] = rides.epoch_day.shift(periods=5)
    rides['prev_6_day'] = rides.epoch_day.shift(periods=6)
    rides['prev_7_day'] = rides.epoch_day.shift(periods=7)

    rides['resting_days'] = rides.epoch_day.sub(rides['prev_1_day'], axis=0)

    return rides
