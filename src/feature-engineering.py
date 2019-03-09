import pandas as pd 
import numpy as np 
import calendar as cal 

from multiprocessing import Pool 

num_partitions = 10
num_cores = 4

def parallelize_dataframe(data, function):
    data_split = np.array_split(data, num_partitions)
    pool = Pool(num_cores)
    data = pd.concat(pool.map(function, data_split))
    pool.close()
    pool.join()
    return data

def to_rad(degree):
    return degree * (np.pi / 180)

# in KM
def distance(start_lat, start_lon, end_lat, end_lon):
    radius = 6731
    dLat = to_rad(end_lat - start_lat)
    dLon = to_rad(end_lon - start_lon)

    a = (np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(to_rad(start_lat)) * 
        np.cos(to_rad(end_lat)) * np.sin(dLon / 2) * np.sin(dLon / 2))
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c
    return distance


def convert_lat_log(lat, lon):
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return x, y, z

def encode_cyclical(data, column, max_value):
    data[column + '_sin'] = np.sin(2 * np.pi * data[column]/max_value)
    data[column + '_cos'] = np.cos(2 * np.pi * data[column]/max_value)
    return data

def date_manipulation(data, date_col):
    data['day_of_week'] = data.apply(lambda x: cal.day_name[x[date_col].weekday()], axis=1)

    data['month'] = data.pickup_datetime.dt.month
    data['day'] = data.pickup_datetime.dt.day
    data['hour'] = data.pickup_datetime.dt.hour
    data['minute'] = data.pickup_datetime.dt.minute

    data = encode_cyclical(data, 'month', data['month'].max())
    data = encode_cyclical(data, 'day', data['day'].max())
    data = encode_cyclical(data, 'hour', data['hour'].max())
    data = encode_cyclical(data, 'minute', data['minute'].max())

    data['time_of_day'] = pd.cut(
        data['hour'],
        [0, 6, 12, 18, 23],
        labels=[0, 1, 2, 3] # night, morning, afternoon, evening
    )

    return data

def process(data_chunk):
    data_chunk['pickup_datetime'] = pd.to_datetime(data_chunk['pickup_datetime'], infer_datetime_format=True)
    data_chunk = date_manipulation(data_chunk, 'pickup_datetime')

    
    return data_chunk 


chunksize = 10 ** 6
filename = '/media/jonathan/HDD/data/new-york-city-taxi-fare/'
train_csv = 'train.csv'
test_csv = 'test.csv'

if __name__ == "__main__":
    debug = True
    for chunk in pd.read_csv(filename + train_csv, chunksize=chunksize):
        processed_data = process(chunk)
        if (debug):
            break
        ## write data to file.
