import pandas as pd 
import numpy as np 
import calendar as cal 
import time
import dask.dataframe as dd 

from multiprocessing import Pool 

import warnings
warnings.filterwarnings('ignore')

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

def distance(row):
    start_lat = row['pickup_latitude']
    end_lat = row['dropoff_latitude']
    start_lon = row['pickup_longitude']
    end_lon = row['dropoff_longitude']

    radius = 6731
    dLat = to_rad(end_lat - start_lat)
    dLon = to_rad(end_lon - start_lon)

    a = (np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(to_rad(start_lat)) * 
        np.cos(to_rad(end_lat)) * np.sin(dLon / 2) * np.sin(dLon / 2))
    
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = radius * c
    return distance

def encode_cyclical(data, column, max_value):
    data[column + '_sin'] = np.sin(2 * np.pi * data[column]/max_value)
    data[column + '_cos'] = np.cos(2 * np.pi * data[column]/max_value)
    return data

def day_of_week(data): return data.pickup_datetime.dt.day_name()
def get_month(data): return data.pickup_datetime.dt.month
def get_day(data): return data.pickup_datetime.dt.day
def get_hour(data): return data.pickup_datetime.dt.hour
def get_min(data): return data.pickup_datetime.dt.minute

def date_manipulation(data, date_col):
    data['day_of_week'] = parallelize_dataframe(data, day_of_week)#data.apply(lambda x: cal.day_name[x[date_col].weekday()], axis=1)

    data['month'] = parallelize_dataframe(data, get_month)#data.pickup_datetime.dt.month
    data['day'] = parallelize_dataframe(data, get_day)#data.pickup_datetime.dt.day
    data['hour'] = parallelize_dataframe(data, get_hour)#data.pickup_datetime.dt.hour
    data['minute'] = parallelize_dataframe(data, get_min)#data.pickup_datetime.dt.minute

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


def x_dimension(lat, lon): return np.cos(lat) * np.cos(lon)
def y_dimension(lat, lon): return np.cos(lat) * np.sin(lon)
def z_dimension(lat): return np.sin(lat)

def three_dimensional_transform(data):
    # convert the lat and lon to three dimensional vectors to represent their closeness in space.
    data['start_x'] = np.vectorize(x_dimension)(data['pickup_latitude'], data['pickup_longitude'])
    data['start_y'] = np.vectorize(y_dimension)(data['pickup_latitude'], data['pickup_longitude'])
    data['start_z'] = np.vectorize(z_dimension)(data['pickup_latitude'])

    data['end_x'] = np.vectorize(x_dimension)(data['dropoff_latitude'], data['dropoff_longitude'])
    data['end_y'] = np.vectorize(y_dimension)(data['dropoff_latitude'], data['dropoff_longitude'])
    data['end_z'] = np.vectorize(z_dimension)(data['dropoff_latitude'])

    return data

# Given a dataframe, add two new features 'abs_diff_longitude' and
# 'abs_diff_latitude' reprensenting the "Manhattan vector" from
# the pickup location to the dropoff location.
def add_travel_vector_features(data):
    data['abs_diff_longitude'] = (data.dropoff_longitude - data.pickup_longitude).abs()
    data['abs_diff_latitude'] = (data.dropoff_latitude - data.pickup_latitude).abs()
    return data

def process(data_chunk):
    data_chunk = data_chunk.dropna(how='any')

    data_chunk['pickup_datetime'] = pd.to_datetime(data_chunk['pickup_datetime'], infer_datetime_format=True)
    data_chunk = date_manipulation(data_chunk, 'pickup_datetime')

    # calculate the distance travelled in KM
    data_chunk['distance_km'] = parallelize_dataframe(data_chunk, distance)

    data_chunk = parallelize_dataframe(data_chunk, add_travel_vector_features)

    # convert the lat and lon into 3d space
    data_chunk = parallelize_dataframe(data_chunk, three_dimensional_transform)#three_dimensional_transform(data_chunk)

    dropable_columns = [
        'month', 'day', 'hour', 'minute', 
        'pickup_datetime', 'pickup_latitude',
        'pickup_longitude', 'dropoff_latitude',
        'dropoff_longitude']

    data_chunk = data_chunk.drop(dropable_columns, axis=1)
   
    return data_chunk 


chunksize = 10 ** 6
filename = '/media/jonathan/HDD/data/new-york-city-taxi-fare/'
train_csv = 'train.csv'
test_csv = 'test.csv'
save_location = '/media/jonathan/HDD/data/new-york-city-taxi-fare/feature-engineered-csvs/'

if __name__ == "__main__":
    debug = False
    main_start_time = time.time()
    
    print('Processing training data...')

    data = pd.read_csv(filename + train_csv, nrows=1000000)
    data = process(data)
    data.to_csv(save_location + 'train/train_features.csv', sep=',', index=False)

    data = pd.read_csv(filename + test_csv, nrows=1000000)
    data = process(data)
    data.to_csv(save_location + 'test/test_features.csv', sep=',', index=False)
    
    # count = 1
    # for chunk in pd.read_csv(filename + train_csv, chunksize=chunksize, nrows=10000000):
    #     chunk_start_time = time.time()
    #     processed_data = process(chunk)
    #     chunk_end_time = time.time()

    #     time_taken = round((chunk_end_time - chunk_start_time), 2)
    #     print('Chunk %s process time (seconds): %s' % (count, time_taken))

    #     saved_file_name = save_location + ('train/training_features_%s.csv' % count)
    #     chunk.to_csv(saved_file_name, sep=',')

    #     count += 1

    #     if (debug):
    #         break

    # print('\nProcessing test data...')

    # count = 1
    # for chunk in pd.read_csv(filename + test_csv, chunksize=chunksize):
    #     chunk_start_time = time.time()
    #     processed_data = process(chunk)
    #     chunk_end_time = time.time()

    #     time_taken = round((chunk_end_time - chunk_start_time), 2)
    #     print('Chunk %s process time (seconds): %s' % (count, time_taken))

    #     saved_file_name = save_location + ('test/test_features_%s.csv' % count)
    #     chunk.to_csv(saved_file_name, sep=',')

    #     count += 1

    #     if (debug):
    #         break

    main_end_time = time.time()
    total_time = round((main_end_time - main_start_time), 2)
    print('\nTotal process time (seconds): %s' % (total_time))
