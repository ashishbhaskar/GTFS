# -*- coding: utf-8 -*-
"""
Created on Fri Feb 16 13:00:23 2024

@author: gozalid
"""

import pandas as pd
from zipfile import ZipFile
import glob
import datetime
import time
import os
pd.set_option('display.max_columns', None)
from pyproj import Geod
wgs84_geod = Geod(ellps='WGS84')
from shapely.geometry import Point, LineString
import geopandas as gpd
from ast import literal_eval
import numpy as np
from shapely.ops import nearest_points, linemerge
from scipy.spatial import cKDTree
import itertools
from operator import itemgetter
import seaborn as sns; sns.set()
import os.path
import sqlalchemy as sql
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore")
    
high_frequency_buses = ['60', '61', '100', '111' ,'120', '130' ,'140', '150', '180', '196' , '199' ,'200' ,'222', '330', '333' ,'340', '345', '385', '412', '444', '555']
Acquired_Directory = r'Y:\\Data\\GTFS_NEW\\'
SELECTED_DATE = datetime.datetime.now()

Working_Directory = r"Y:\\Data\\GTFS_NEW\\"
Static_Folder = Working_Directory + 'GTFS Static\\'
Realtime_folder  = Working_Directory + 'GTFS Realtime\\'
if not os.path.exists(Static_Folder):
    os.makedirs(Static_Folder)
if not os.path.exists(Realtime_folder):
    os.makedirs(Realtime_folder)
gtfs_realtime_link = r'https://gtfsrt.api.translink.com.au/Feed/SEQ'
gtfs_static_link = r"https://gtfsrt.api.translink.com.au/GTFS/SEQ_GTFS.zip"

def shaper_mapper(x):
    d = {}
    d['distance'] = x['distance'].sum()
    d['shape_pt_lat_x']=x['shape_pt_lat_x'].head(1).iloc[0]
    d['shape_pt_lon_x']=x['shape_pt_lon_x'].head(1).iloc[0]
    d['shape_pt_lat_y']=x['shape_pt_lat_y'].tail(2).iloc[0]
    d['shape_pt_lon_y']=x['shape_pt_lon_y'].tail(2).iloc[0]
    d['routes_list']=list(zip(x['shape_pt_lon_x'], x['shape_pt_lat_x']))
    return pd.Series(d, index=['distance', 'shape_pt_lat_x','shape_pt_lon_x', 'shape_pt_lat_y', 'shape_pt_lon_y', 'routes_list'])

def bus_route_path(x, gdf, cols):
    routes = x['routes_list']
    # idxs = [i for i in range(1, len(routes))]
    tempdf = pd.DataFrame(list(enumerate(routes)), columns=['index', 'points'])
    geo = [Point(x) for x in tempdf['points']]
    temp_gpd = gpd.GeoDataFrame(tempdf, crs='EPSG:4376', geometry=geo)
    c_near = ckdnearest(temp_gpd, gdf, cols)
    return ' | '.join(c_near.iloc[:, 3].unique())
    
def get_all_routes(date_name):
    print("getting all routes.....")
    print(f"{Static_Folder}GTFS Static {date_name}.zip")
    zip_file = ZipFile(Static_Folder + 'GTFS Static ' +date_name + '.zip')
    routes = pd.read_csv(zip_file.open('routes.txt'))
    unique_routes = list(routes.route_id.unique())
    print("unique routes = ", len(unique_routes))
    return unique_routes

def contains_number(string):
    return any(char.isdigit() for char in string)

def distance_calc(x):
    lat1 = x['shape_pt_lat_x']
    lon1 = x['shape_pt_lon_x']
    lat2 = x['shape_pt_lat_y']
    lon2 = x['shape_pt_lon_y']
    az12 ,az21,dist = wgs84_geod.inv(lon1,lat1,lon2,lat2) #Yes, this order is correct
    x['distance'] = dist / 1000 ## convert from metre to km
    x['routes_list'] = [(lon1, lat1)]
    return x

def to_float(result):
    newresult = []
    for tuple in result:
        temp = []
        for x in tuple:
            if x.isalpha():
                temp.append(x)
            elif x.isdigit():
                temp.append(int(x))
            else:
                temp.append(float(x))
        newresult.append((temp[0],temp[1]))
    return newresult

def save_shapefile(x):
    routes = x['routes_list']
    new_routes = to_float(routes)
    try:
        line_string_shape = LineString(new_routes)
    except:
        line_string_shape = Point(new_routes[0])
    return line_string_shape

def static_merger(date_name):
    print("static merger fetched..........")
    zip_file = ZipFile(Static_Folder + '/GTFS Static ' +date_name + '.zip')
    trips = pd.read_csv(zip_file.open('trips.txt'))
    stop_times = pd.read_csv(zip_file.open('stop_times.txt'))
    stops = pd.read_csv(zip_file.open('stops.txt'))
    routes = pd.read_csv(zip_file.open('routes.txt'))
    calendar = pd.read_csv(zip_file.open('calendar.txt'))
    calendar_dates = pd.read_csv(zip_file.open('calendar_dates.txt'))
    
    stop_times['stop_id'] = stop_times['stop_id'].astype(str)
    static_df = stop_times.merge(stops, on='stop_id', how='inner')
    static_df = trips.merge(static_df, on='trip_id', how='inner')
    static_df = static_df.merge(routes, on='route_id', how='inner')
    static_df = static_df.merge(calendar, on='service_id', how='inner')
    static_df = static_df.merge(calendar_dates, on='service_id', how='left')
    print(static_df.shape)
    return static_df

def load_new_static(bus_num, date_name, Static_month, Static_Folder):
    print("Processing Static GTFS for date = ", date_name)
    zip_file = ZipFile(Static_Folder + '/GTFS Static ' +date_name + '.zip')
    trips = pd.read_csv(zip_file.open('trips.txt'))
    stop_times = pd.read_csv(zip_file.open('stop_times.txt'))
    shapes = pd.read_csv(zip_file.open('shapes.txt'), sep=',', index_col=False, dtype='unicode')
    shapes['shape_pt_sequence'] = shapes['shape_pt_sequence'].astype(str)
    Route_Shape_stop = stop_times.merge(trips, on = 'trip_id', how = 'left')
    Route_Shape_stop = Route_Shape_stop[['shape_id', 'stop_id', 'stop_sequence', 'route_id', 'trip_id']].drop_duplicates(keep = 'first')
    Route_Shape_stop['stop_sequence'] = Route_Shape_stop['stop_sequence'].astype(int)
    Route_Shape_stop['stop_id'] = Route_Shape_stop['stop_id'].astype(str)
    shapes2 = shapes.copy(deep = True)
    shapes2['shape_pt_sequence_next'] = (shapes2['shape_pt_sequence'].astype(int) + 1).astype(str)
    shapes2['stop_seq_1'] = (shapes2['shape_pt_sequence'].astype(int)/10000).astype(int)
    shapes2['stop_seq_2'] = (shapes2['shape_pt_sequence_next'].astype(int)/10000).astype(int)
    df = shapes2.merge(shapes, left_on = ['shape_id', 'shape_pt_sequence_next'], right_on = ['shape_id', 'shape_pt_sequence'], how = 'left')
    all_static_df_5 = pd.DataFrame()
    new_df = df.copy(deep=True)
    new_df['bus_num'] = new_df.shape_id.apply(lambda x: str(x)[:-4])
    Static_Date = Static_month + "Static " + date_name + "\\"
    if not os.path.exists(Static_Date):
        os.makedirs(Static_Date)
    for bus in bus_num:
        fname = Static_Date+bus+"_full_static_data.csv"
        if os.path.isfile(fname):
            pass
        else:
            temp_df = new_df.loc[new_df['bus_num'] == bus]
            if len(temp_df) < 1:
                continue
            temp_df = temp_df.apply(distance_calc, axis=1)
            temp_df.to_csv(Static_month+"temp_df_distance_route" + date_name + ".csv")
            temp_df = temp_df.loc[temp_df['stop_seq_1'] == temp_df['stop_seq_2']]
            temp_df['stop_seq_2'] = temp_df['stop_seq_2'] + 1
            shapes5 = temp_df.groupby(['shape_id', 'stop_seq_1', 'stop_seq_2'])['distance'].sum().reset_index()
            temp_df = temp_df.groupby(['shape_id', 'stop_seq_1', 'stop_seq_2']).agg({'distance': 'sum', 'routes_list':'sum', 'shape_pt_lat_x': 'first', 'shape_pt_lon_x': 'first', 'shape_pt_lat_y': 'last', 'shape_pt_lon_y': 'last'}).reset_index()
            temp_df['shape_file_name'] = temp_df.apply(save_shapefile, axis=1)
            temp_df['shape_id'] = temp_df['shape_id'].astype(str)
            temp_df.to_csv(Static_Date+bus+"_full_static_data.csv")
            all_static_df_5 = pd.concat([all_static_df_5, shapes5], ignore_index=True)
    return all_static_df_5

def remove_unnamed(df):
    list_unnameds = [s for s in df.columns if 'Unnamed:' in s]
    if len(list_unnameds) > 0:
        df.drop(list_unnameds, axis=1, inplace=True)
    return df

def apply_lambda(static_sample, x):
    if len(list(static_sample.loc[static_sample.stop_sequence == x].stop_id.unique())) > 0:
        return list(static_sample.loc[static_sample.stop_sequence == x].stop_id.unique())[0]
    else:
        return None
    
def ckdnearest(gdfA, gdfB, gdfB_cols):
    """
    Mapping the point to the closest linestring
    Parameters
    ----------
    gdfA : TYPE
        DESCRIPTION.
    gdfB : TYPE
        DESCRIPTION.
    gdfB_cols : TYPE, optional
        DESCRIPTION. The default is ['Name', 'direction_'].

    Returns
    -------
    gdf : TYPE
        DESCRIPTION.

    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
        A = np.concatenate(
            [np.array(geom.coords) for geom in gdfA.geometry.to_list()])
        B = [np.array(geom.coords) for geom in gdfB.geometry.to_list()]
        
    B_ix = tuple(itertools.chain.from_iterable(
        [itertools.repeat(i, x) for i, x in enumerate(list(map(len, B)))]))
    B = np.concatenate(B)
    ckd_tree = cKDTree(B)
    dist, idx = ckd_tree.query(A, k=1)    
    idx = itemgetter(*idx)(B_ix)
    gdf = pd.concat(
        [gdfA, gdfB.loc[idx, gdfB_cols].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

def convert_to_float(x):
    return float(x[0]), float(x[1])

def bus_routes_paths_x(x, gdf, cols):
    routes = literal_eval(x['routes_list'])
    routes = tuple(map(convert_to_float, routes))
    bus_num = x['bus_num']
    idxs = [i for i in range(1, len(routes))]
    tempdf = pd.DataFrame(list(zip(idxs, routes)), columns=['index', 'points'])
    geo = [Point(x) for x in tempdf['points']]
    temp_gpd = gpd.GeoDataFrame(tempdf, crs='EPSG:4376', geometry=geo)
    static_gdf = gdf.loc[gdf['Associate'].str.contains(bus_num)].reset_index()
    if len(static_gdf) < 1:
        static_gdf = gdf.copy(deep=True)
    c_near = ckdnearest(temp_gpd, static_gdf, cols)
    return list(c_near.iloc[:, 3].unique())
    
def matrix_mapping(all_static_buses_df, gdf, cols, is_static=False):
    inbound_static_buses = all_static_buses_df.loc[all_static_buses_df.direction_id == 0]
    outbound_static_buses = all_static_buses_df.loc[all_static_buses_df.direction_id == 1]
    inbound_gdf = gdf.loc[gdf.direction_id == 0]
    outbound_gdf = gdf.loc[gdf.direction_id == 1]
    inbound_gdf = inbound_gdf.reset_index()
    inbound_static_buses = inbound_static_buses.reset_index()
    c_inbound = ckdnearest(inbound_static_buses, inbound_gdf, cols)
    outbound_static_buses = outbound_static_buses.reset_index()
    outbound_gdf = outbound_gdf.reset_index()
    c_outbound = ckdnearest(outbound_static_buses, outbound_gdf, cols)
    if is_static:
        c_inbound['names_list'] = c_inbound.apply(lambda x: bus_routes_paths_x(x, inbound_gdf, cols), axis=1)
        c_outbound['names_list'] = c_outbound.apply(lambda x: bus_routes_paths_x(x, outbound_gdf, cols), axis=1)
    c_all = pd.concat([c_inbound, c_outbound], ignore_index=True)
    # print(c_all)
    c_all_line = c_all.merge(gdf, on=['Name'], how='inner')
    return c_all_line

def static_filtering(date_name, shape_sample, route_sample, bus_num, static_sample, Static_month):
    from shapely import wkt
    try:
        distance_static = pd.read_csv(Static_month + "Static " + date_name + "\\" + bus_num + "_full_static_data.csv")
        static_sample = remove_unnamed(static_sample)
        distance_static = remove_unnamed(distance_static)

        distance_static['shape_id'] = distance_static['shape_id'].astype(str)
        distance_static = distance_static.loc[distance_static.shape_id == shape_sample]
        dist_df = distance_static.merge(static_sample[['stop_id', 'stop_sequence', 'direction_id', 'arrival_time']], left_on=['stop_seq_1'], right_on=['stop_sequence'], how='inner')
        dist_df_grouped = dist_df.groupby(['shape_id', 'stop_seq_1', 'stop_seq_2'])
        dist_df = dist_df_grouped.agg(shape_pt_lat_x=('shape_pt_lat_x', 'first'), 
                                      shape_pt_lon_x=('shape_pt_lon_x', 'first'), 
                                      shape_pt_lat_y=('shape_pt_lat_y', 'last'), 
                                      shape_pt_lon_y=('shape_pt_lon_y', 'last'), 
                                      distance=('distance', 'first'), 
                                      routes_list=('routes_list', 'first'), 
                                      shape_file_name=('shape_file_name', 'first'),
                                      direction_id=('direction_id', 'first'), 
                                      arrival_time=('arrival_time', 'first')).reset_index()
        #########################################################
        dist_df['stop_id'] = dist_df['stop_seq_1'].apply(lambda x: list(static_sample.loc[static_sample.stop_sequence == x].stop_id.unique())[0])
        dist_df['stop_name'] = dist_df['stop_seq_1'].apply(lambda x: list(static_sample.loc[static_sample.stop_sequence == x].stop_name.unique())[0])
        dist_df['from_stop_id'] = dist_df['stop_seq_1'].apply(lambda x: list(static_sample.loc[static_sample.stop_sequence == x].stop_id.unique())[0])
        dist_df['to_stop_id'] = dist_df['stop_seq_2'].apply(lambda x: apply_lambda(static_sample, x))
        dist_df['line_geometry'] = dist_df['shape_file_name'].apply(wkt.loads)
        
        return dist_df
    except (pd.errors.EmptyDataError, FileNotFoundError):
        print("no data")
        return pd.DataFrame()
    
def static_matrix_mapping(all_static_buses_df, gdf):
    all_static_buses_df['bus_num'] = all_static_buses_df['shape_id'].apply(lambda x: x[:-4])
    all_static_buses_df['point'] = all_static_buses_df[['shape_pt_lon_x', 'shape_pt_lat_x']].apply(lambda x: Point(x['shape_pt_lon_x'], x['shape_pt_lat_x']), axis=1)
    geo = [Point(x) for x in all_static_buses_df['point']]
    all_static_gdf = gpd.GeoDataFrame(all_static_buses_df, crs='EPSG:4376', geometry=geo)
    all_static_gdf = gpd.GeoDataFrame(all_static_buses_df, crs='EPSG:4376', geometry=[Point(x) for x in all_static_buses_df['point']])
    gdf.rename(columns={'direction_':'direction_id', 'Associated':'Associate', 'Length' :'length'}, inplace=True)
    c_all_line = matrix_mapping(all_static_gdf, gdf, cols=['Name', 'direction_id'], is_static=True)
    c_all_line = c_all_line.drop(['index', 'direction_id_x'], axis=1)
    new_gdf = c_all_line.groupby(['Name']).agg(Associat_1=('shape_id', set), Associate=('bus_num', set), length=('length', 'first'), direction_id=('direction_id_y', 'first'), geometry=('geometry_y', 'first')).reset_index()
    new_gdf['Associate'] = new_gdf.Associate.apply(lambda xs: ' | '.join(str(x) for x in xs))
    new_gdf['Associat_1'] = new_gdf.Associat_1.apply(lambda xs: ' | '.join(str(x) for x in xs))
    gddf = gpd.GeoDataFrame(new_gdf, crs='EPSG:4376', geometry=[LineString(x) for x in new_gdf['geometry']])
    non_gdf = gdf[~gdf['Name'].isin(gddf['Name'])]
    non_gdf.rename(columns={'direction_':'direction_id', 'Associated':'Associate', 'Length' :'length'}, inplace=True)
    merged_gdf = gpd.GeoDataFrame(pd.concat([gddf, non_gdf], ignore_index=True))
    c_all_line.rename(columns={"geometry_x":"point_geometry", "geometry_y" : "linestring_geometry", "direction_id_x" : "direction_id", "length_x": "length"}, inplace=True)
    return c_all_line, merged_gdf

def fetch_data_from_hetrogen(date_name, month_name):
    from datetime import datetime
    data_path = f"{Realtime_folder}VehiclePosition entity\\VP {month_name}\\VP {date_name}"
    vp_csv_files = glob.glob(data_path + "\\*.csv")
    all_df = pd.DataFrame()
    for file in vp_csv_files:
        df = pd.read_csv(file)
        all_df = pd.concat([all_df, df], ignore_index=True)
    if len(all_df) < 1:
        return all_df
    all_df = all_df.dropna()
    sample_vp_trip_df_copy = all_df.copy(deep=True)
    
    ## Convert arrival and departure time into datetime format
    sample_vp_trip_df_copy['timestamp_dt'] = sample_vp_trip_df_copy['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    
    if sample_vp_trip_df_copy['stop_id'].dtypes == 'float64':
        sample_vp_trip_df_copy['stop_id'] = sample_vp_trip_df_copy['stop_id'].astype(int)
        sample_vp_trip_df_copy['stop_id'] = sample_vp_trip_df_copy['stop_id'].astype(str)
    elif sample_vp_trip_df_copy['stop_id'].dtypes == 'int64':
        sample_vp_trip_df_copy['stop_id'] = sample_vp_trip_df_copy['stop_id'].astype(str)
    else:
        sample_vp_trip_df_copy['stop_id'] = sample_vp_trip_df_copy['stop_id'].astype(str)
        
    list_unnameds = [s for s in sample_vp_trip_df_copy.columns if 'Unnamed:' in s]
    if len(list_unnameds) > 0:
        sample_vp_trip_df_copy.drop(list_unnameds, axis=1, inplace=True)
        
    zip_file = ZipFile(Static_Folder + 'GTFS Static ' +date_name + '.zip')
    trips = pd.read_csv(zip_file.open('trips.txt'))
    stop_times = pd.read_csv(zip_file.open('stop_times.txt'))
    stops = pd.read_csv(zip_file.open('stops.txt'))
    
    trips_df = trips[['route_id', 'trip_id', 'shape_id', 'direction_id']]
    stop_times_df = stop_times[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']]
    stops_df = stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
    
    
    sample_vp_trip_df_non_duplicates = sample_vp_trip_df_copy.drop_duplicates(subset=['timestamp', 'id'], keep='first')
    sample_vp_trip_df_non_duplicates = sample_vp_trip_df_non_duplicates.merge(trips_df, on=['route_id', 'trip_id'], how='inner')
    stop_times_df['stop_id'] = stop_times_df['stop_id'].astype(str)
    sample_vp_trip_df_non_duplicates = sample_vp_trip_df_non_duplicates.merge(stop_times_df, on=['trip_id', 'stop_id'], how='inner')
    sample_vp_trip_df_non_duplicates = sample_vp_trip_df_non_duplicates.merge(stops_df, on=['stop_id'], how='inner')
    
    return sample_vp_trip_df_non_duplicates

def fetch_data_from_sql_remote(query, db='gtfs_v'):
    start = time.time()
    print("fetching data from remote sql.....")
    mysql_account = os.environ.get("mysql_account")
    try:
        data = pd.read_sql_query(query, f'mysql://{mysql_account}' + str(db))
        end = time.time()
        print(f"sql query fetching data executed in: {str((end - start) / 60)} mins")
        print("sql remote data = ", data.shape)
        return data
    except (pd.errors.DatabaseError, sql.exc.ProgrammingError):
        return pd.DataFrame()
    
def vp_matrix_mapping(vp_df, static_stops, static_gdf):
    vp_df['current_point'] = vp_df[['latitude', 'longitude']].apply(lambda x: Point(x['longitude'], x['latitude']), axis=1)
    vp_gdf = gpd.GeoDataFrame(vp_df, crs='EPSG:4376', geometry=[Point(x) for x in vp_df['current_point']])
    static_gdf.rename(columns={'direction_':'direction_id', 'Associated':'Associate', 'Length' :'length'}, inplace=True)
    vp_line = matrix_mapping(vp_gdf, static_gdf, cols=['Name', 'length'])
    vp_line.rename(columns={"Name": "Link Name", "direction_id_x": "direction_id", "length_x": "link_length", "dist": "link_dist"}, inplace=True)
    vp_line = vp_line.drop(['Associate', 'Associat_1', "geometry_y"], axis=1)
    static_stops.rename(columns={'direction_id_y': 'direction_id'}, inplace=True)
    vp_line['stop_id'] = vp_line['stop_id'].astype(str)
    vp_line['shape_id'] = vp_line['shape_id'].astype(str)
    static_stops['stop_id'] = static_stops['stop_id'].astype(str)
    static_stops['shape_id'] = static_stops['shape_id'].astype(str)
    vp_gdf_2 = vp_line.merge(static_stops[['shape_id', 'distance', 'stop_id', 'Name', 'names_list', 'length', 'direction_id', 'linestring_geometry']], on=['shape_id', 'stop_id', 'direction_id'], how='inner')
    vp_gdf_2.rename(columns={"distance": "stop_distance", "Name": "Actual_Link_Name", "length": "Actual_length"}, inplace=True)
    return vp_gdf_2

def stop_geometry(x):
    new_stop = x[['stop_seq_1','shape_pt_lat_x','shape_pt_lon_x','distance', 'routes_list', 'shape_file_name', 'direction_id', 'stop_id','stop_name', 'bus_num','point_geometry', 'linestring_geometry', 'Name', 'names_list', 'arrival_time', 'line_geometry']]
    a = x.iloc[[-1]][['stop_seq_2','shape_pt_lat_y','shape_pt_lon_y','distance', 'routes_list', 'shape_file_name', 'direction_id', 'to_stop_id','stop_name', 'bus_num', 'point_geometry', 'linestring_geometry', 'Name', 'names_list', 'arrival_time', 'line_geometry']]
    new_stop.rename(columns={'stop_seq_1':'stop_sequence', 'shape_pt_lat_x':'shape_pt_lat','shape_pt_lon_x': 'shape_pt_lon'}, inplace=True)
    a.rename(columns={'stop_seq_2':'stop_sequence', 'shape_pt_lat_y':'shape_pt_lat', 'shape_pt_lon_y': 'shape_pt_lon', 'to_stop_id': 'stop_id'}, inplace=True)
    new_stop = pd.concat([new_stop, a], ignore_index=True)
    new_stop_shift = new_stop.copy(deep=True)
    new_stop_shift[['distance', 'routes_list', 'shape_file_name', 'linestring_geometry', 'line_geometry']] = new_stop_shift[['distance', 'routes_list', 'shape_file_name', 'linestring_geometry', 'line_geometry']].shift(1)
    new_stop_shift.loc[0, ['distance', 'routes_list', 'shape_file_name', 'linestring_geometry', 'line_geometry']] = new_stop_shift.loc[1, ['distance', 'routes_list', 'shape_file_name', 'linestring_geometry', 'line_geometry']]
    new_stop_shift.iloc[0, new_stop_shift.columns.get_loc('distance')] = 0
    new_stop_shift.iloc[-1, new_stop_shift.columns.get_loc('Name')] = new_stop.iloc[-1]['names_list'][-1]
    return new_stop_shift

def get_sort_data(x, stop_df, gdf, date_name):
    data = x[['Link Name', 'timestamp', 'stop_id', 'stop_sequence', 'current_point', 'link_length', 'stop_distance', 'linestring_geometry', 'arrival_time', 'names_list']]
    sorted_data = data.sort_values(by=['timestamp'])
    actual_line = LineString(stop_df['routes_list'].apply(lambda x: tuple(map(convert_to_float, literal_eval(x)))).sum())
    sorted_data['distance'] = sorted_data['current_point'].apply(lambda x: actual_line.project(nearest_points(actual_line, x)[0]) * 100)
    sorted_data['cumulative_space'] = sorted_data['distance'].diff().cumsum()
    sorted_data['cumulative_space'] = sorted_data['cumulative_space'].fillna(0)
    sorted_data['travel_time'] = abs(sorted_data['timestamp'].shift(-1) - sorted_data['timestamp'])
    sorted_data['travel_time'] = sorted_data['travel_time'].shift(1)
    sorted_data.iloc[0, sorted_data.columns.get_loc('travel_time')] = 0
    sorted_data['cumulative_tt'] = sorted_data['travel_time'].cumsum()
    sorted_data['scheduled_arrival_dt'] = (pd.to_datetime(date_name, format='%d-%m-%Y') + pd.to_timedelta(sorted_data['arrival_time']))
    sorted_data['scheduled_timestamp'] = sorted_data['scheduled_arrival_dt'].apply(lambda x: datetime.datetime.timestamp(x))
    stop_df['scheduled_arrival_dt'] = (pd.to_datetime(date_name, format='%d-%m-%Y') + pd.to_timedelta(stop_df['arrival_time']))
    stop_df['scheduled_timestamp'] = stop_df['scheduled_arrival_dt'].apply(lambda x: datetime.datetime.timestamp(x))
    stop_df['scheduled_tt'] = abs(stop_df['scheduled_timestamp'].shift(-1) - stop_df['scheduled_timestamp'])
    stop_df['scheduled_tt'] = stop_df['scheduled_tt'].shift(1)
    stop_df.iloc[0, stop_df.columns.get_loc('scheduled_tt')] = 0
    stop_df['cumulative_scheduled_tt'] = stop_df['scheduled_tt'].cumsum()
    stop_df['cumulative_space'] = stop_df['distance'].cumsum()
    return sorted_data, stop_df

def get_cumulative_plot(sorted_data, stop_df):
    actual_cum_plot = sorted_data[['timestamp', 'cumulative_space', 'cumulative_tt', 'stop_sequence']].copy(deep=True)
    actual_cum_plot.rename(columns={'cumulative_space':'space', 'cumulative_tt':'tt'}, inplace=True)
    sch_cum_plot = stop_df.groupby(['stop_sequence']).agg(sch_space=('distance', 'last'), sch_tt=('cumulative_scheduled_tt', 'last'), scheduled_timestamp=('scheduled_timestamp', 'last'), scheduled_arr_dt=('scheduled_arrival_dt', 'last'), bus_num=('bus_num', 'first'))
    sch_cum_plot = sch_cum_plot.reset_index()
    sch_cum_plot['sch_space'] = sch_cum_plot['sch_space'].cumsum()
    return actual_cum_plot, sch_cum_plot

def flatten_link_names(x):
    """
    flatten the list of names lists to generate a single list of items
    """
    flat_list = []
    q_list = []
    a = x.copy()
    for sublist in a:
        if isinstance(sublist, str):
            sublist = literal_eval(sublist)
        if not sublist in q_list:
            q_list.append(sublist)
            for item in sublist[:-1]:
                if item not in flat_list:
                    flat_list.append(item)
    return flat_list

def intersection_analysis_new(x, gdf, stop_stats_df, all_static_df, date_name):
    """
    Generate the intersection cumulative plot for various tripID and output new dataframe for list of Link Name

    Parameters
    ----------
    x : Dataframe for a particular tripID
        DESCRIPTION.
    gdf : GeoDataFrame
        HFS intersection level shapefile.

    Returns Dataframe with full link-to-link from first stop to the last stop before terminus
    -------
    DataFrame
        Full Dataframe of Link-to-Link From start to end of trip based on interpolated values.

    """
    stop_df = stop_stats_df.loc[(stop_stats_df['shape_id'] == x['shape_id'].unique()[0]) & (stop_stats_df['direction_id'] == x['direction_id'].unique()[0])]
    sorted_data, stop_df = get_sort_data(x, stop_df, gdf, date_name)
    actual_cum_plot, sch_cum_plot = get_cumulative_plot(sorted_data, stop_df)
    if len(actual_cum_plot['stop_sequence'].unique()) < 2:
        return None
    list_links = flatten_link_names(stop_df['names_list'])
    link_df = pd.DataFrame(list_links, columns=['Name'])
    link_df = link_df.merge(gdf[['Name', 'length']], on=['Name'], how='inner')
    link_df.rename(columns={"Name": "Link Name", "length": "link_length"}, inplace=True)
    link_df['cum_space'] = link_df['link_length'].cumsum()
    list_links = flatten_link_names(sorted_data['names_list'])
    link_df = link_df.loc[link_df['Link Name'].isin(list_links)]
    UNDEF = np.nan
    link_df['timestamp'] = np.interp(link_df['cum_space'], actual_cum_plot['space'], actual_cum_plot['timestamp'], right=UNDEF)
    link_df['scheduled_timestamp'] = np.interp(link_df['cum_space'], sch_cum_plot['sch_space'], sch_cum_plot['scheduled_timestamp'], right=UNDEF)
    link_df['actual_cum_tt'] = np.interp(link_df['cum_space'], actual_cum_plot['space'], actual_cum_plot['tt'], right=UNDEF)
    link_df['expected_cum_tt'] = np.interp(link_df['cum_space'], sch_cum_plot['sch_space'], sch_cum_plot['sch_tt'], right=UNDEF)
    link_df = link_df.dropna()
    if len(link_df) < 1:
        return None
    link_df['travel_time'] = link_df['actual_cum_tt'] - link_df['actual_cum_tt'].shift(1)
    link_df = link_df.fillna(link_df.iloc[0]['actual_cum_tt'])
    link_df['expected_tt'] = link_df['expected_cum_tt'] - link_df['expected_cum_tt'].shift(1)
    link_df = link_df.fillna(link_df.iloc[0]['expected_cum_tt'])
    link_df['delay'] = link_df['travel_time'] - link_df['expected_tt']
    link_df_drop = link_df.dropna(how='any', axis=0)
    link_df_drop['speed'] = link_df_drop['link_length'] / (link_df_drop['travel_time'] / 3600)
    link_df_drop.loc[link_df_drop.speed > 120, ['speed', 'travel_time', 'delay']] = np.nan
    link_df_drop['travel_time'] = link_df_drop['travel_time'].interpolate(method='linear')
    link_df_drop['delay'] = link_df_drop['travel_time'] - link_df_drop['expected_tt']
    link_df_drop['speed'] = link_df_drop['link_length'] / (link_df_drop['travel_time'] / 3600)
    df = link_df_drop.drop(link_df_drop[link_df_drop.speed > 120].index)
    return df

def speed_agg_link_analysis(date_name, static_stops, static_gdf, all_static_df, month_name): 
    start = time.time()
    month, year = month_name.split(' ,')
    month = month.lower()
    zip_file = ZipFile(Static_Folder + 'GTFS Static ' +date_name + '.zip')
    data = fetch_data_from_sql_remote(f"select * from `{month}_{year}` where DATE(FROM_UNIXTIME(`timestamp`)) = str_to_date('{date_name}', '%%d-%%m-%%Y')");
    stop_times = pd.read_csv(zip_file.open('stop_times.txt'))
    stop_times_df = stop_times[['trip_id', 'arrival_time', 'departure_time', 'stop_id', 'stop_sequence']]
    stop_times_df['stop_id'] = stop_times_df['stop_id'].astype(str)
    if len(data) > 0:
        data = data.drop(['index'], axis=1)
        data['timestamp_dt'] = data['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
        
        data = data.drop_duplicates(subset=['timestamp', 'id'], keep='first')
        
        trips = pd.read_csv(zip_file.open('trips.txt'))
        stops = pd.read_csv(zip_file.open('stops.txt'))
        
        trips_df = trips[['route_id', 'trip_id', 'shape_id', 'direction_id']]
        stops_df = stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']]
        
        data = data.merge(trips_df, on=['route_id', 'trip_id'], how='inner')
        data = data.merge(stop_times_df, on=['trip_id', 'stop_id'], how='inner')
        data = data.merge(stops_df, on=['stop_id'], how='inner')
    else:
        data = fetch_data_from_hetrogen(date_name, month_name)    
    if len(data) < 1:
        return pd.DataFrame(columns=['trip_id', 'Link Name', 'link_length', 'cum_space', 'timestamp', 'scheduled_timestamp', 'actual_cum_tt', 'expected_cum_tt', 'travel_time', 'expected_tt', 'delay', 'speed', 'dt'])
    data_new = data.loc[(data['latitude'].notnull()) & (data['longitude'].notnull())]
    data_new = vp_matrix_mapping(data_new, static_stops, static_gdf)
    static_stops['stop_id'] = static_stops['stop_id'].astype(str)
    static_stops = static_stops.sort_values(by=['shape_id', 'stop_seq_1'])
    static_stops = static_stops.drop_duplicates(subset=['shape_id', 'stop_seq_1', 'stop_seq_2'], keep='first')
    stop_stats_df = static_stops.groupby(['shape_id']).apply(stop_geometry).reset_index()
    stop_stats_df = stop_stats_df.drop(['level_1'],axis=1)
    sorted_df_merged = data_new.groupby(['trip_id']).apply(intersection_analysis_new, gdf=static_gdf, stop_stats_df=stop_stats_df, all_static_df=stop_times_df, date_name=date_name)
    sorted_df_merged = sorted_df_merged.dropna()
    if len(sorted_df_merged) > 0:
        sorted_df_merged['dt'] = sorted_df_merged['timestamp'].apply(lambda x: datetime.datetime.fromtimestamp(x))
    end = time.time()
    print(f"The duration it takes to execute for the whole day intersection analysis = {(end - start)/60} mins")
    return sorted_df_merged
    
def speed_trip_trajectory_preprocessing_analysis(SELECTED_DATE):
    date_name = str(SELECTED_DATE.day)+"-"+str(SELECTED_DATE.month)+"-"+str(SELECTED_DATE.year)
    month_name = SELECTED_DATE.strftime("%B") + " ," +str(SELECTED_DATE.year)
    print(date_name)
    print("HELLO FROM SPEED!!")
    Working_Directory = r"Y:\\Sentosa\\GTFS_preprocessed\\gtfs\\Saved_Dirs\\"
    Realtime_folder = Working_Directory + 'GTFS Realtime\\'
    Static_folder = Working_Directory + 'GTFS Static\\'
    Static_month = Static_folder + 'Static '+ month_name + '\\'
    if not os.path.exists(Static_month):
        os.makedirs(Static_month)
    ######## HIGH FREQUENCY BUSES ANALYSIS #################
    HFB_DIR = Working_Directory + "GTFS Realtime\\HFB\\"
    if not os.path.exists(HFB_DIR):
        os.makedirs(HFB_DIR)
    ######## TRIP UPDATE DIRECTORY #########################
    TU_Saved_DIR = HFB_DIR + 'TripUpdate entity\\'
    if not os.path.exists(TU_Saved_DIR):
        os.makedirs(TU_Saved_DIR)
    TU_Saved_DIR2 = TU_Saved_DIR + 'TU ' + month_name + '\\'
    if not os.path.exists(TU_Saved_DIR2):
        os.makedirs(TU_Saved_DIR2)
    ######## VEHICLE POSITION DIRECTORY ####################
    VP_Saved_DIR = HFB_DIR + 'VehiclePosition entity\\'
    if not os.path.exists(VP_Saved_DIR):
        os.makedirs(VP_Saved_DIR)
    VP_Saved_DIR2 = VP_Saved_DIR + 'VP ' + month_name + '\\'
    if not os.path.exists(VP_Saved_DIR2):
        os.makedirs(VP_Saved_DIR2)
    
    HFS_DIR = r"C:\\Users\\gozalid\\OneDrive - Queensland University of Technology (1)\\Shubham-Sentosa\\Transit Dashboards\\Data used\\Rough Work\\Ultra final\\Shape file\\"
    gdf = gpd.read_file(HFS_DIR + "HFS.shp")
    routes = get_all_routes(date_name)
    buses = []
    for route in routes:
        bus_num = route.split('-')[0]
        if contains_number(bus_num):
            buses.append(bus_num)
    dist_file = load_new_static(buses, date_name, Static_month, Static_Folder)
    start = time.time()
    all_static_high_freq_bus = pd.DataFrame()
    all_static_df = static_merger(date_name)
    for route in routes:
        bus_num = route.split('-')[0]
        if bus_num in buses:
            static_df = all_static_df.loc[all_static_df.route_id == route]
            unique_shapes = list(static_df.shape_id.unique())
            static_df_stops_shapes = pd.DataFrame()
            for shape in unique_shapes:
                static_sample = static_df.loc[static_df['shape_id'].astype(str) == str(shape)]
                if len(static_sample) < 1:
                    pass
                static_distance_df = static_filtering(date_name, shape, route, bus_num, static_sample, Static_month)
                static_df_stops_shapes = pd.concat([static_df_stops_shapes, static_distance_df], ignore_index=True)
            # all_static_high_freq_bus = all_static_high_freq_bus.append(static_df_stops_shapes, ignore_index=True)
            all_static_high_freq_bus = pd.concat([all_static_high_freq_bus, static_df_stops_shapes], ignore_index=True)
    end = time.time()
    print(f"The duration it takes to execute for the whole day static mapping = {(end - start)/60} mins")

    print("routing is done....")
    print(f"The duration it takes to execute for the whole day static preprocessing = {(end - start)/60} mins")
    print("static_matrix mapping is starting....")
    static_stops, static_gdf = static_matrix_mapping(all_static_high_freq_bus, gdf)
    start = time.time()
    df_grouped = speed_agg_link_analysis(date_name, static_stops, gdf, all_static_df, month_name)
    end = time.time()
    print(f"The duration it takes to execute for the whole day new_whole day analysis = {(end - start)/60} mins")
    print('-'*100)
    Saved_Worked = f"Y:\\Data\\GTFS_NEW\\GTFS Realtime\\VehicleTrajectory entity\\Traj {month_name}\\"
    if len(df_grouped) < 1:
        df_grouped.to_csv(Saved_Worked + "trip_trajectory_" + date_name + ".csv")
        return static_stops, static_gdf, df_grouped
    df_grouped_reset = df_grouped.reset_index()
    if 'index' in df_grouped_reset:
        df_grouped_reset = df_grouped_reset.drop(['index'], axis=1)
    if 'level_1' in df_grouped_reset:
        df_grouped_reset = df_grouped_reset.drop(['level_1'], axis=1)
    df_grouped_reset.to_csv(Saved_Worked + "trip_trajectory_" + date_name + ".csv")
    return static_stops, static_gdf, df_grouped_reset