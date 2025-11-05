#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Wang Yidi, May 2024                              #
############################################################


import os
import datetime as dt
import numpy as np
import math
from scipy import interpolate
from mintpy.objects import ionex, timeseries
from mintpy.simulation import iono
from mintpy.utils import readfile, writefile
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from mintpy import iono_tec
from mintpy.cli import diff, ifgram_inversion, modify_network, reference_point, reference_date
from mintpy.utils import utils as ut

def compute_lat_lon_ipp(geo_file, utc_sec):
    incidenceAngle = readfile.read(geo_file, datasetName='incidenceAngle')[0]
    incidenceAngle = np.squeeze(incidenceAngle)
    incidenceAngle[incidenceAngle == 0] = np.nan

    azimuthAngle = readfile.read(geo_file, datasetName='azimuthAngle')[0]
    azimuthAngle = np.squeeze(azimuthAngle)
    azimuthAngle[azimuthAngle == 0] = np.nan

    latitude = readfile.read(geo_file, datasetName='latitude')[0]
    latitude = np.squeeze(latitude)
    latitude[latitude == 0] = np.nan

    longitude = readfile.read(geo_file, datasetName='longitude')[0]
    longitude = np.squeeze(longitude)
    longitude[longitude == 0] = np.nan
    
    theta = incidenceAngle*np.pi/180
    Re = 6371000
    h_ipp = 450e3
    theta_ipp = np.arcsin(Re*np.sin(theta)/(Re+h_ipp))
    HEADING = azimuthAngle*np.pi/180

    alpha_ipp = theta - theta_ipp

    latitude_pi = latitude *np.pi/180
    longitude_pi = longitude *np.pi/180

    lat_ipp_pi = np.arcsin(np.sin(latitude_pi)*np.cos(alpha_ipp) + np.cos(latitude_pi)*np.sin(alpha_ipp)*np.cos(HEADING))
    atan2_func = np.vectorize(math.atan2)
    delta = atan2_func(-np.sin(alpha_ipp)*np.cos(latitude_pi)*np.sin(HEADING), np.cos(alpha_ipp) - np.sin(latitude_pi)*np.sin(lat_ipp_pi))
    lon_ipp_pi = np.mod(longitude_pi + delta + np.pi, 2*np.pi) - np.pi

    lat_ipp = lat_ipp_pi *180/np.pi
    lon_ipp = lon_ipp_pi *180/np.pi

    return lat_ipp, lon_ipp ,theta_ipp


def compute_r_iono(utc_sec, tec_file, lat_ipp, lon_ipp , theta_ipp, geo_file,times_path):
    
    mins, lats, lons, tec_maps, rms_maps = ionex.read_ionex(tec_file)
    
    minutes1 = utc_sec / 60
    for i in range(len(mins) - 1):
        if minutes1 >= mins[i] and minutes1 <= mins[i+1]:
            break

    valid_mask = ~(np.isnan(lat_ipp) | np.isnan(lon_ipp))
    valid_lat_ipp = lat_ipp[valid_mask]
    valid_lon_ipp = lon_ipp[valid_mask]

    interp_func = RegularGridInterpolator((mins, lats, lons), tec_maps, method='linear')

    Ei = np.column_stack((np.full(valid_lat_ipp.size, mins[i]),
                          valid_lat_ipp,
                          valid_lon_ipp + (minutes1 - mins[i]) * 360. / (24. * 60.)))

    Ei1 = np.column_stack((np.full(valid_lat_ipp.size, mins[i+1]),
                           valid_lat_ipp,
                           valid_lon_ipp + (minutes1 - mins[i+1]) * 360. / (24. * 60.)))

    new_tec_map1 = np.full_like(lat_ipp, np.nan)
    new_tec_map1[valid_mask] = ((mins[i+1] - minutes1) / (mins[i+1] - mins[i]) * interp_func(Ei) + (minutes1 - mins[i]) / (mins[i+1] - mins[i]) * interp_func(Ei1))

    k = 40.31
    c = 299792458
    meta = readfile.read_attribute(times_path)
    freq = c / float(meta['WAVELENGTH'])
    h_ipp = 450e3
    Re = 6371000
   
    VTEC = new_tec_map1*1e16
    a = VTEC * k / (freq ** 2)
    r_iono = a / np.cos(np.arcsin(np.sin(theta_ipp) / (1 + a)))
    
    return r_iono

def create_iono_timeseries(times_path, tec_dir, geo_file, iono_file):
    
    # download
    date_list = timeseries(times_path).get_date_list()
    tec_files = iono_tec.download_ionex_files(date_list, tec_dir, sol_code='jpl')
    
    # run
    meta = readfile.read_attribute(times_path)
    utc_sec = float(meta['CENTER_LINE_UTC'])
    lat_ipp, lon_ipp ,theta_ipp = compute_lat_lon_ipp(geo_file, utc_sec)
    
    # write
    num_files = len(tec_files)
    meta = readfile.read_attribute(geo_file)
    width = int(meta['WIDTH'])
    length = int(meta['LENGTH'])
    r_iono = np.zeros((num_files, length,width), dtype=np.float32)
    
    for i in tqdm(range(num_files)):
        r_iono[i,:,:] = compute_r_iono(utc_sec, tec_files[i], lat_ipp, lon_ipp , theta_ipp, geo_file,times_path)

    meta = readfile.read_attribute(times_path)
    ref = meta['REF_DATE']
    
    for i_date_ion in range(len(date_list) - 1):
        if ref == date_list[i_date_ion]: 
            break
            
    r_iono[:,:,:] = r_iono[:,:,:] - r_iono[i_date_ion,:,:]

    ref_y = int(meta['REF_Y'])
    ref_x = int(meta['REF_X'])
    r_iono[:,:,:] = r_iono[:,:,:] - r_iono[:,ref_y,ref_x][:, None, None]
    
    # prepare meta
    meta = readfile.read_attribute(times_path)
    meta['FILE_TYPE'] = 'timeseries'
    meta['UNIT'] = 'm'
    # absolute delay without double reference
    #for key in ['REF_X','REF_Y','REF_LAT','REF_LON','REF_DATE']:
        #if key in meta.keys():
            #meta.pop(key)

    ds_dict = {}
    ds_dict['date'] = np.array(date_list, dtype=np.string_)
    ds_dict['timeseries'] = r_iono
    
    writefile.write(ds_dict, iono_file, metadata=meta)
    return iono_file

def run_iono_GIM(inps):
    """Estimate (and correct) ionospheric delay time-series."""
    #1. estimate iono time-series
    print('\n'+'-'*80)
    print('estimate iono time-series')
    create_iono_timeseries(inps.dis_file, inps.tec_dir, inps.geo_file, inps.iono_file)

    # 2. correct iono delay from displacement time-series via diff.py
    if inps.dis_file:
        print('\n'+'-'*80)
        print('Apply ionospheric correction to displacement file via diff.py ...')
        if ut.run_or_skip(inps.dis_file, [inps.dis_file, inps.iono_file]) == 'run':
            # [in the future] diff.py should handle different resolutions between
            # the iono time-series and displacement time-series files
            cmd = f'diff.py {inps.dis_file} {inps.iono_file} -o {inps.cor_dis_file} --force'
            print(cmd)
            diff.main(cmd.split()[1:])
        else:
            print(f'Skip re-applying and use existed corrected displacement file: {inps.cor_dis_file}.')
    else:
        print('No input displacement file, skip correcting ionospheric delays.')
        
    return
