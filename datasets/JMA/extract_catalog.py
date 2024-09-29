# %%
import gzip
import multiprocessing as mp
import os
import re
import json
import shutil
import logging
import logging.handlers
from concurrent_log_handler import ConcurrentRotatingFileHandler
from collections import namedtuple
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from functools import partial

import numpy as np
import obspy
import pandas as pd
import obspy.geodetics.base
from tqdm import tqdm

import warnings
# remove DeprecationWarning in obspy.geodetics.base
warnings.filterwarnings("ignore", category=DeprecationWarning)
with open('extract_catalog.log', 'w') as f:
    f.write('')
logging.basicConfig(
    filename="extract_catalog.log", level=logging.INFO, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = ConcurrentRotatingFileHandler("extract_catalog.log", "a", maxBytes=50*1024*1024, backupCount=2)
logger.addHandler(handler)
    

# %%
root_path = "dataset"
catalog_path = f"{root_path}/arrivaltime"
mechanism_path = f"{root_path}/mechanism"
station_path = f"{root_path}/station"

result_path = "dataset"
if not os.path.exists(result_path):
    os.makedirs(result_path)
if not os.path.exists(f"{result_path}/catalog"):
    os.makedirs(f"{result_path}/catalog")
if not os.path.exists(f"{result_path}/catalog_raw"):
    os.makedirs(f"{result_path}/catalog_raw")

# %%
## https://hinetwww11.bosai.go.jp/auth/manual/dlDialogue.php?r=jmaformat&LANG=en
## https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html
event_columns = {
    "Agency":                       (0, 1),  
    "Year":                         (1, 5),  
    "Month":                        (5, 7),  
    "Day":                          (7, 9),  
    "Hour":                         (9, 11), 
    "Minute":                       (11, 13),  
    "Second":                       (13, 17),  # 2 decimal
    "t_error":                      (17, 21),  # s, 2 decimal
    "latitude_deg":                 (21, 24),  
    "latitude_min":                 (24, 28),  # 2 decimal   
    "lat_error":                    (28, 32),  # min, 2 decimal
    "longitude_deg":                (32, 36),           
    "longitude_min":                (36, 40),           
    "lon_error":                    (40, 44),     
    "depth_km":                     (44, 49),  # 2 decimal
    "depth_error":                  (49, 52),  # km, 2 decimal
    "magnitude":                    (52, 54),  # 1 decimal
    "magnitude_type":               (54, 55),  
    "magnitude_2":                  (55, 57),       
    "magnitude_type_2":             (57, 58),            
    "travel_time_table":            (58, 59),       
    "location_precision":           (59, 60),              
    "subsidiary_information":       (60, 61),                  
    "max_intensity":                (61, 62),         
    "damage_class":                 (62, 63),        
    "tsunami_class":                (63, 64),         
    "district_number":              (64, 65),           
    "region_number":                (65, 68),         
    "region_name":                  (68, 92),       
    "num_stations":                 (92, 95),        
    "hypocenter_determination_flag":(95, 96),
}

event_decimal_number = {
    "Second": 100,
    "latitude_deg": 1,
    "latitude_min": 100,
    "longitude_deg": 1,
    "longitude_min": 100,
    "depth_km": 100,
    "magnitude": 10,
}

## https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/fmtdk_e.html
phase_columns = {
    "record_type":              (0, 1),                  
    "station_code":             (1, 7),   
    "station_number":           (7, 11),           
    "seismometer_type":         (12, 13),                   
    "date":                     (13, 15),           
    "p_remark":                 (15, 19),               
    "arrival_time_hour":        (19, 21),                          
    "arrival_time_p_minute":    (21, 23),                            
    "arrival_time_p_second":    (23, 27),                            
    "s_remark":                 (27, 31),               
    "arrival_time_s_minute":    (31, 33),                            
    "arrival_time_s_second":    (33, 37),                            
    "max_amp_x":                (37, 42),               
    "max_amp_x_period":         (42, 45),                       
    "max_amp_x_time":           (45, 48),                     
    "max_amp_y":                (48, 53),                
    "max_amp_y_period":         (53, 56),                       
    "max_amp_y_time":           (56, 59),                     
    "max_amp_z":                (59, 64),                
    "max_amp_z_period":         (64, 67),                       
    "max_amp_z_time":           (67, 70),                     
    "max_amp_unit":             (70, 71),                   
    "initial_motion_x_plority": (71, 72),
    "initial_motion_x_amp":     (72, 75),                           
    "initial_motion_y_plority": (75, 76),
    "initial_motion_y_amp":     (76, 79),                           
    "initial_motion_z_plority": (79, 80),
    "initial_motion_z_amp":     (80, 83),                           
    "initial_motion_unit":      (83, 84),                          
    "duration":                 (84, 87),               
    "year":                     (87, 89),           
    "month":                    (89, 91),            
    "p_reliability":            (91, 92),             
    "s_reliability":            (92, 93),             
    "amp_flag":                 (93, 94),               
    "other_flag":               (94, 95),                 
    "location_weight":          (95, 96),                      
}

phase_decimal_number = {
    "arrival_time_p_second": 100,
    "arrival_time_s_second": 100,
    "max_amp_x_period": 10,
    "max_amp_y_period": 10,
    "max_amp_z_period": 10,
}


def read_event_line(line):
    event = {}
    event["is_valid"] = True
    event["event_id"] = line[0:17].strip()
    for key, (start, end) in event_columns.items():
        if key in event_decimal_number:
            try:
                if line[start:end].strip() == "":
                    event[key] = ""
                    event["is_valid"] = False
                    return event
                else:
                    if key == "magnitude" and line[start].isupper():
                        convertio = {'A': -1.0, 'B': -2.0, 'C': -3.0}
                        mag = line[start:end]
                        event[key] = convertio[mag[0]] + float(mag[1:]) / event_decimal_number[key]
                    else:
                        event[key] = float(line[start:end].replace(' ', '')) / event_decimal_number[key]
            except:
                event["is_valid"] = False
                #print(key, line[start:end])
                logging.warning(f"Read event line failed at {key}, {line[start:end]}")
        else:
            event[key] = line[start:end]

    if event["Second"] < 60:
        event[
            "time"
        ] = f"{event['Year']}-{event['Month']}-{event['Day']}T{event['Hour']}:{event['Minute']}:{event['Second']:06.3f}"
    else:
        tmp = datetime.fromisoformat(
            f"{event['Year']}-{event['Month']}-{event['Day']}T{event['Hour']}:{event['Minute']}"
        )
        tmp += timedelta(seconds=event["Second"])
        event["time"] = tmp.strftime("%Y-%m-%dT%H:%M:%S.%f")

    event["latitude"] = round(event["latitude_deg"] + event["latitude_min"] / 60, 6)
    event["longitude"] = round((event["longitude_deg"] + event["longitude_min"] / 60), 6)
    
    return event


def read_phase_line(line, only_snet=True):
    # NOTE: p_remark is the first phase actually, it could be S phase, but it doesn't matter
    ## check p_remark
    phases = []
    p_phase_time = None
    start, end = phase_columns["p_remark"]
    if len(line[start:end].strip()) > 0:
        p_phase = {}
        for key, (start, end) in phase_columns.items():
            # ######## filter strange data ############
            # if key == "p_travel_time_residual":
            #     if line[start : end + 3] == " " * 3 + "0" + " " * 2 + "0":
            #         # print(f"strange data: {line}")
            #         return []
            # #########################################
            if key in phase_decimal_number:
                if len(line[start:end].strip()) == 0:
                    p_phase[key] = ""
                else:
                    p_phase[key] = float(line[start:end].strip()) / phase_decimal_number[key]
            else:
                p_phase[key] = line[start:end]
        
        #NOTE: hard code for year
        p_phase['year'] = "20" + p_phase['year'] if int(p_phase['year']) < 50 else "19" + p_phase['year']
        if (p_phase["arrival_time_p_second"] < 60) and (p_phase["arrival_time_p_second"] >= 0):
            tmp = datetime.fromisoformat( 
                f"{p_phase['year']}-{int(p_phase['month']):02}-{int(p_phase['date']):02}T{p_phase['arrival_time_hour']}:{p_phase['arrival_time_p_minute']}:{p_phase['arrival_time_p_second']:06.3f}"
            )
        else:
            tmp = datetime.fromisoformat(
                f"{p_phase['year']}-{int(p_phase['month']):02}-{int(p_phase['date']):02}T{p_phase['arrival_time_hour']}:{p_phase['arrival_time_p_minute']}"
            )
            tmp += timedelta(seconds=p_phase["arrival_time_p_second"])
        p_phase_time = tmp
        p_phase["phase_time"] = tmp.strftime("%Y-%m-%dT%H:%M:%S.%f")
        # NOTE: in the record, s phase doesn't have its own arrival hour, so we use p phase's instead
        # But, for p phase arrival 12:59:55.0, s phase arrival 13:00:05.0, 
        # the s phase arrival minute still is 00, so we need to add 1 hour to s phase arrival time
        # p_phase_time is a flag for checking if s phase arrival time is later than p phase
        p_phase["phase_polarity"] = p_phase["initial_motion_z_plority"]
        p_phase["phase_remark"] = p_phase["p_remark"].strip()
        p_phase["phase_score"] = ""
        p_phase["phase_type"] = p_phase["p_remark"].strip()[-1]
        p_phase["location_weight"] = p_phase["location_weight"]
        # station code: N.101S -> station: N.S1N01
        p_phase["station_id"] = f"{p_phase['station_code'][:2]}S{p_phase['station_code'][2]}N{p_phase['station_code'][3:5]}" if only_snet else p_phase['station_code']
        if p_phase['station_code'] == "N.213S":
            p_phase["station_id"] = "N.S2N13A"
        p_phase["phase_reliability"] = p_phase["p_reliability"]
        phases.append(p_phase)
    start, end = phase_columns["s_remark"]
    if len(line[start:end].strip()) > 0:
        s_phase = {}
        for key, (start, end) in phase_columns.items():
            if key in phase_decimal_number:
                if len(line[start:end].strip()) == 0:
                    s_phase[key] = ""
                else:
                    s_phase[key] = float(line[start:end].strip()) / phase_decimal_number[key]
            else:
                s_phase[key] = line[start:end]
        
        #NOTE: hard code for year    
        s_phase['year'] = "20" + s_phase['year'] if int(s_phase['year']) < 50 else "19" + s_phase['year']
        if (s_phase["arrival_time_s_second"] < 60) and (s_phase["arrival_time_s_second"] >= 0):
            tmp = datetime.fromisoformat(
                f"{s_phase['year']}-{int(s_phase['month']):02}-{int(s_phase['date']):02}T{s_phase['arrival_time_hour']}:{s_phase['arrival_time_s_minute']}:{s_phase['arrival_time_s_second']:06.3f}"
            )
        else:
            tmp = datetime.fromisoformat(
                f"{s_phase['year']}-{int(s_phase['month']):02}-{int(s_phase['date']):02}T{s_phase['arrival_time_hour']}:{s_phase['arrival_time_s_minute']}"
            )
            tmp += timedelta(seconds=s_phase["arrival_time_s_second"])
        # if tmp is earlier than p_phase_time, add 1 hour
        if p_phase_time is not None and tmp < p_phase_time:
            tmp += timedelta(hours=1)
            if p_phase_time.minute <= 58 or tmp.minute >= 2:
                # warning huge interval between p and s phase
                #print(f"Warning: 1-hour correction is triggered, but huge interval between p and s phase, p: {p_phase_time}, s: {tmp}")
                logging.warning(f"1-hour correction is triggered, but huge interval between p and s phase, p: {p_phase_time}, s: {tmp}")
        s_phase["phase_time"] = tmp.strftime("%Y-%m-%dT%H:%M:%S.%f")
        s_phase["phase_remark"] = s_phase["s_remark"].strip()
        s_phase["phase_score"] = ""
        s_phase["phase_type"] = s_phase["s_remark"].strip()[-1]
        s_phase["location_weight"] = s_phase["location_weight"]
        # station code: N.101S -> station: N.S1N01
        s_phase["station_id"] = f"{s_phase['station_code'][:2]}S{s_phase['station_code'][2]}N{s_phase['station_code'][3:5]}" if only_snet else s_phase['station_code']
        if s_phase['station_code'] == "N.213S":
            s_phase["station_id"] = "N.S2N13A"
        s_phase["phase_reliability"] = s_phase["s_reliability"]
        phases.append(s_phase)

    return phases


# %%
def process(phase_file, stations_info, only_snet=True, ncpu=1):
    #for phase_file in sorted(glob(f"{catalog_path}/d{year}??"))[::-1]:
    phase_filename = phase_file.split("/")[-1]

    # if not os.path.exists(f"{result_path}/catalog_raw/{phase_filename[:-2]}"):
    shutil.copy(phase_file, f"{result_path}/catalog_raw/{phase_filename}")

    with open(f"{result_path}/catalog_raw/{phase_filename}") as f:
        lines = f.readlines()
    catalog = {}
    event_id = None
    event = {}
    picks = []
    if ncpu > 1:
        logging.info(f"Parsing {phase_filename} with {ncpu} processes")
        lines_iterator = lines
    else:
        lines_iterator = tqdm(lines, desc=phase_filename+" parsing")
    for line in lines_iterator:
        if line[0] == "C": # comment line
            continue
        if line[0] == "E": # end line
            if event_id is not None:
                assert event["event_id"] == event_id
                if event["is_valid"] and len(picks) > 0:
                    catalog[event_id] = {"event": event, "picks": picks}
                event_id = None
                event = {}
                picks = []
            continue
        
        if line[0] in ["J", "U", "I"]:  # event_line
            if event != {}: # not the first record
                continue
            assert event_id is None, f"event_id: {event_id}, line: {line}"
            event_id = line[0:17].strip()
            event = read_event_line(line)
            picks = []
        elif line[0] == 'j': # template event line
            event_id = None
            event = {}
            picks = []
            continue
        else:  # phase_line
            if line[0] != "_": # not observation record
                continue
            if only_snet and (re.match(r"N.\d\d\dS", line[1:7]) is None):
                continue # not S-net station
            picks.extend(read_phase_line(line, only_snet=only_snet))

    events = []
    phases = []
    if ncpu > 1:
        logging.info(f"Building {phase_filename} with {ncpu} processes")
        event_ids_iterator = catalog.keys()
    else:
        event_ids_iterator = tqdm(catalog.keys(), desc=phase_filename+" building")
    for event_id in event_ids_iterator:
        events.append(catalog[event_id]["event"])
        phase = pd.DataFrame(catalog[event_id]["picks"])
        phase["event_id"] = event_id
        assert 'station_id' in phase.columns, f'{phase_filename} {event_id} {phase.columns}'
        # calc dist and azimuth with obspy
        phase["dist_azi"] = phase['station_id'].apply(
            lambda x: obspy.geodetics.base.gps2dist_azimuth(
                catalog[event_id]["event"]["latitude"], catalog[event_id]["event"]["longitude"], stations_info[x]["latitude"], stations_info[x]["longitude"]
                ))
        phase["distance_km"] = phase["dist_azi"].apply(lambda x: round(x[0] / 1000, 6))
        phase["azimuth"] = phase["dist_azi"].apply(lambda x: round(x[1], 6)) # event -> station
        phase["back_azimuth"] = phase["dist_azi"].apply(lambda x: round(x[2], 6)) # station -> event
        phase.drop(columns=["dist_azi"], inplace=True)
        phases.append(phase)

    events = pd.DataFrame(events)
    if len(phases) == 0:
        logging.warning(f"No valid S-Net phase record in {phase_filename}")
        return # no valid arrival time record
    phases = pd.concat(phases)
    events = events[["event_id", "time", "latitude", "longitude", "depth_km", "magnitude", "magnitude_type"]]
    # events["event_id"] = events["event_id"].apply(lambda x: "jma" + x)
    events["time"] = events["time"].apply(lambda x: x + "+09:00")
    # TODO: find takeoff_angle record？
    phases["takeoff_angle"] = ""
    phases["network"] = phases["station_id"].apply(lambda x: x.split(".")[0])
    phases["station"] = phases["station_id"].apply(lambda x: x.split(".")[1])
    phases["location"] = ""
    phases["instrument"] = phases['seismometer_type']
    phases["component"] = "XYZ" if only_snet else ""
    phases = phases[
        [
            "event_id",
            "network",
            "station",
            "location",
            "instrument",
            "component",
            "phase_type",
            "phase_time",
            "phase_score",
            "phase_polarity",
            "phase_remark",
            "distance_km",
            "azimuth",
            "takeoff_angle",
            "phase_reliability",
            "location_weight",
        ]
    ]
    # phases["event_id"] = phases["event_id"].apply(lambda x: "jma" + x)
    phases["phase_time"] = phases["phase_time"].apply(lambda x: x + "+09:00")
    phases["station"] = phases["station"].str.strip()
    phases["phase_polarity"] = phases["phase_polarity"].str.strip()
    # phases["azimuth"] = phases["azimuth"].str.strip()
    # phases["takeoff_angle"] = phases["takeoff_angle"].str.strip()
    # phases = phases[phases["distance_km"] != ""]
    phases = phases[~(phases["location_weight"] == 0)]

    # %%
    phases_ps = []
    event_ids = []
    if ncpu > 1:
        logging.info(f"Cleaning {phase_filename} with {ncpu} processes")
        cleaning_iterator = phases.groupby(["event_id", "station"])
    else:
        cleaning_iterator = tqdm(phases.groupby(["event_id", "station"]), desc=phase_filename+" cleaning")
    for (event_id, station), picks in cleaning_iterator:
        if len(picks) >= 2:
            phase_type = picks["phase_type"].unique()
            if ("P" in phase_type) and ("S" in phase_type):
                phases_ps.append(picks)
                event_ids.append(event_id)
        if len(picks) >= 3:
            #print(event_id, station, len(picks))
            logging.warning(f"Event {event_id} station {station} has more than 2 phases: {len(picks)} picks")
    if len(phases_ps) == 0:
        logging.warning(f"No valid S-Net phase record with both P and S phase in {phase_filename}")
        return # no valid arrival time record
    phases_ps = pd.concat(phases_ps)
    #NOTE: only keep records with at least one station with both P and S phase
    events = events[events.event_id.isin(event_ids)]
    phases = phases[phases.event_id.isin(event_ids)]

    # %%
    #events["time"] = pd.to_datetime(events["time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    #phases_ps["phase_time"] = pd.to_datetime(phases_ps["phase_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    #phases["phase_time"] = pd.to_datetime(phases["phase_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
    events = events.sort_values("event_id", ignore_index=True)
    phases_ps = phases_ps.sort_values("event_id", ignore_index=True)
    phases = phases.sort_values("event_id", ignore_index=True)
    events.to_csv(f"{result_path}/catalog/{phase_filename[1:]}.event.csv", index=False)
    phases_ps.to_csv(f"{result_path}/catalog/{phase_filename[1:]}.phase_ps.csv", index=False)
    phases.to_csv(f"{result_path}/catalog/{phase_filename[1:]}.phase.csv", index=False)

    # year, month = phase_filename.split("/")[-1].split(".")[0:2]
    # if not os.path.exists(f"{result_path}/catalog/{year}"):
    #     os.makedirs(f"{result_path}/catalog/{year}")
    # events.to_csv(f"{result_path}/catalog/{year}/{year}_{month}.event.csv", index=False)
    # phases_ps.to_csv(f"{result_path}/catalog/{year}/{year}_{month}.phase_ps.csv", index=False)
    # phases.to_csv(f"{result_path}/catalog/{year}/{year}_{month}.phase.csv", index=False)


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    # years = range(2023, 2024)[::-1]
    years = range(2016, 2024)[::-1]
    phase_files = []
    for year in years:
        phase_files.extend(sorted(glob(f"{catalog_path}/d{year}??"))[::-1])
    phase_files = sorted(phase_files)[::-1]
    stations_info = json.load(open(f"{station_path}/stations.json"))
    ncpu = min(16, mp.cpu_count())
    
    process_partial = partial(process, stations_info=stations_info, only_snet=True, ncpu=ncpu)
    with ctx.Pool(processes=ncpu) as pool:
        pbar = tqdm(phase_files, total=len(phase_files), desc="Processing", leave=True)
        processes = []
        for phase_file in phase_files:
            p = pool.apply_async(process_partial, args=(phase_file,), callback=lambda _: pbar.update(1))
            processes.append(p)
        for p in processes:
            p.get()
        pbar.close()
