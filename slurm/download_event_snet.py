# %% [markdown]
##  Important: Before downloading
# - Make sure on Hinet website you select which stations and networks you want to download continuous waveform data for and then run this notebook (ie Hinet vs Fnet data and which province(s))
import os
from typing import Dict

import numpy as np
from HinetPy import Client, win32


# %%
def download_waveform_event(
    root_path: str,
    region: str,
    config: Dict,
    hinet_client,
    rank: int = 0,
    protocol: str = "file",
    bucket: str = "",
    token: Dict = None,
):
    import json
    import os
    import random
    import threading
    import time
    from collections import namedtuple

    import fsspec
    import numpy as np
    import obspy
    import obspy.clients.fdsn
    import obspy.geodetics.base
    import obspy.taup
    import pandas as pd

    # %%
    def calc_arrival_time(event, station_locs):
        taup_model = obspy.taup.TauPyModel(model="iasp91")
        Location = namedtuple("location", ["longitude", "latitude", "depth_km"])
        dist = np.sqrt(
                ((event["longitude"] - station_locs[1]) * np.cos(np.deg2rad(event["latitude"]))) ** 2
                + (event["latitude"] - station_locs[0]) ** 2
            )
        idx_min = np.argmin(dist)
        closest = Location(
            longitude=station_locs[1][idx_min], latitude=station_locs[0][idx_min], depth_km=station_locs[2][idx_min]
        )
        idx_max = np.argmax(dist)
        farthest = Location(
            longitude=station_locs[1][idx_max], latitude=station_locs[0][idx_max], depth_km=station_locs[2][idx_max]
        )
        
        min_dist_km = (
            obspy.geodetics.base.gps2dist_azimuth(
                event.latitude, event["longitude"], closest.latitude, closest.longitude
            )[0]
            / 1000
        )
        min_dist_deg = obspy.geodetics.base.kilometer2degrees(min_dist_km)
        min_tt = taup_model.get_travel_times(
            distance_in_degree=min_dist_deg,
            source_depth_in_km=max(0, event.depth_km),
            receiver_depth_in_km=max(0, closest.depth_km),
        )[0].time
        max_dist_km = (
            obspy.geodetics.base.gps2dist_azimuth(
                event.latitude, event["longitude"], farthest.latitude, farthest.longitude
            )[0]
            / 1000
        )
        max_dist_deg = obspy.geodetics.base.kilometer2degrees(max_dist_km)
        max_tt = taup_model.get_travel_times(
            distance_in_degree=max_dist_deg,
            source_depth_in_km=max(0, event.depth_km),
            receiver_depth_in_km=max(0, farthest.depth_km),
        )[0].time
        span = pd.to_timedelta(max_tt - min_tt, unit="s")

        return event.time + pd.to_timedelta(min_tt, unit="s"), span.total_seconds()

    # %%
    def download_event(
        event, station_locs, client, root_path, waveform_dir, time_before=30, time_after=30, lock=None, rank=0
    ):
        if not os.path.exists(f"{root_path}/{waveform_dir}"):
            os.makedirs(f"{root_path}/{waveform_dir}")

        max_retry = 10

        arrival_time, span_min = calc_arrival_time(event, station_locs)
        starttime = arrival_time - pd.to_timedelta(time_before, unit="s")
        span_min = int(max(2, (time_before + time_after + span_min)/60))

        print(f"Downloading {event.event_id} ...")
        outdir = f"{root_path}/{waveform_dir}/{event['event_id']}"

        if os.path.exists(f"{root_path}/{waveform_dir}/{event['event_id']}"):
            print(f"{root_path}/{waveform_dir}/{event['event_id']} already exists. Skip.")
            if protocol != "file":
                if not fs.exists(f"{bucket}/{waveform_dir}/{event['event_id']}"):
                    fs.put(f"{root_path}/{waveform_dir}/{event['event_id']}", f"{bucket}/{waveform_dir}/{event['event_id']}")
            return

        if protocol != "file":
            if fs.exists(f"{bucket}/{waveform_dir}/{event['event_id']}"):
                print(f"{bucket}/{waveform_dir}/{event['event_id']} already exists. Skip.")
                fs.get(f"{bucket}/{waveform_dir}/{event['event_id']}", f"{root_path}/{waveform_dir}/{event['event_id']}")
                return

        retry = 0
        while retry < max_retry:
            try:
                data, ctable = client.get_continuous_waveform("0120", starttime, span=span_min, outdir=outdir.replace("waveforms", "win32"), threads=3)
                win32.extract_sac(data, ctable, outdir=outdir+"/velo")

                if protocol != "file":
                    fs.put(f"{root_path}/{waveform_dir}/{event['event_id']}", f"{bucket}/{waveform_dir}/{event['event_id']}")
                break

            except Exception as err:
                err = str(err).rstrip("\n")
                message = "No data available for request"
                if err[: len(message)] == message:
                    print(f"{message}: {event['event_id']}")
                    break
                else:
                    print(f"Error occurred:\n{err}\nRetrying...")
                retry += 1
                time.sleep(20 ** ((random.random() + 1)))
                continue

        if retry == max_retry:
            print(f"Failed to download {event['event_id']} after {max_retry} retries.")
            os.system(f"touch {str(event['event_id'])+'.failed'}")
            
        retry = 0
        while retry < max_retry:
            try:
                data, ctable = client.get_continuous_waveform("0120A", starttime, span=span_min, outdir=outdir.replace("waveforms", "win32"), threads=3)
                win32.extract_sac(data, ctable, outdir=outdir+"/accl")

                if protocol != "file":
                    fs.put(f"{root_path}/{waveform_dir}/{event['event_id']}", f"{bucket}/{waveform_dir}/{event['event_id']}")
                break

            except Exception as err:
                err = str(err).rstrip("\n")
                message = "No data available for request"
                if err[: len(message)] == message:
                    print(f"{message}: {event['event_id']}")
                    break
                else:
                    print(f"Error occurred:\n{err}\nRetrying...")
                retry += 1
                time.sleep(20 ** ((random.random() + 1)))
                continue

        if retry == max_retry:
            print(f"Failed to download {event['event_id']} after {max_retry} retries.")
            os.system(f"touch {str(event['event_id'])+'.failed'}")    


    # %%
    fs = fsspec.filesystem(protocol=protocol, token=token)
    # %%
    data_dir = f"{region}/obspy"
        
    # with open(f"{root_path}/{region}/config.json") as f:
    #     config = json.load(f)
    num_nodes = config["kubeflow"]["num_nodes"] if "num_nodes" in config["kubeflow"] else 1
    waveform_dir = f"{region}/waveforms"
    if not os.path.exists(f"{root_path}/{waveform_dir}"):
        os.makedirs(f"{root_path}/{waveform_dir}")

    if protocol == "file":
        events = pd.read_csv(f"{root_path}/{data_dir}/catalog.csv", parse_dates=["time"])
    else:
        events = pd.read_csv(f"{protocol}://{bucket}/{data_dir}/catalog.csv", parse_dates=["time"])
    events = pd.read_csv(f"{root_path}/{data_dir}/catalog.csv")
    events['time'] = pd.to_datetime(events['time'], utc=True).dt.tz_convert('Asia/Tokyo')
    events['time'] = events['time'].dt.tz_localize(None)
    events['event_id'] = events['event_id'].apply(lambda x: "jma"+x.split("=")[1])
    events = events.iloc[rank::num_nodes, :]
    
    if not os.path.exists(f"{root_path}/{data_dir}/stations.json"):
        stations = client.get_station_list('0120')
        name = [sta.name for sta in stations]
        latitude = [sta.latitude for sta in stations]
        longitude = [sta.longitude for sta in stations]
        elevation = [sta.elevation for sta in stations]
        stations_df = pd.DataFrame({'network':'S-Net', 'station': name, 'location': '', 'instrument':'velo', 'latitude': latitude, 'longitude': longitude, 'elevation_m': elevation})
        stations_df['depth_km'] = -stations_df['elevation_m'] / 1000
        stations_df['provider'] = 'NIED'
        stations_config = {row['station']: row.to_dict() for i, row in stations_df.iterrows()}
        with open(f"{root_path}/{data_dir}/stations.json", "w") as f:
            json.dump(stations_config, f, indent=4)
        stations_df.to_csv(f"{root_path}/{data_dir}/stations.csv", index=False)
    
    if protocol == "file":
        with open(f"{root_path}/{data_dir}/stations.json") as f:
            stations = json.load(f)
    else:
        with fs.open(f"{bucket}/{data_dir}/stations.json") as f:
            stations = json.load(f)
    stations = {key: station for key, station in stations.items()}
    station_locs = np.array([[station['latitude'] for _, station in stations.items()],
                         [station['longitude'] for _, station in stations.items()],
                         [station['depth_km'] for _, station in stations.items()]])


    threads = []
    MAX_THREADS = 1
    TIME_BEFORE = 30
    TIME_AFTER = 90
    lock = threading.Lock()
    events = events.sort_values(by="time", ignore_index=True, ascending=False)
    for _, event in events.iterrows():
        t = threading.Thread(
            target=download_event,
            args=(event, station_locs, hinet_client, root_path, waveform_dir, TIME_BEFORE, TIME_AFTER, lock, len(threads)),
        )
        t.start()
        threads.append(t)
        time.sleep(1)
        if (len(threads) - 1) % MAX_THREADS == (MAX_THREADS - 1):
            for t in threads:
                t.join()
            threads = []
    for t in threads:
        t.join()

# %%
if __name__ == "__main__":
    import json
    import sys

    # %%
    root_path = "local"
    region = "Snet"
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        region = sys.argv[2]

    # %%
    USERNAME = ""
    PASSWORD = ""
    TIMEOUT = 60 # seconds
    client = Client(user=USERNAME, password=PASSWORD, timeout=TIMEOUT, retries=1)
    
    with open(f"{root_path}/{region}/config.json", "r") as fp:
        config = json.load(fp)

    download_waveform_event(root_path, region=region, config=config, hinet_client=client)