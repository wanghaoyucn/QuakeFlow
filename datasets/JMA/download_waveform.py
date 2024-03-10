# %%
# !pip install HinetPy
# # !wget https://github.com/AI4EPS/software/releases/download/win32tools/win32tools.tar.gz
# ! [ -e win32tools.tar.gz ] || wget https://github.com/AI4EPS/software/releases/download/win32tools/win32tools.tar.gz
# !tar -xvf win32tools.tar.gz
# !cd win32tools && make

# NOTE: The unit of data in SAC file is in nm/s or nm/s^2

# %%
import os
import json
import time
import random
import warnings
from glob import glob
from pathlib import Path
from typing import Dict
import threading
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import obspy
import pandas as pd
from tqdm.auto import tqdm
from HinetPy import Client, win32

matplotlib.use("Agg")
warnings.filterwarnings("ignore")
os.environ["PATH"] += os.pathsep + os.path.abspath("win32tools/catwin32.src") + os.pathsep + os.path.abspath("win32tools/win2sac.src")

# %%
root_path = "dataset"
catalog_path = f"{root_path}/catalog"
station_path = f"{root_path}/station"
waveform_path = f"{root_path}/waveforms"
raw_data_path = f"{root_path}/win32"

network: str = "snet"
if_accel: bool = True

USERNAME = ""
PASSWORD = ""
TIMEOUT = 60 # seconds

CODE = {
    "hinet": ["0101"],
    "fnet": ["0103"],
    "fnetA": ["0103A"],
    "snet": ["0120"],
    "snetA": ["0120A"],
    "mesonet": ["0131"],
}


# %%
def download_event(
    event, phases, client, waveform_path, network_codes=["0120"], time_before=30, time_after=30, lock=None, rank=0
):

    max_retry = 6

    arrival_time = phases[phases["phase_type"] == "P"]["phase_time"].min()
    end_time = phases[phases["phase_type"] == "S"]["phase_time"].max()
    starttime = (arrival_time - pd.to_timedelta(time_before, unit="s")).replace(tzinfo=None)
    # wave takes 5 minutes to across the whole network
    span_min = int(max(5, ((time_before + time_after + end_time-arrival_time).seconds) / 60))

    print(f"Downloading {event.event_id} ...")
    outdir = f"{waveform_path}/{event['event_id']}"

    if os.path.exists(outdir):
        print(f"{outdir} already exists. Skip.")
        return

    for code in network_codes:
        retry = 0
        while retry < max_retry:
            try:
                data, ctable = client.get_continuous_waveform(code, starttime, span=span_min, outdir=outdir.replace("waveforms", "win32"), threads=3)
                win32.extract_sac(data, ctable, outdir=outdir)

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
if __name__ == "__main__":
    
    client = Client(user=USERNAME, password=PASSWORD, timeout=TIMEOUT, retries=1)

    if not os.path.exists(f"{waveform_path}"):
        os.makedirs(f"{waveform_path}")
    if not os.path.exists(f"{raw_data_path}"):
        os.makedirs(f"{raw_data_path}")
        
    event_list = sorted(list(glob(f"{catalog_path}/*.event.csv")))[::-1]
    start_year = "2022"
    end_year = "2023"
    tmp = []
    for event_file in event_list:
        if (
            event_file.split("/")[-1].split(".")[0][:4] >= start_year
            and event_file.split("/")[-1].split(".")[0][:4] <= end_year
        ):
            tmp.append(event_file)
    event_list = sorted(tmp, reverse=True)
    
    network_codes = CODE[network]
    if if_accel:
        try:
            network_codes+=CODE[network+"A"]
        except:
            print(f"{network} does not have acceleration sensors.")
    
  
    with open(f"{station_path}/stations.json") as f:
            stations = json.load(f)
    stations = {key: station for key, station in stations.items()}

    for event_file in event_list:
        print(event_file)
        events = pd.read_csv(event_file, parse_dates=["time"])
        events['time'] = pd.to_datetime(events['time'], utc=True).dt.tz_convert('Asia/Tokyo')
        events['time'] = events['time'].dt.tz_localize(None)
        phases = pd.read_csv(
            f"{event_file.replace('event.csv', 'phase.csv')}",
            parse_dates=["phase_time"],
            keep_default_na=False,
        )
        phases = phases.loc[
            phases.groupby(["event_id", "network", "station", "location", "instrument"]).phase_time.idxmin()
        ]
        phases['time'] = pd.to_datetime(phases['phase_time'], utc=True).dt.tz_convert('Asia/Tokyo')
        phases['time'] = phases['time'].dt.tz_localize(None)
        phases.set_index("event_id", inplace=True)

        events = events[events.event_id.isin(phases.index)]
        #pbar = tqdm(events, total=len(events))
        
        for _, event in events.iterrows():
            threads = []
            MAX_THREADS = 1
            TIME_BEFORE = 30
            TIME_AFTER = 90
            lock = threading.Lock()
            events = events.sort_values(by="time", ignore_index=True, ascending=False)
            for _, event in events.iterrows():
                t = threading.Thread(
                    target=download_event,
                    args=(event, phases.loc[event.event_id], client, waveform_path, network_codes, TIME_BEFORE, TIME_AFTER, lock, len(threads)),
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
        #pbar.close()
