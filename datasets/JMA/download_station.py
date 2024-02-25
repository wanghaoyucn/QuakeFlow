# %%
# !pip install HinetPy
# # !wget https://github.com/AI4EPS/software/releases/download/win32tools/win32tools.tar.gz
# ! [ -e win32tools.tar.gz ] || wget https://github.com/AI4EPS/software/releases/download/win32tools/win32tools.tar.gz
# !tar -xvf win32tools.tar.gz
# !cd win32tools && make

# %%
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os
import sys
import json

import numpy as np
import obspy
import pandas as pd
from tqdm import tqdm
from glob import glob
from HinetPy import Client, win32

os.environ["PATH"] += os.pathsep + os.path.abspath("win32tools/catwin32.src") + os.pathsep + os.path.abspath("win32tools/win2sac.src")

# %%
root_path = "./"
waveform_path = f"{root_path}/waveform"
catalog_path = f"{root_path}/catalog"
station_path = f"{root_path}/station"
result_path = f"dataset/station"
if not os.path.exists(result_path):
    os.makedirs(result_path)

CODE = {
    "hinet": "0101",
    "fnet": "0103",
    "fnetA": "0103A",
    "snet": "0120",
    "snetA": "0120A",
    "mesonet": "0131",
}
NETWORK = {
    "hinet": "Hi-net",
    "fnet": "F-net",
    "fnetA": "F-net",
    "snet": "S-net",
    "snetA": "S-net",
    "mesonet": "MeSO-net",
}

def download_station(client, network: str, result_path):
    if not network in ["hinet", "fnet", "snet", "mesonet"]:
        raise NotImplementedError(f"Network {network} is not supported, only supports the following networks: Hi-net (0101), F-net (0103, 0103A), S-net (0120, 0120A) and MeSO-net (0131).")
    
    
    stations = client.get_station_list(CODE[network])
    name = [sta.name for sta in stations]
    latitude = [sta.latitude for sta in stations]
    longitude = [sta.longitude for sta in stations]
    elevation = [sta.elevation for sta in stations]
    jma_code = [f"{sta.name[:2]}{sta.name[3]}{sta.name[5:7]}S" for sta in stations] if "snet" in network else name
    instrument = "V" if CODE[network][-1] != "A" else "A"
    stations_df = pd.DataFrame({'network': NETWORK[network], 'station': name, 'location': '', 'instrument':'V', 'latitude': latitude, 'longitude': longitude, 'elevation_m': elevation})
    stations_df['depth_km'] = -stations_df['elevation_m'] / 1000
    stations_df['jma_code'] = jma_code
    stations_df['provider'] = 'NIED'
    stations_config = {row['station']: row.to_dict() for i, row in stations_df.iterrows()}
    with open(f"{result_path}/stations.json", "w") as f:
        json.dump(stations_config, f, indent=4)
    stations_df.to_csv(f"{result_path}/stations.csv", index=False)


# %%
if __name__ == "__main__":
    network = ["snet"]
    if len(sys.argv) > 1:
        network = sys.argv[1:]
    
    client = Client(timeout=120, retries=6)
    USERNAME = ""
    PASSWORD = ""
    client.login(USERNAME, PASSWORD)
    
    for net in tqdm(network, desc="Network"):
        download_station(client, net, result_path)