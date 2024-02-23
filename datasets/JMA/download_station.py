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

# for network in station_path.glob("*.info"):
for network in glob(f"{station_path}/*.info"):
    network_name = network.split("/")[-1]
    if network_name in ["broadband.info", "BARD.info", "CISN.info"]:
        continue
    # if (root_path / "station" / f"{network.stem}.csv").exists():
    network_stem = network.split("/")[-1].split(".")[0]
    if os.path.exists(f"{result_path}/{network_stem}.csv"):
        print(f"Skip {network_stem}")
        # continue
    print(f"Parse {network_stem}")
    inv = obspy.Inventory()
    # for xml in (network / f"{network_stem}.FDSN.xml").glob(f"{network_stem}.*.xml"):
    for xml in glob(f"{network}/{network_stem}.FDSN.xml/{network_stem}.*.xml"):
        inv += obspy.read_inventory(xml)
    stations = parse_inventory_csv(inv)
    if len(stations) > 0:
        stations.to_csv(f"{result_path}/{network_stem}.csv", index=False)


# %%
