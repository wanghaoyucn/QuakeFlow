# %%
import os
from datetime import datetime

# %%
import pandas as pd
import requests
from tqdm import tqdm

# %%
# Define column specifications https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html
colspecs = [
    (0, 1),
    (1, 5),
    (5, 7),
    (7, 9),
    (9, 11),
    (11, 13),
    (13, 17),
    (17, 21),
    (21, 24),
    (24, 28),
    (28, 32),
    (32, 36),
    (36, 40),
    (40, 44),
    (44, 49),
    (49, 52),
    (52, 54),
    (54, 55),
    (55, 57),
    (57, 58),
    (58, 59),
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
    (63, 64),
    (64, 65),
    (65, 68),
    (68, 92),
    (92, 95),
    (95, 96),
]

# Define column names https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/format/hypfmt_e.html
colnames = [
    "Record type identifier",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Second",
    "Standard error (s)",
    "Latitude (deg)",
    "Latitude (min)",
    "Standard error (min)",
    "Longitude (deg)",
    "Longitude (min)",
    "Standard error (min)",
    "Depth (km)",
    "Standard error (km)",
    "Magnitude 1",
    "Magnitude type 1",
    "Magnitude 2",
    "Magnitude type 2",
    "Travel time table",
    "Hypocenter location precision",
    "Subsidiary information",
    "Maximum intensity",
    "Damage class",
    "Tsunami class",
    "District number",
    "Region number",
    "Region name",
    "Number of stations",
    "Hypocenter determination flag",
]

colnames = [
    "Agency",
    "Year",
    "Month",
    "Day",
    "Hour",
    "Minute",
    "Second",
    "t_error",
    "Latitude (deg)",
    "Latitude (min)",
    "lat_error",
    "Longitude (deg)",
    "Longitude (min)",
    "lon_error",
    "depth_km",
    "z_error",
    "magnitude",
    "magnitude type",
    "magnitude 2",
    "magnitude type 2",
    "travel_time",
    "location_precision",
    "subsidiary_information",
    "max_intensity",
    "damage_class",
    "tsunami_class",
    "district_number",
    "region_number",
    "region_name",
    "num_stations",
    "Hypocenter determination flag",
]

# %%
if __name__ == "__main__":
    result_path = "dataset/hypocenter"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    for year in tqdm(range(2020, 2021)):
        if not os.path.exists(os.path.join(result_path, f"h{year}")):
            os.system(f"wget https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/hypo/h{year}.zip -P {result_path}")
            os.system(f"unzip {os.path.join(result_path, f'h{year}.zip')} -d {result_path}")

        df = pd.read_fwf(os.path.join(result_path, f"h{year}"), colspecs=colspecs, header=None, names=colnames, skiprows=0, dtype=str)

        # # Read the file
        # df = pd.read_fwf("mechanism.txt", colspecs=colspecs, header=None, names=colnames, skiprows=7, dtype=str)

        # %%
        df["datetime"] = df.apply(
            lambda x: datetime.strptime(
                f"{x['Year']}-{x['Month']}-{x['Day']}T{x['Hour']}:{x['Minute']}:{x['Seconds']}", "%Y-%m-%dT%H:%M:%S.%f"
            ),
            axis=1,
        )
        df["Longitude"] = df.apply(
            lambda x: -round(float(x["Longitude (deg)"]) + float(x["Longitude (min)"]) / 60.0, 5), axis=1
        )
        df["Latitude"] = df.apply(
            lambda x: round(float(x["Latitude (deg)"]) + float(x["Latitude (min)"]) / 60.0, 5), axis=1
        )
        df["Latitude"] = df.apply(lambda x: -1 * x["Latitude"] if x["South"] == "S" else x["Latitude"], axis=1)
        df["Longitude"] = df.apply(lambda x: -1 * x["Longitude"] if x["East"] == "E" else x["Longitude"], axis=1)
        df["strike"] = df.apply(lambda x: (float(x["dip_direction"]) - 90) % 360, axis=1)
        df.drop(
            columns=[
                "Year",
                "Month",
                "Day",
                "Hour",
                "Minute",
                "Seconds",
                "Latitude (deg)",
                "Latitude (min)",
                "Longitude (deg)",
                "Longitude (min)",
                "South",
                "East",
                "dip_direction",
            ],
            inplace=True,
        )
        df["event_id"] = df.apply(lambda x: f"nc{x['event_id']}", axis=1)

        df.to_csv(os.path.join(result_path, f"{year}.csv"), index=False, date_format="%Y-%m-%dT%H:%M:%S.%f")
