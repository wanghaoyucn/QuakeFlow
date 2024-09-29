# %%
import logging
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from glob import glob
from pathlib import Path

import fsspec
import h5py
import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from obspy.signal.rotate import rotate2zne
from tqdm import tqdm

with open("convert_hdf5.log", "w") as f:
    f.write("")
logging.basicConfig(
    filename="convert_hdf5.log", level=logging.INFO, filemode="a", format="%(asctime)s - %(levelname)s - %(message)s"
)

# warnings.filterwarnings("error")
os.environ["OPENBLAS_NUM_THREADS"] = "2"

# NOTE: The unit of data in SAC file is in nm/s or nm/s^2 (1e-9m/s)
# NOTE: All traces's starttime are minutes sharp, second and microsecond are 0

# %%
protocol = "file"#"gs"
token = None#f"{os.environ['HOME']}/.config/gcloud/application_default_credentials.json"
# token = "cloud"
bucket = "quakeflow_dataset"
#fs = fsspec.filesystem(protocol=protocol, token=token)
fs = fsspec.filesystem("file")

#root_path = "dataset"
region = "JMA"
#root_path = f"{bucket}/{region}"
root_path = "dataset"
sac_path = f"{root_path}/waveforms"
catalog_path = f"{root_path}/catalog"
station_path = f"{root_path}/station/stations.csv"

# %%
result_path = f"{root_path}/waveform_h5"
if not os.path.exists(result_path):
    os.makedirs(result_path)

sampling_rate = 100
NT = 30000  # 300 s


# %%
def calc_snr(data, index0, noise_window=300, signal_window=300, gap_window=50):
    snr = []
    for i in range(data.shape[0]):
        j = index0
        if (len(data[i, j - noise_window : j - gap_window]) == 0) or (
            len(data[i, j + gap_window : j + signal_window]) == 0
        ):
            snr.append(0)
            continue
        noise = np.std(data[i, j - noise_window : j - gap_window])
        signal = np.std(data[i, j + gap_window : j + signal_window])

        if (noise > 0) and (signal > 0):
            snr.append(signal / noise)
        else:
            snr.append(0)

    return snr


# %%
def extract_pick(picks, begin_time, sampling_rate):
    phase_type = []
    phase_index = []
    phase_score = []
    phase_reliability = []
    phase_time = []
    phase_polarity = []
    phase_remark = []
    phase_picking_channel = []
    event_id = []
    for idx, pick in picks.sort_values("phase_time").iterrows():
        phase_type.append(pick.phase_type)
        phase_index.append(int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
        phase_score.append(pick.phase_score)
        phase_reliability.append(pick.reliability)
        phase_time.append(pick.phase_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z"))
        phase_remark.append(pick.phase_remark)
        phase_polarity.append(pick.phase_polarity)
        phase_picking_channel.append(pick.instrument + pick.component)
        event_id.append(pick.event_id)
    
    if len(set(phase_index)) != len(phase_index):
        logging.warning(f"{event_id[0]}/{picks.iloc[0]['network']}.{picks.iloc[0]['station']} has duplicate picks: {phase_index}")
        idxs = np.unique(np.array(phase_index), return_index=True)[1]
        phase_type = np.array(phase_type)[idxs].tolist()
        phase_index = np.array(phase_index)[idxs].tolist()
        phase_score = np.array(phase_score)[idxs].tolist()
        phase_reliability = np.array(phase_reliability)[idxs].tolist()
        phase_time = np.array(phase_time)[idxs].tolist()
        phase_remark = np.array(phase_remark)[idxs].tolist()
        phase_polarity = np.array(phase_polarity)[idxs].tolist()
        phase_picking_channel = np.array(phase_picking_channel)[idxs].tolist()
        event_id = np.array(event_id)[idxs].tolist()

    return (
        phase_type,
        phase_index,
        phase_score,
        phase_reliability,
        phase_time,
        phase_remark,
        phase_polarity,
        phase_picking_channel,
        event_id,
    )


# %%
def flip_polarity(phase_polarity, channel_dip=None):
    pol_out = []
    if channel_dip is None:
        return [polarity if polarity in ["U", "D"] else "N" for polarity in phase_polarity]
    
    for pol, dip in zip(phase_polarity, channel_dip):
        if pol == "U" or pol == "+":
            if dip == -90:
                pol_out.append("U")
            elif dip == 90:
                pol_out.append("D")
            else:
                pol_out.append("N")
        elif pol == "D" or pol == "-":
            if dip == -90:
                pol_out.append("D")
            elif dip == 90:
                pol_out.append("U")
            else:
                pol_out.append("N")
        else:
            pol_out.append("N")
    return pol_out


# Refer to: https://github.com/ltauxe/Python-for-Earth-Science-Students/blob/master/Lecture_22.ipynb
def dir2cart(Dir):
    """
    converts polar directions to cartesian coordinates
    Parameters:
        Dir[Azimuth,Plunge]:  directions in degreess
    Returns:
        [X,Y,Z]: cartesian coordinates
    """
    Az, Pl = np.radians(Dir[0]), np.radians(Dir[1])
    return np.array([np.cos(Az) * np.cos(Pl), np.sin(Az) * np.cos(Pl), np.sin(Pl)])


def cart2dir(X):
    """
    converts cartesian coordinates to polar azimuth and plunge
    Parameters:
        X: list of X,Y,Z coordinates
    Returns:
        [Az,Pl]: list of polar coordinates in degrees
    """
    R = np.sqrt(X[0] ** 2 + X[1] ** 2 + X[2] ** 2)  # calculate resultant vector length
    Az = (
        np.degrees(np.arctan2(X[1], X[0])) % 360.0
    )  # calculate declination taking care of correct quadrants (arctan2) and making modulo 360.
    Pl = np.degrees(np.arcsin(X[2] / R))  # calculate inclination (converting to degrees) #
    return [Az, Pl]


# %%
def convert_jday(jday, catalog_path, result_path, protocol='file', token=None):
    inv_dict = {}
    fs_ = fsspec.filesystem(protocol=protocol, token=token)
    ## NCEDC
    tmp = datetime.strptime(jday, "%Y.%j")
    year, month, day = f"{tmp.year:04d}", f"{tmp.month:02d}", f"{tmp.day:02d}"
    year, dayofyear = jday.split(".")
    
    if not os.path.exists(f"{result_path}/{year}{month}"):
        os.makedirs(f"{result_path}/{year}{month}", exist_ok=True)
    
    with h5py.File(f"{result_path}/{year}{month}/{dayofyear}.h5", "w") as fp:
        
        ## JMA
        with fs_.open(f"{catalog_path}/{year}{month}.event.csv", "rb") as f:
            events = pd.read_csv(f, parse_dates=["time"], date_format="%Y-%m-%dT%H:%M:%S.%f%z")
        events["time"] = pd.to_datetime(events["time"])
        events.set_index("event_id", inplace=True)

        ## NCEDC
        with fs_.open(f"{catalog_path}/{year}{month}.phase.csv", "rb") as f:
            phases = pd.read_csv(
                f,
                parse_dates=["phase_time"],
                date_format="%Y-%m-%dT%H:%M:%S.%f%z",
                dtype={"location": str},
            )

        phases["phase_time"] = pd.to_datetime(phases["phase_time"])
        phases["phase_polarity"] = phases["phase_polarity"].fillna("N")
        phases["phase_score"] = phases["phase_score"].fillna("")
        phases["instrument"] = phases["instrument"].fillna("V")
        phases["location"] = phases["location"].fillna("")
        phases["takeoff_angle"] = phases["takeoff_angle"].fillna("")
        phases["station_id"] = phases["network"] + "." + phases["station"] + "." + phases["location"]
        phases.sort_values(["event_id", "phase_time"], inplace=True)
        phases_by_station = phases.copy()
        phases_by_station.set_index(["station_id"], inplace=True)
        phases_by_event = phases.copy()
        phases_by_event.set_index(["event_id"], inplace=True)
        phases.set_index(["event_id", "station_id"], inplace=True)
        phases = phases.sort_index()
        
        with fs_.open(f"{catalog_path.replace('catalog', 'station')}/stations.csv", "rb") as f:
            stations = pd.read_csv(f)
        stations["location"] = stations["location"].fillna("")
        stations['station_idloc'] = stations['network'] + "." + stations['station'] + "." + stations['location']
        stations.set_index("station_idloc", inplace=True)
        stations = stations.sort_index()
        

        event_ids = sorted(fs_.ls(f"{sac_path}/{year}{month}/{jday}"), reverse=True)
        event_fnames = [x.split("/")[-1] for x in event_ids]
        event_ids = [x.split("/")[-1] for x in event_ids]
        #print(event_fnames)
        for event_id, event_fname in zip(event_ids, event_fnames):#tqdm(zip(event_ids, event_fnames)):
            if event_id not in events.index:
                continue

            if event_id in fp:
                logging.warning(f"Duplicate {event_id}: {event_fname}")
                continue

            gp = fp.create_group(event_id)
            # event info
            gp.attrs["event_id"] = event_id
            gp.attrs["event_time"] = events.loc[event_id, "time"].strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            gp.attrs["latitude"] = events.loc[event_id, "latitude"]
            gp.attrs["longitude"] = events.loc[event_id, "longitude"]
            gp.attrs["depth_km"] = events.loc[event_id, "depth_km"]
            gp.attrs["magnitude"] = events.loc[event_id, "magnitude"]
            gp.attrs["magnitude_type"] = events.loc[event_id, "magnitude_type"]
            gp.attrs["source"] = region

            # waveform info
            sac_list = list(fs_.glob(f"{sac_path}/{year}{month}/{jday}/{event_fname}/*.SAC"))
            sac_list = sorted(list(set([s[:-5] for s in sac_list])))
            arrival_time = phases.loc[event_id, "phase_time"].min()
            begin_time = (arrival_time - pd.Timedelta(seconds=30)).replace(second=0, microsecond=0)
            end_time = begin_time + pd.Timedelta(seconds=NT//sampling_rate)
            gp.attrs["begin_time"] = begin_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            gp.attrs["end_time"] = end_time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
            gp.attrs["event_time_index"] = int(
                round((events.loc[event_id, "time"] - begin_time).total_seconds() * 100)
            )
            gp.attrs["sampling_rate"] = sampling_rate
            gp.attrs["nt"] = NT
            #gp.attrs["nx"] = len(sac_list)

            # read sac
            nx = 0
            has_station = False
            for sac in sac_list:
                st = obspy.Stream()
                try:
                    for ch in ['X', 'Y', 'Z']:
                        ch_path = f"{sac}{ch}.SAC"
                        #with fs_.open(ch_path, "rb") as f:
                        st += obspy.read(ch_path)
                except Exception as e:
                    logging.warning(f"{event_id}/{sac.split('/')[-1]} has error: {e}")
                    continue
                # st.detrend("constant")
                # st.merge(fill_value=0)
                if st[0].stats.sampling_rate != sampling_rate:
                    st.resample(sampling_rate)
                st.sort()
                components = "".join([tr.stats.channel[-1] for tr in st])
                assert components=="XYZ", f"{event_id}/{sac} has invalid components: {components}"

                array = np.zeros((3, NT))
                for i, t in enumerate(st):
                    index0 = int(
                        round(
                            (t.stats.starttime.datetime.replace(tzinfo=timezone(timedelta(hours=9))) - begin_time).total_seconds()
                            * sampling_rate
                        )
                    )
                    if index0 > 3000:
                        logging.warning(f"{event_id}/{sac} has index0 > 3000")
                        break

                    if index0 > 0:
                        i_trace = 0
                        i_array = index0
                        ll = min(len(t.data), len(array[i, i_array:]))  # data length
                    elif index0 < 0:
                        i_trace = -index0
                        i_array = 0
                        ll = min(len(t.data[i_trace:]), len(array[i, :]))
                    else:
                        i_trace = 0
                        i_array = 0
                        ll = min(len(t.data), len(array[i, :]))
                    array[i, i_array : i_array + ll] = t.data[i_trace : i_trace + ll] / 1e3 # from 1e-9 m/s to 1e-6 m/s

                if index0 > 3000:
                    continue

                station_channel_id = sac.split("/")[-1]
                network, station, instrument = station_channel_id.split(".")
                location = ''

                station_id = f"{network}.{station}.{location}"
                if station_id not in phases_by_station.index:
                    logging.warning(f"{event_id}/{station_id} station not in phase picks")
                    continue
                picks_ = phases_by_station.loc[[station_id]]
                picks_ = picks_[(picks_["phase_time"] > begin_time) & (picks_["phase_time"] < end_time)]
                if len(picks_[picks_["event_id"] == event_id]) == 0:
                    logging.warning(f"{event_id}/{station_id} no phase picks")
                    continue

                pick = picks_[picks_["event_id"] == event_id].iloc[0]  # after sort_value
                tmp = int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate))
                if (tmp - 300 < 0) or (tmp + 300 >= NT):
                    logging.warning(f"{event_id}/{station_id} picks out of time range")
                    continue

                snr = calc_snr(array, int(round((pick.phase_time - begin_time).total_seconds() * sampling_rate)))
                if max(snr) == 0:
                    logging.warning(f"{event_id}/{station_id} has zero snr")
                    continue
                (
                    phase_type,
                    phase_index,
                    phase_score,
                    phase_reliability,
                    phase_time,
                    phase_remark,
                    phase_polarity,
                    phase_picking_channel,
                    phase_event_id,
                ) = extract_pick(picks_, begin_time, sampling_rate)

                # flip the P polarity if the vertical channel is reversed
                # phase_picking_channel_x = [".".join([station_id, x]) for x in phase_picking_channel]
                # channel_dip = []
                # for x in phase_picking_channel_x:
                #     try:
                #         channel_dip.append(inv.get_channel_metadata(x, arrival_time)["dip"])
                #     except:
                #         channel_dip.append("none")
                # if 90.0 in channel_dip:
                #     logging.warning(
                #         f"{event_id}/{station_id}: {phase_picking_channel_x} {channel_dip} has 90.0 dip"
                #     )
                # phase_polarity = flip_polarity(phase_polarity, channel_dip)

                # save to hdf5
                ds = gp.create_dataset(station_channel_id, data=array, dtype=np.float32)
                ds.attrs["network"] = network
                ds.attrs["station"] = station
                ds.attrs["location"] = location
                ds.attrs["instrument"] = instrument
                ds.attrs["component"] = components
                ds.attrs["unit"] = "1e-6m/s" if instrument[0] == "V" else "1e-6m/s**2"
                ds.attrs["dt_s"] = 0.01
                # at least one channel is available
                ds.attrs["longitude"] = stations.loc[station_id, "longitude"]
                ds.attrs["latitude"] = stations.loc[station_id, "latitude"]
                ds.attrs["elevation_m"] = stations.loc[station_id, "elevation_m"]
                ds.attrs["local_depth_m"] = stations.loc[station_id, "depth_km"] # no local depth term
                ds.attrs["depth_km"] = stations.loc[station_id, "depth_km"]
                ds.attrs["azimuth"] = pick.azimuth # event -> station
                ds.attrs["back_azimuth"] = pick.back_azimuth # station -> event
                ds.attrs["distance_km"] = pick.distance_km
                ds.attrs["takeoff_angle"] = pick.takeoff_angle
                ds.attrs["snr"] = snr
                ds.attrs["phase_type"] = phase_type
                ds.attrs["phase_index"] = phase_index
                ds.attrs["phase_score"] = phase_score
                ds.attrs["phase_reliability"] = phase_reliability
                ds.attrs["phase_time"] = phase_time
                ds.attrs["phase_remark"] = phase_remark
                ds.attrs["phase_polarity"] = phase_polarity
                ds.attrs["phase_picking_channel"] = phase_picking_channel
                ds.attrs["event_id"] = phase_event_id
                main_event_idx = np.array(phase_event_id)==event_id
                main_phase_type = np.array(phase_type)[main_event_idx]
                assert len(main_phase_type) > 0 and len(main_phase_type) < 3, f"{event_id} has no main phase or more than 2 main phases: {main_phase_type}"
                main_phase_index = np.array(phase_index)[main_event_idx]
                main_phase_score = np.array(phase_score)[main_event_idx]
                main_phase_reliability = np.array(phase_reliability)[main_event_idx]
                main_phase_time = np.array(phase_time)[main_event_idx]
                main_phase_plolarity = np.array(phase_polarity)[main_event_idx]
                ds.attrs["p_phase_index"] = main_phase_index[main_phase_type=="P"][0] if "P" in main_phase_type else np.nan
                ds.attrs["s_phase_index"] = main_phase_index[main_phase_type=="S"][0] if "S" in main_phase_type else np.nan
                ds.attrs["p_phase_score"] = main_phase_score[main_phase_type=="P"][0] if "P" in main_phase_type else np.nan
                ds.attrs["s_phase_score"] = main_phase_score[main_phase_type=="S"][0] if "S" in main_phase_type else np.nan
                ds.attrs["p_phase_reliability"] = main_phase_reliability[main_phase_type=="P"][0] if "P" in main_phase_type else ''
                ds.attrs["s_phase_reliability"] = main_phase_reliability[main_phase_type=="S"][0] if "S" in main_phase_type else ''
                ds.attrs["p_phase_time"] = main_phase_time[main_phase_type=="P"][0] if "P" in main_phase_type else ''
                ds.attrs["s_phase_time"] = main_phase_time[main_phase_type=="S"][0] if "S" in main_phase_type else ''
                ds.attrs["p_phase_polarity"] = main_phase_plolarity[main_phase_type=="P"][0] if "P" in main_phase_type else ''
                ds.attrs["s_phase_polarity"] = main_phase_plolarity[main_phase_type=="S"][0] if "S" in main_phase_type else ''
                

                # TODO: phase reliability
                if (
                    len(
                        np.array(phase_type)[(np.array(phase_event_id) == event_id)]
                    )
                    > 1 # both P and S wave
                ):
                    ds.attrs["phase_status"] = "manual"
                else:
                    ds.attrs["phase_status"] = "automatic"
                has_station = True
                nx+=1

            if not has_station:
                logging.warning(f"{event_id} has no stations")
                del fp[event_id]
            gp.attrs["nx"] = nx
                
    return None


if __name__ == "__main__":
    # %%
    year_months = sorted(fs.ls(sac_path), reverse=False)
    year_months = [x.split("/")[-1] for x in year_months]
    print(f"list: {year_months}")

    ncpu = min(len(year_months), 16)
    ctx = mp.get_context("spawn")
    # ctx = mp.get_context("fork")
    with ctx.Pool(ncpu) as pool:
        for year_month in year_months:
            jdays = sorted(fs.ls(f"{sac_path}/{year_month}"), reverse=False)
            jdays = [x.split("/")[-1] for x in jdays]
            pbar = tqdm(jdays, total=len(jdays), desc=f"{year_month}", leave=True)
            
            processes = []
            for jday in jdays:
                p = pool.apply_async(convert_jday, args=(jday, catalog_path, result_path, protocol, token), callback=lambda x: pbar.update(1))
                processes.append(p)
            for p in processes:
                #try:
                out = p.get()
                if out is not None:
                    print(out)
                #except Exception as e:
                #    print(f"Error: {e}")
            pbar.close()
            
            with h5py.File(f"{result_path}/{year_month}.h5", "w") as fp:
                for jday in tqdm(jdays):
                    year, dayofyear = jday.split(".")
                    with h5py.File(f"{result_path}/{year_month}/{dayofyear}.h5", "r") as f:
                        for event_id in f:
                            f.copy(event_id, fp)
            
            os.system(f"rm -rf {result_path}/{year_month}")

    # # check hdf5
    # with h5py.File("2000.h5", "r") as fp:
    #     for event_id in fp:
    #         print(event_id)
    #         for k in sorted(fp[event_id].attrs.keys()):
    #             print(k, fp[event_id].attrs[k])
    #         for station_id in fp[event_id]:
    #             print(station_id)
    #             print(fp[event_id][station_id].shape)
    #             for k in sorted(fp[event_id][station_id].attrs.keys()):
    #                 print(k, fp[event_id][station_id].attrs[k])
    #         raise
    # raise
