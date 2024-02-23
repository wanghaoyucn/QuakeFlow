import os
import sys
from glob import glob

def download_hypocenter(year, output_dir="dataset/hypocenter"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(f"{output_dir}/h{year}"):
        print(f"File h{year} already exists, skipping download")
        return
    print(f"Downloading hypocenter data for {year}")
    # Download the file
    if not os.path.exists(f"{output_dir}/h{year}.zip"):
        print(f"Downloading h{year}.zip")
        os.system(f"wget https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/hypo/h{year}.zip -P {output_dir}")
    else:
        print(f"File h{year}.zip already exists, skipping download")
    os.system(f"unzip {os.path.join(output_dir, f'h{year}.zip')} -d {output_dir}")
    
    return  
    
def download_mechanism(year, month, output_dir="dataset/mechanism"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(f"{output_dir}/mc{year}{month:02}"):
        print(f"File mc{year}{month:02} already exists, skipping download")
        return
    if not os.path.exists(f"{output_dir}/mc{year}{month:02}.zip"):
        print(f"Downloading mc{year}{month:02}.zip")
        os.system(f"wget https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/mech/mc{year}{month:02}.zip -P {output_dir}")
    else:
        print(f"File mc{year}{month:02}.zip already exists, skipping download")
    os.system(f"unzip {os.path.join(output_dir, f'mc{year}{month:02}.zip')} -d {output_dir}")

def download_arrivaltime(year, month, output_dir="dataset/arrivaltime"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if os.path.exists(f"{output_dir}/d{year}{month:02}"):
        print(f"File d{year}{month:02} already exists, skipping download")
        return
    if not os.path.exists(f"{output_dir}/d{year}{month:02}.zip"):
        print(f"Downloading d{year}{month:02}.zip")
        os.system(f"wget https://www.data.jma.go.jp/svd/eqev/data/bulletin/data/deck/d{year}{month:02}.zip -P {output_dir}")
    else:
        print(f"File d{year}{month:02}.zip already exists, skipping download")
    os.system(f"unzip {os.path.join(output_dir, f'd{year}{month:02}.zip')} -d {output_dir}")
    # merge all files into one
    files = glob(f"{output_dir}/d{year}{month:02}*")
    # remove the zip file
    files = [f for f in files if not f.endswith(".zip")]
    with open(f"{output_dir}/d{year}{month:02}", "w") as outfile:
        for fname in files:
            with open(fname, "r") as infile:
                outfile.write(infile.read())
    for fname in files:
        os.remove(fname)

if __name__ == "__main__":
    result_path = "dataset/"
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    years = range(2021, 2022)
    for year in years:
        download_hypocenter(year, output_dir=os.path.join(result_path, "hypocenter"))
        for month in range(1, 2):
            download_mechanism(year, month, output_dir=os.path.join(result_path, "mechanism"))
            download_arrivaltime(year, month, output_dir=os.path.join(result_path, "arrivaltime"))