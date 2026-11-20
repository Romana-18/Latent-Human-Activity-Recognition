"""Download WISDM v1.1 (second dataset). Usage: python data/download_wisdm.py
If the download fails (site down), download manually from
https://www.cis.fordham.edu/wisdm/dataset.php and put WISDM_ar_v1.1_raw.txt in data/raw/wisdm/"""
import os
import tarfile
import urllib.request

URL = "https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "wisdm")
os.makedirs(OUT, exist_ok=True)
tpath = os.path.join(OUT, "wisdm.tar.gz")
if not os.path.exists(tpath):
    print("Downloading WISDM ...")
    urllib.request.urlretrieve(URL, tpath)
with tarfile.open(tpath) as t:
    for m in t.getmembers():
        if m.name.endswith("WISDM_ar_v1.1_raw.txt"):
            m.name = "WISDM_ar_v1.1_raw.txt"
            t.extract(m, OUT)
print("Done ->", OUT)
