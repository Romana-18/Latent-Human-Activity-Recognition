"""Download + unzip UCI HAR (raw inertial signals). Usage: python data/download_uci_har.py"""
import os
import urllib.request
import zipfile

URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw", "uci_har")
os.makedirs(OUT, exist_ok=True)
zpath = os.path.join(OUT, "uci_har.zip")
if not os.path.exists(zpath):
    print("Downloading UCI HAR ...")
    urllib.request.urlretrieve(URL, zpath)
with zipfile.ZipFile(zpath) as z:
    z.extractall(OUT)
# the UCI archive contains a nested zip -> extract it too
for f in os.listdir(OUT):
    if f.endswith(".zip") and f != "uci_har.zip":
        with zipfile.ZipFile(os.path.join(OUT, f)) as z:
            z.extractall(OUT)
print("Done ->", OUT)
