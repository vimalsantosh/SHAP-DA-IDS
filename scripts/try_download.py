import urllib.request

urls = [
    "https://github.com/dephiloper/ds2os/raw/master/ds2os.csv",
    "https://github.com/rahulvigneswaran/IoT-Datasets/raw/main/DS2OS/DS2OS_traffic_traces.csv",
    "https://github.com/Dataset-Researcher/DS2OS/raw/main/DS2OS_traffic_traces.csv",
]
for url in urls:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urllib.request.urlopen(req, timeout=10)
        ct = resp.headers.get("Content-Type", "unknown")
        print(f"SUCCESS: {url} - ContentType: {ct}")
    except Exception as e:
        print(f"FAILED: {url} - {type(e).__name__}: {e}")
