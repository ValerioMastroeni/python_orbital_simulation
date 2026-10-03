import array
import requests
import pandas as pd
import numpy as np
from io import StringIO
import os


def Horizon_get_data(ID, START, STOP, STEP, OBJ_DATA = "NO", EPH_TYPE = "VECTOR", VEC_TABLE = "2", CENTER = "500@0") -> pd.DataFrame:
    """
    ID: id code of the object
    START: date of start, in format YYYY-MM-DD
    STOP: date of end, same format
    STEP: step size, given in {integer}{unit}, unit: m, h, d, mo, y
        if step is an integer without unit, the time period is divided by the step size: a 1h period with 3600 as step size will produce outputs every
        second. max accuracy is 0.5 seconds
    OBJ_DATA: YES/NO, displays data at the start, no impact on the function
    EPH_TYPE: type of table, if any other than VECTOR is used, need to remove VEC_TABLE
    VEC_TABLE: type of vector, 1 is positions, 2 is positions + velocity, 3 adds distance from center and some other data, 4-5-6 are not useful here
    CENTER: reference point, 500@0 is the baricenter of the solar system, 500@ID is the baricenter of the ID object, otherwise use an ID
    CSV: csv table output or not, leave on YES

    Y = 0, X > 0 at ~23th September
    X = 0, Y > 0 at ~21th December
    """

    url = f"https://ssd.jpl.nasa.gov/api/horizons.api?format=json&COMMAND='{ID}'&OBJ_DATA='{OBJ_DATA}'&MAKE_EPHEM='YES'&EPHEM_TYPE='{EPH_TYPE}'&VEC_TABLE='{VEC_TABLE}'&CENTER='{CENTER}'&START_TIME='{START}'&STOP_TIME='{STOP}'&CSV_FORMAT='YES'&STEP_SIZE='{STEP}'&QUANTITIES='1,9,20,23,24,29'"
    resp = requests.get(url)
    data = resp.json()
    result = data.get("result")
    #print(resp.text)

    lines = result.splitlines()

    soe_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "$$SOE":
            soe_idx = i
            break
    if soe_idx is None:
        raise RuntimeError("Non ho trovato $$SOE nel risultato")
    
    eoe_idx = None
    for i, line in enumerate(lines[soe_idx:], start=soe_idx):
        if line.strip() == "$$EOE":
            eoe_idx = i
            break
    if eoe_idx is None:
        raise RuntimeError("Non ho trovato $$EOE nel risultato")

    header_line = lines[soe_idx - 2].strip()
    if header_line.endswith(","):
        header_line = header_line[:-1]

    data_lines = []
    for line in lines[soe_idx + 1 : eoe_idx]:
        line = line.strip()
        if line.endswith(","):
            line = line[:-1]
        if line:
            data_lines.append(line)

    clean_csv = header_line + "\n" + "\n".join(data_lines)

    df = pd.read_csv(StringIO(clean_csv))

    return df

#df = Horizon_get_data("499", "2006-01-01", "2006-01-20", "1d")
#df.to_csv("horizon_system/euph2.csv", index=False)

def Horizon_get_data_cached(obj_id, start, stop, step, cache_dir="cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{obj_id}_{start}_{stop}_{step}.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)
    
    df = Horizon_get_data(obj_id, start, stop, step)
    df.to_csv(cache_path, index=False)
    
    return df