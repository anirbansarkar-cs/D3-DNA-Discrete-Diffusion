from pathlib import Path
import os
import pwd
import h5py
import numpy as np

def get_owner(path):
    try:
        return pwd.getpwuid(os.stat(path).st_uid).pw_name
    except Exception:
        return None

def scan_h5(path):
    datasets = []
    with h5py.File(path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                datasets.append({
                    "key": name,
                    "shape": str(obj.shape),
                    "dtype": str(obj.dtype)
                })
        f.visititems(visitor)
    return datasets

def scan_npz(path):
    datasets = []
    with np.load(path, allow_pickle=False) as f:
        for k in f.files:
            arr = f[k]
            datasets.append({
                "key": k,
                "shape": str(arr.shape),
                "dtype": str(arr.dtype)
            })
    return datasets

def scan_file(path):
    stat = os.stat(path)
    file_type = path.suffix.lower()

    result = {
        "path": str(path),
        "file_type": file_type,
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "owner": get_owner(path),
        "datasets": []
    }

    if file_type in [".h5", ".hdf5"]:
        result["datasets"] = scan_h5(path)
    elif file_type == ".npz":
        result["datasets"] = scan_npz(path)

    return result

