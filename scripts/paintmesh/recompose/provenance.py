"""Per-process reuse of SHA256 for files whose identity and metadata are unchanged."""
from pathlib import Path
import time
from .io import fingerprint as hash_file


class Fingerprinter:
    def __init__(self):self.cache={}

    @staticmethod
    def identity(path):
        stat=path.stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)

    def __call__(self,path):
        path=Path(path).resolve(strict=True);key=self.identity(path)
        previous=self.cache.get(path)
        if previous and previous[0]==key and previous[2]:return dict(previous[1])
        # Some ext filesystems expose ctime at clock-tick granularity. A new
        # file can be rewritten and have its mtime restored in the same tick.
        # Cache only files already stable beyond that interval when hashed.
        reusable=time.time_ns()-key[-1]>2_000_000_000
        value=hash_file(path)
        if self.identity(path)!=key:raise ValueError(f'Artifact changed while hashing: {path}')
        self.cache[path]=(key,value,reusable)
        return dict(value)
