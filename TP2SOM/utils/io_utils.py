#---------------------------------------
# netcdf I/O handler
#---------------------------------------
from netCDF4 import Dataset
import netCDF4 as nc
import gzip
import shutil
import tempfile
import os

class NetCDFHandler:
    def __init__(self, filename):
        self.filename = filename
        self.ncfile = None
        self.tempfile_name = None

    def __enter__(self):
        self.ncfile = self._open_netcdf()
        return self.ncfile

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.ncfile is not None:
            self.ncfile.close()
        if self.tempfile_name is not None and os.path.exists(self.tempfile_name):
            os.unlink(self.tempfile_name)

    def _open_netcdf(self):
        if self.filename.endswith(".gz"):
            return self._open_gzipped_netcdf()
        else:
            return nc.Dataset(self.filename, 'r')

    def _open_gzipped_netcdf(self):
        with gzip.open(self.filename, 'rb') as gzfile:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                shutil.copyfileobj(gzfile, tmp)
                self.tempfile_name = tmp.name
        return nc.Dataset(self.tempfile_name, 'r')
    