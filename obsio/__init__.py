import os

try:
    LOCAL_DATA_PATH = os.environ['OBSIO_DATA']
except KeyError:
    import tempfile
    LOCAL_DATA_PATH = tempfile.gettempdir()

from .factory import ObsIoFactory
from .util.misc import BBox
from .providers.generic import ObsIO
from .providers.hdf import HdfObsIO
from .providers.multi import MultiObsIO