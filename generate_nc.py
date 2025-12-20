import logging
import numpy as np
from ai4klima.utils import build_netcdf_from_array
from pathlib import Path

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("NetCDFBuilder")

# -----------------------------------------------------------------------------
# Paths and parameters
# -----------------------------------------------------------------------------
data_path = Path(
    "/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/EXPSDATA/P03B/P03B.QXX.RFBIL.E5AUXV.RAW/val_predictions_upscaled_320x320.npy"
)

ref_ds_path = Path(
    "/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/DATASETS/DATASET_IND32M/MASTER/grids/IND32M_GTOP_010_ELEV.nc"
)

file_name = "val_predictions_upscaled_320x320.nc"

start_date = "2010-01-01"
end_date   = "2023-12-31"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
logger.info("Loading prediction array from: %s", data_path)

try:
    prec = np.load(data_path)
    logger.info(
        "Prediction array loaded successfully | shape=%s | dtype=%s",
        prec.shape, prec.dtype
    )
except Exception as e:
    logger.exception("Failed to load NumPy array")
    raise

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
logger.info("Performing sanity checks on input data")

if prec.ndim != 3:
    logger.error("Expected 3D array (time, lat, lon), got ndim=%d", prec.ndim)
    raise ValueError("Invalid input array dimensions")

logger.info("Sanity checks passed")

# -----------------------------------------------------------------------------
# Build NetCDF
# -----------------------------------------------------------------------------
logger.info("Building NetCDF file")

try:
    build_netcdf_from_array(
        prec,
        ref_ds=str(ref_ds_path),
        varname="prec",
        start_date=start_date,
        end_date=end_date,
        calendar="standard",
        save_dir="./..",
        filename=file_name,
        delete_existing_file=True,
    )
    logger.info("NetCDF file created successfully: %s.nc", file_name)

except Exception as e:
    logger.exception("NetCDF creation failed")
    raise

logger.info("Process completed successfully")
