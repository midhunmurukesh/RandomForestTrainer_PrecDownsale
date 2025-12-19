from ai4klima.utils import build_netcdf_from_array
import numpy as np

data_path = "/home/midhunm/KNRMD/RFTUNE_OUT/val_predictions_upscaled_320x320.npy"
prec = np.load(data_path)
print(prec.shape)  # (365, 320, 320)

build_netcdf_from_array(
    prec,
    ref_ds="/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/DATASETS/DATASET_IND32M/MASTER/grids/IND32M_GTOP_010_ELEV.nc",
    varname='tp',
    start_date="2010-01-01",
    end_date="2023-12-31",
    calendar='standard',
    save_dir="/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/EXPSDATA/P03B/RF_DATA",
    filename="RF_010_PREC_DAY_2010_2023",
    delete_existing_file=True,
)
