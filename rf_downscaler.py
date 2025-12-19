# v1.1: Prec-DWARF with memory-mapped arrays for 64 GB systems

import numpy as np
from scipy import interpolate
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


class PrecDWARF_MemoryEfficient:
    """
    Memory-efficient Prec-DWARF implementation
    Combines He et al. (2016) methodology with memory-mapped arrays
    """
    
    def __init__(self, n_estimators=50, n_jobs=-1, wet_threshold=1.0, 
                 extreme_percentile=99.95, train_sample_ratio=0.1, 
                 temporal_chunk_size=500, random_state=42):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of trees (default: 50 from He et al. 2016)
        n_jobs : int
            Number of parallel jobs
        wet_threshold : float
            Wet day threshold (mm/day)
        extreme_percentile : float
            Extreme precipitation threshold
        train_sample_ratio : float
            Fraction of pixels sampled per timestep (default: 0.1)
        temporal_chunk_size : int
            Chunk size for upscaling
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.wet_threshold = wet_threshold
        self.extreme_percentile = extreme_percentile
        self.train_sample_ratio = train_sample_ratio
        self.temporal_chunk_size = temporal_chunk_size
        self.random_state = random_state
        
        self.rf_moderate = None
        self.rf_extreme = None
        self.extreme_threshold = None
        
    def bilinear_upscale_memmap(self, X_low, target_shape, output_file):
        """Memory-efficient bilinear upscaling (adapted from GLM code)"""
        t, h_low, w_low, c = X_low.shape
        h_high, w_high = target_shape
        
        print(f"Upscaling from {h_low}x{w_low} to {h_high}x{w_high}...")
        print(f"Using memory-mapped file: {output_file}")
        
        # Create memory-mapped array
        X_high = np.memmap(output_file, dtype='float32', mode='w+', 
                           shape=(t, h_high, w_high, c))
        
        # Coordinate grids
        x_low = np.linspace(0, 1, w_low)
        y_low = np.linspace(0, 1, h_low)
        x_high = np.linspace(0, 1, w_high)
        y_high = np.linspace(0, 1, h_high)
        
        # Process in temporal chunks
        num_chunks = int(np.ceil(t / self.temporal_chunk_size))
        
        for chunk_idx in tqdm(range(num_chunks), desc="Upscaling chunks"):
            start_t = chunk_idx * self.temporal_chunk_size
            end_t = min((chunk_idx + 1) * self.temporal_chunk_size, t)
            
            for ch in range(c):
                for ti in range(start_t, end_t):
                    interp_func = interpolate.RectBivariateSpline(
                        y_low, x_low, X_low[ti, :, :, ch], kx=1, ky=1
                    )
                    X_high[ti, :, :, ch] = interp_func(y_high, x_high)
            
            X_high.flush()
        
        print(f"Memory-mapped array created: {os.path.getsize(output_file) / 1e9:.2f} GB")
        return X_high
    
    def upscale_precipitation(self, y_high, coarse_shape):
        """Upscale high-res precipitation to coarse (box averaging)"""
        print(f"\nUpscaling precipitation from {y_high.shape[1:3]} to {coarse_shape}...")
        
        t, h_high, w_high, c = y_high.shape
        h_coarse, w_coarse = coarse_shape
        
        factor_h = h_high // h_coarse
        factor_w = w_high // w_coarse
        
        y_coarse = np.zeros((t, h_coarse, w_coarse, c), dtype=np.float32)
        
        for ti in tqdm(range(t), desc="Upscaling precipitation"):
            for i in range(h_coarse):
                for j in range(w_coarse):
                    y_coarse[ti, i, j, 0] = np.mean(
                        y_high[ti, i*factor_h:(i+1)*factor_h, 
                               j*factor_w:(j+1)*factor_w, 0]
                    )
        
        print(f"Original mean: {y_high.mean():.3f} mm/day")
        print(f"Upscaled mean: {y_coarse.mean():.3f} mm/day")
        
        return y_coarse
    
    def compute_dry_drift(self, precip_coarse):
        """Compute distance to closest dry cell (key innovation!)"""
        print("\nComputing 'dry drift'...")
        
        t, h, w, _ = precip_coarse.shape
        dry_drift = np.zeros((t, h, w, 1), dtype=np.float32)
        
        for ti in tqdm(range(t), desc="Computing dry drift"):
            wet_mask = precip_coarse[ti, :, :, 0] > self.wet_threshold
            
            if wet_mask.any():
                coords = np.argwhere(~wet_mask)  # dry cells
                wet_coords = np.argwhere(wet_mask)  # wet cells
                
                if len(coords) > 0:
                    distances = cdist(wet_coords, coords, metric='euclidean')
                    min_distances = distances.min(axis=1)
                    
                    for idx, (i, j) in enumerate(wet_coords):
                        dry_drift[ti, i, j, 0] = min_distances[idx]
        
        return dry_drift
    
    def get_adjacent_cells(self, precip_disagg):
        """Get precipitation from 4 adjacent cells"""
        print("\nExtracting adjacent cell precipitation...")
        
        t, h, w, _ = precip_disagg.shape
        adjacent = np.zeros((t, h, w, 4), dtype=np.float32)
        
        # Up, down, left, right (with wrap-around)
        adjacent[:, :, :, 0] = np.roll(precip_disagg[:, :, :, 0], 1, axis=1)   # up
        adjacent[:, :, :, 1] = np.roll(precip_disagg[:, :, :, 0], -1, axis=1)  # down
        adjacent[:, :, :, 2] = np.roll(precip_disagg[:, :, :, 0], 1, axis=2)   # left
        adjacent[:, :, :, 3] = np.roll(precip_disagg[:, :, :, 0], -1, axis=2)  # right
        
        return adjacent
    
    def prepare_covariates_memmap(self, X_low, y_high, S_high, save_dir, prefix='train'):
        """
        Prepare covariates using memory-mapped arrays
        Following He et al. (2016) + GLM memory efficiency
        """
        print("\n" + "="*80)
        print(f"PREPARING COVARIATES ({prefix.upper()}) - MEMORY-MAPPED")
        print("="*80)
        
        t = X_low.shape[0]
        h_high, w_high = y_high.shape[1:3]
        
        # 1. Upscale high-res precipitation to coarse
        y_coarse = self.upscale_precipitation(y_high, (32, 32))
        
        # 2. Disaggregate coarse precip back to high-res (to memory-mapped file)
        print("\n--- Disaggregating coarse precipitation ---")
        precip_disagg_file = os.path.join(save_dir, f'{prefix}_precip_disagg.dat')
        precip_disagg = self.bilinear_upscale_memmap(y_coarse, (h_high, w_high), 
                                                     precip_disagg_file)
        
        # 3. Get adjacent cell precipitation (in-memory, small)
        adjacent = self.get_adjacent_cells(precip_disagg)
        
        # 4. Compute dry drift and disaggregate
        dry_drift_coarse = self.compute_dry_drift(y_coarse)
        print("\n--- Disaggregating dry drift ---")
        dry_drift_file = os.path.join(save_dir, f'{prefix}_dry_drift.dat')
        dry_drift = self.bilinear_upscale_memmap(dry_drift_coarse, (h_high, w_high),
                                                 dry_drift_file)
        
        # 5. Disaggregate atmospheric fields (to memory-mapped file)
        print("\n--- Disaggregating atmospheric fields ---")
        atmos_file = os.path.join(save_dir, f'{prefix}_atmos.dat')
        atmos_fine = self.bilinear_upscale_memmap(X_low, (h_high, w_high), atmos_file)
        
        # 6. Prepare elevation
        if S_high.shape[0] == 1:
            elev = np.repeat(S_high, t, axis=0)
        else:
            elev = S_high
        
        # 7. Stack all covariates in memory-mapped file
        print("\n--- Stacking all covariates ---")
        covariates_file = os.path.join(save_dir, f'{prefix}_covariates.dat')
        covariates = np.memmap(covariates_file, dtype='float32', mode='w+',
                               shape=(t, h_high, w_high, 34))
        
        # Stack in chunks to manage memory
        chunk_size = 500
        for i in tqdm(range(0, t, chunk_size), desc="Stacking covariates"):
            end_i = min(i + chunk_size, t)
            
            covariates[i:end_i, :, :, 0:1] = precip_disagg[i:end_i, :, :, :]      # 1
            covariates[i:end_i, :, :, 1:5] = adjacent[i:end_i, :, :, :]           # 4
            covariates[i:end_i, :, :, 5:6] = dry_drift[i:end_i, :, :, :]          # 1
            covariates[i:end_i, :, :, 6:33] = atmos_fine[i:end_i, :, :, :]        # 27
            covariates[i:end_i, :, :, 33:34] = elev[i:end_i, :, :, :]             # 1
        
        covariates.flush()
        
        # Clean up intermediate files
        del precip_disagg, dry_drift, atmos_fine
        
        print(f"\nCovariates prepared: {covariates.shape}")
        print(f"  Features: central(1) + adjacent(4) + dry_drift(1) + atmos(27) + elev(1) = 34")
        
        return covariates, y_high
    
    def sample_training_data(self, covariates, target):
        """Sample training data (10% per timestep)"""
        print("\n" + "="*80)
        print("SAMPLING TRAINING DATA")
        print("="*80)
        
        t, h, w, c = covariates.shape
        total_pixels = h * w
        n_samples_per_time = int(total_pixels * self.train_sample_ratio)
        
        print(f"Sampling {self.train_sample_ratio*100}% of pixels per timestep")
        print(f"Total pixels: {total_pixels:,}")
        print(f"Samples per timestep: {n_samples_per_time:,}")
        
        np.random.seed(self.random_state)
        
        X_train_list = []
        y_train_list = []
        
        for ti in tqdm(range(t), desc="Sampling timesteps"):
            # Random sample indices
            sampled_indices = np.random.choice(total_pixels, n_samples_per_time, 
                                              replace=False)
            
            sampled_i = sampled_indices // w
            sampled_j = sampled_indices % w
            
            # Extract from memory-mapped arrays
            X_train_list.append(covariates[ti, sampled_i, sampled_j, :])
            y_train_list.append(target[ti, sampled_i, sampled_j, 0])
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        print(f"\nTraining data: X={X_train.shape}, y={y_train.shape}")
        print(f"Wet frequency: {(y_train > self.wet_threshold).mean()*100:.2f}%")
        print(f"Memory: {(X_train.nbytes + y_train.nbytes) / 1e9:.2f} GB")
        
        return X_train, y_train
    
    def fit(self, X_train, y_train):
        """Fit Double RF"""
        print("\n" + "="*80)
        print("FITTING DOUBLE RANDOM FOREST")
        print("="*80)
        
        # Calculate extreme threshold
        wet_vals = y_train[y_train > self.wet_threshold]
        if len(wet_vals) > 0:
            self.extreme_threshold = np.percentile(wet_vals, self.extreme_percentile)
        else:
            self.extreme_threshold = np.percentile(y_train, 99)
        
        print(f"Extreme threshold (>{self.extreme_percentile}%): {self.extreme_threshold:.2f} mm/day")
        
        # Separate samples
        moderate_mask = y_train <= self.extreme_threshold
        extreme_mask = y_train > self.extreme_threshold
        
        print(f"\nSample distribution:")
        print(f"  Moderate: {moderate_mask.sum():,} ({moderate_mask.mean()*100:.2f}%)")
        print(f"  Extreme: {extreme_mask.sum():,} ({extreme_mask.mean()*100:.2f}%)")
        
        # Fit moderate RF
        print(f"\n[1/2] Training RF for moderate precipitation...")
        start_time = time.time()
        
        self.rf_moderate = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features='sqrt',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1
        )
        
        self.rf_moderate.fit(X_train[moderate_mask], y_train[moderate_mask])
        print(f"Moderate RF trained in {time.time() - start_time:.1f}s")
        
        # Fit extreme RF if enough samples
        if extreme_mask.sum() > 100:
            print(f"\n[2/2] Training RF for extreme precipitation...")
            start_time = time.time()
            
            self.rf_extreme = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features='sqrt',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1
            )
            
            self.rf_extreme.fit(X_train[extreme_mask], y_train[extreme_mask])
            print(f"Extreme RF trained in {time.time() - start_time:.1f}s")
        else:
            print(f"\n[2/2] Skipping extreme RF (only {extreme_mask.sum()} samples)")
            self.rf_extreme = None
        
        return self
    
    def predict(self, covariates):
        """Predict using Double RF"""
        print("\n" + "="*80)
        print("PREDICTING WITH DOUBLE RF")
        print("="*80)
        
        t, h, w, c = covariates.shape
        predictions = np.zeros((t, h, w), dtype=np.float32)
        
        print(f"Predicting for {t} timesteps at {h}x{w} resolution...")
        
        # Predict in temporal chunks to manage memory
        chunk_size = 100
        
        for chunk_start in tqdm(range(0, t, chunk_size), desc="Predicting chunks"):
            chunk_end = min(chunk_start + chunk_size, t)
            
            # Load chunk from memory-mapped array
            X_chunk = covariates[chunk_start:chunk_end].reshape(-1, c)
            
            # Predict with moderate RF
            pred_moderate = self.rf_moderate.predict(X_chunk)
            
            # Apply extreme RF where needed
            if self.rf_extreme is not None:
                extreme_mask = pred_moderate > (self.extreme_threshold * 0.8)
                if extreme_mask.any():
                    pred_extreme = self.rf_extreme.predict(X_chunk[extreme_mask])
                    pred_moderate[extreme_mask] = pred_extreme
            
            # Reshape and store
            predictions[chunk_start:chunk_end] = pred_moderate.reshape(
                chunk_end - chunk_start, h, w
            )
        
        # Ensure non-negative
        predictions = np.maximum(predictions, 0)
        
        print(f"\nPrediction statistics:")
        print(f"  Range: [{predictions.min():.3f}, {predictions.max():.3f}] mm/day")
        print(f"  Mean: {predictions.mean():.3f} mm/day")
        print(f"  Wet frequency: {(predictions > self.wet_threshold).mean()*100:.2f}%")
        
        return predictions
    
    def save_predictions(self, predictions, save_path):
        """Save predictions"""
        print(f"\nSaving predictions to {save_path}...")
        np.savez_compressed(save_path, predictions=predictions)
        print(f"Saved! Size: {os.path.getsize(save_path) / 1e6:.1f} MB")


def load_data_from_files(data_path):
    """Load data (reuse from GLM code)"""
    print("\n" + "="*80)
    print("LOADING DATA FROM FILES")
    print("="*80)
    print(f"Data path: {data_path}")
    
    inputs_channels = {
        'z850': f'{data_path}/IND32M_ERA5_100_Z850_DAY_1979_2023_RBIL_STD.npy',
        'z700': f'{data_path}/IND32M_ERA5_100_Z700_DAY_1979_2023_RBIL_STD.npy',
        'z500': f'{data_path}/IND32M_ERA5_100_Z500_DAY_1979_2023_RBIL_STD.npy',
        'z250': f'{data_path}/IND32M_ERA5_100_Z250_DAY_1979_2023_RBIL_STD.npy',
        'q850': f'{data_path}/IND32M_ERA5_100_Q850_DAY_1979_2023_RBIL_STD.npy',
        'q700': f'{data_path}/IND32M_ERA5_100_Q700_DAY_1979_2023_RBIL_STD.npy',
        'q500': f'{data_path}/IND32M_ERA5_100_Q500_DAY_1979_2023_RBIL_STD.npy',
        'q250': f'{data_path}/IND32M_ERA5_100_Q250_DAY_1979_2023_RBIL_STD.npy',
        't850': f'{data_path}/IND32M_ERA5_100_T850_DAY_1979_2023_RBIL_STD.npy',
        't700': f'{data_path}/IND32M_ERA5_100_T700_DAY_1979_2023_RBIL_STD.npy',
        't500': f'{data_path}/IND32M_ERA5_100_T500_DAY_1979_2023_RBIL_STD.npy',
        't250': f'{data_path}/IND32M_ERA5_100_T250_DAY_1979_2023_RBIL_STD.npy',
        'u850': f'{data_path}/IND32M_ERA5_100_U850_DAY_1979_2023_RBIL_STD.npy',
        'u700': f'{data_path}/IND32M_ERA5_100_U700_DAY_1979_2023_RBIL_STD.npy',
        'u500': f'{data_path}/IND32M_ERA5_100_U500_DAY_1979_2023_RBIL_STD.npy',
        'u250': f'{data_path}/IND32M_ERA5_100_U250_DAY_1979_2023_RBIL_STD.npy',
        'v850': f'{data_path}/IND32M_ERA5_100_V850_DAY_1979_2023_RBIL_STD.npy',
        'v700': f'{data_path}/IND32M_ERA5_100_V700_DAY_1979_2023_RBIL_STD.npy',
        'v500': f'{data_path}/IND32M_ERA5_100_V500_DAY_1979_2023_RBIL_STD.npy',
        'v250': f'{data_path}/IND32M_ERA5_100_V250_DAY_1979_2023_RBIL_STD.npy',
        'w850': f'{data_path}/IND32M_ERA5_100_W850_DAY_1979_2023_RBIL_STD.npy',
        'w700': f'{data_path}/IND32M_ERA5_100_W700_DAY_1979_2023_RBIL_STD.npy',
        'w500': f'{data_path}/IND32M_ERA5_100_W500_DAY_1979_2023_RBIL_STD.npy',
        'w250': f'{data_path}/IND32M_ERA5_100_W250_DAY_1979_2023_RBIL_STD.npy',
        'msl': f'{data_path}/IND32M_ERA5_100_MSL_DAY_1979_2023_RBIL_STD.npy',
        't2m': f'{data_path}/IND32M_ERA5_100_T2M_DAY_1979_2023_RBIL_STD.npy',
        'cape': f'{data_path}/IND32M_ERA5_100_CAPE_DAY_1979_2023_RBIL_STD.npy',
    }
    
    static_file = f'{data_path}/IND32M_GTOP_010_ELEV_DAY_1979_2023_LOGN.npy'
    target_file = f'{data_path}/IND32M_MSWX_010_PREC_DAY_1979_2023.npy'
    
    print(f"\nLoading {len(inputs_channels)} atmospheric channels...")
    X_list = []
    for i, (var_name, file_path) in enumerate(inputs_channels.items()):
        print(f"  [{i+1}/{len(inputs_channels)}] {var_name}...", end=' ')
        X_list.append(np.load(file_path))
        print(f"✓")
    
    X = np.stack(X_list, axis=-1)
    S = np.load(static_file)
    y = np.load(target_file)
    
    if S.ndim == 3:
        S = np.expand_dims(S, axis=-1)
    if y.ndim == 3:
        y = np.expand_dims(y, axis=-1)
    
    print(f"\nData loaded: X{X.shape}, S{S.shape}, y{y.shape}")
    return X, y, S


def main_pipeline(data_path, save_dir, train_size=11323, val_size=3287):
    """Complete Prec-DWARF pipeline with memory-mapped arrays"""
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PREC-DWARF: RANDOM FOREST BASELINE (MEMORY-EFFICIENT)")
    print("Following He et al. (2016) with memory-mapped arrays")
    print("="*80)
    
    # Load data
    X_low, y_high, S_high = load_data_from_files(data_path)
    
    # Split
    X_train = X_low[:train_size]
    y_train = y_high[:train_size]
    S_train = S_high[:1] if S_high.shape[0] == 1 else S_high[:train_size]
    
    X_val = X_low[train_size:train_size+val_size]
    y_val = y_high[train_size:train_size+val_size]
    S_val = S_high[:1] if S_high.shape[0] == 1 else S_high[train_size:train_size+val_size]
    
    # Initialize
    downscaler = PrecDWARF_MemoryEfficient(
        n_estimators=50,
        n_jobs=-1,
        wet_threshold=1.0,
        extreme_percentile=99.95,
        train_sample_ratio=0.1,
        temporal_chunk_size=500,
        random_state=42
    )
    
    # Prepare training covariates
    covariates_train, target_train = downscaler.prepare_covariates_memmap(
        X_train, y_train, S_train, save_dir, prefix='train'
    )
    
    # Sample training data
    X_train_sampled, y_train_sampled = downscaler.sample_training_data(
        covariates_train, target_train
    )
    
    # Fit RF
    downscaler.fit(X_train_sampled, y_train_sampled)
    
    # Prepare validation covariates
    covariates_val, target_val = downscaler.prepare_covariates_memmap(
        X_val, y_val, S_val, save_dir, prefix='val'
    )
    
    # Predict
    predictions = downscaler.predict(covariates_val)
    
    # Save
    pred_path = os.path.join(save_dir, 'rf_predictions.npz')
    downscaler.save_predictions(predictions, pred_path)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED!")
    print("="*80)
    
    return downscaler, predictions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        downscaler, predictions = main_pipeline(
            data_path=sys.argv[1],
            save_dir=sys.argv[2],
            train_size=11323,
            val_size=3287
        )
    else:
        print("Usage: python rf_downscaler.py <DATA_PATH> <SAVE_DIR>")
