# v1.2: Prec-DWARF with scipy.zoom optimization (10× faster upscaling) 

import numpy as np
from scipy.ndimage import zoom
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
import warnings
import os
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')


class PrecDWARF_Fast:
    """
    Fast memory-efficient Prec-DWARF implementation
    Uses scipy.zoom for 10× faster bilinear interpolation
    Following He et al. (2016) Water Resources Research
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
            Number of parallel jobs (-1 for all cores)
        wet_threshold : float
            Wet day threshold in mm/day
        extreme_percentile : float
            Extreme precipitation threshold percentile
        train_sample_ratio : float
            Fraction of pixels sampled per timestep (default: 0.1)
        temporal_chunk_size : int
            Chunk size for processing timesteps
        random_state : int
            Random seed for reproducibility
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
        
    def bilinear_upscale_fast(self, X_low, target_shape, output_file):
        """
        Fast bilinear upscaling using scipy.zoom with memory-mapped output
        10× faster than RectBivariateSpline approach
        
        Parameters:
        -----------
        X_low : array (t, h_low, w_low, c)
        target_shape : tuple (h_high, w_high)
        output_file : str - path to memory-mapped file
        
        Returns:
        --------
        X_high : memory-mapped array (t, h_high, w_high, c)
        """
        t, h_low, w_low, c = X_low.shape
        h_high, w_high = target_shape
        
        print(f"\nUpscaling from {h_low}x{w_low} to {h_high}x{w_high} using scipy.zoom...")
        print(f"Memory-mapped file: {output_file}")
        
        # Calculate zoom factors (keep time and channels, zoom spatial)
        zoom_factors = (1, h_high/h_low, w_high/w_low, 1)
        
        # Create memory-mapped array
        X_high = np.memmap(output_file, dtype='float32', mode='w+',
                           shape=(t, h_high, w_high, c))
        
        # Process in temporal chunks
        num_chunks = int(np.ceil(t / self.temporal_chunk_size))
        
        for chunk_idx in tqdm(range(num_chunks), desc="Upscaling chunks"):
            start_t = chunk_idx * self.temporal_chunk_size
            end_t = min((chunk_idx + 1) * self.temporal_chunk_size, t)
            
            # Zoom this chunk (bilinear interpolation with order=1)
            X_high[start_t:end_t] = zoom(X_low[start_t:end_t], zoom_factors, 
                                         order=1, mode='nearest')
            
            # Flush to disk periodically
            if (chunk_idx + 1) % 5 == 0:
                X_high.flush()
        
        X_high.flush()
        print(f"Memory-mapped array created: {os.path.getsize(output_file) / 1e9:.2f} GB")
        
        return X_high
    
    def upscale_precipitation(self, y_high, coarse_shape):
        """
        Upscale high-res precipitation to coarse using box averaging
        Preserves total precipitation mass
        
        Parameters:
        -----------
        y_high : array (t, 320, 320, 1)
        coarse_shape : tuple (32, 32)
        
        Returns:
        --------
        y_coarse : array (t, 32, 32, 1)
        """
        print(f"\nUpscaling precipitation from {y_high.shape[1:3]} to {coarse_shape}...")
        
        t, h_high, w_high, c = y_high.shape
        h_coarse, w_coarse = coarse_shape
        
        factor_h = h_high // h_coarse
        factor_w = w_high // w_coarse
        
        y_coarse = np.zeros((t, h_coarse, w_coarse, c), dtype=np.float32)
        
        # Box averaging (exact, preserves mass)
        for ti in tqdm(range(t), desc="Upscaling precipitation"):
            for i in range(h_coarse):
                for j in range(w_coarse):
                    y_coarse[ti, i, j, 0] = np.mean(
                        y_high[ti, i*factor_h:(i+1)*factor_h, 
                               j*factor_w:(j+1)*factor_w, 0]
                    )
        
        print(f"Original mean: {y_high.mean():.3f} mm/day")
        print(f"Upscaled mean: {y_coarse.mean():.3f} mm/day (mass preserved!)")
        
        return y_coarse
    
    def compute_dry_drift(self, precip_coarse):
        """
        Compute distance to closest dry cell (key innovation from He et al. 2016)
        
        Parameters:
        -----------
        precip_coarse : array (t, 32, 32, 1)
        
        Returns:
        --------
        dry_drift : array (t, 32, 32, 1) - distance in grid cells
        """
        print("\nComputing 'dry drift' (distance to closest dry cell)...")
        
        t, h, w, _ = precip_coarse.shape
        dry_drift = np.zeros((t, h, w, 1), dtype=np.float32)
        
        for ti in tqdm(range(t), desc="Computing dry drift"):
            wet_mask = precip_coarse[ti, :, :, 0] > self.wet_threshold
            
            if wet_mask.any():
                # Find dry and wet cell coordinates
                dry_coords = np.argwhere(~wet_mask)
                wet_coords = np.argwhere(wet_mask)
                
                if len(dry_coords) > 0:
                    # Calculate Euclidean distance to nearest dry cell
                    distances = cdist(wet_coords, dry_coords, metric='euclidean')
                    min_distances = distances.min(axis=1)
                    
                    # Assign distances to wet cells
                    for idx, (i, j) in enumerate(wet_coords):
                        dry_drift[ti, i, j, 0] = min_distances[idx]
        
        print(f"Dry drift statistics: min={dry_drift.min():.2f}, max={dry_drift.max():.2f}")
        
        return dry_drift
    
    def get_adjacent_cells(self, precip_disagg):
        """
        Extract precipitation from 4 adjacent cells (up, down, left, right)
        
        Parameters:
        -----------
        precip_disagg : array (t, 320, 320, 1)
        
        Returns:
        --------
        adjacent : array (t, 320, 320, 4)
        """
        print("\nExtracting adjacent cell precipitation...")
        
        t, h, w, _ = precip_disagg.shape
        adjacent = np.zeros((t, h, w, 4), dtype=np.float32)
        
        # Spatial shifts with periodic boundary conditions
        adjacent[:, :, :, 0] = np.roll(precip_disagg[:, :, :, 0], 1, axis=1)   # up
        adjacent[:, :, :, 1] = np.roll(precip_disagg[:, :, :, 0], -1, axis=1)  # down
        adjacent[:, :, :, 2] = np.roll(precip_disagg[:, :, :, 0], 1, axis=2)   # left
        adjacent[:, :, :, 3] = np.roll(precip_disagg[:, :, :, 0], -1, axis=2)  # right
        
        return adjacent
    
    def prepare_covariates_memmap(self, X_low, y_high, S_high, save_dir, prefix='train'):
        """
        Prepare all 34 covariates using memory-mapped arrays
        Following He et al. (2016) methodology
        
        Parameters:
        -----------
        X_low : array (t, 32, 32, 27) - atmospheric fields
        y_high : array (t, 320, 320, 1) - precipitation target
        S_high : array (1 or t, 320, 320, 1) - elevation
        save_dir : str - directory for memory-mapped files
        prefix : str - 'train' or 'val'
        
        Returns:
        --------
        covariates : memory-mapped array (t, 320, 320, 34)
        target : array (t, 320, 320, 1)
        """
        print("\n" + "="*80)
        print(f"PREPARING COVARIATES ({prefix.upper()}) - MEMORY-MAPPED")
        print("="*80)
        
        t = X_low.shape[0]
        h_high, w_high = y_high.shape[1:3]
        
        # Step 1: Upscale precipitation to coarse (box averaging)
        y_coarse = self.upscale_precipitation(y_high, (32, 32))
        
        # Step 2: Disaggregate coarse precipitation to fine (bilinear)
        print("\n--- Disaggregating coarse precipitation ---")
        precip_disagg_file = os.path.join(save_dir, f'{prefix}_precip_disagg.dat')
        precip_disagg = self.bilinear_upscale_fast(y_coarse, (h_high, w_high), 
                                                   precip_disagg_file)
        
        # Step 3: Extract adjacent cell precipitation (in-memory)
        adjacent = self.get_adjacent_cells(precip_disagg)
        
        # Step 4: Compute dry drift at coarse resolution
        dry_drift_coarse = self.compute_dry_drift(y_coarse)
        
        # Step 5: Disaggregate dry drift to fine resolution
        print("\n--- Disaggregating dry drift ---")
        dry_drift_file = os.path.join(save_dir, f'{prefix}_dry_drift.dat')
        dry_drift = self.bilinear_upscale_fast(dry_drift_coarse, (h_high, w_high),
                                               dry_drift_file)
        
        # Step 6: Disaggregate atmospheric fields to fine resolution
        print("\n--- Disaggregating atmospheric fields (27 channels) ---")
        atmos_file = os.path.join(save_dir, f'{prefix}_atmos.dat')
        atmos_fine = self.bilinear_upscale_fast(X_low, (h_high, w_high), atmos_file)
        
        # Step 7: Prepare elevation
        print("\n--- Preparing elevation ---")
        if S_high.shape[0] == 1:
            print("Broadcasting static elevation across all timesteps...")
            elev = np.repeat(S_high, t, axis=0)
        else:
            elev = S_high
        
        # Step 8: Stack all 34 covariates in memory-mapped file
        print("\n--- Stacking all covariates ---")
        print("Feature composition:")
        print("  [0]      Central precipitation (disaggregated)")
        print("  [1-4]    Adjacent cell precipitation (up, down, left, right)")
        print("  [5]      Dry drift (distance to nearest dry cell)")
        print("  [6-32]   Atmospheric fields (27 ERA5 variables)")
        print("  [33]     Elevation")
        print("  TOTAL: 34 features")
        
        covariates_file = os.path.join(save_dir, f'{prefix}_covariates.dat')
        covariates = np.memmap(covariates_file, dtype='float32', mode='w+',
                               shape=(t, h_high, w_high, 34))
        
        # Stack in temporal chunks to manage memory
        chunk_size = 500
        for i in tqdm(range(0, t, chunk_size), desc="Stacking covariates"):
            end_i = min(i + chunk_size, t)
            
            covariates[i:end_i, :, :, 0:1] = precip_disagg[i:end_i, :, :, :]      # central precip
            covariates[i:end_i, :, :, 1:5] = adjacent[i:end_i, :, :, :]           # adjacent cells
            covariates[i:end_i, :, :, 5:6] = dry_drift[i:end_i, :, :, :]          # dry drift
            covariates[i:end_i, :, :, 6:33] = atmos_fine[i:end_i, :, :, :]        # atmospheric
            covariates[i:end_i, :, :, 33:34] = elev[i:end_i, :, :, :]             # elevation
        
        covariates.flush()
        
        # Clean up intermediate memory-mapped arrays
        del precip_disagg, dry_drift, atmos_fine
        
        print(f"\nCovariates prepared: {covariates.shape}")
        print(f"Memory-mapped file size: {os.path.getsize(covariates_file) / 1e9:.2f} GB")
        
        return covariates, y_high
    
    def sample_training_data(self, covariates, target):
        """
        Sample training data following He et al. (2016) 3D space-time strategy
        Randomly samples 10% of pixels at each timestep
        
        Parameters:
        -----------
        covariates : memory-mapped array (t, 320, 320, 34)
        target : array (t, 320, 320, 1)
        
        Returns:
        --------
        X_train : array (n_samples, 34)
        y_train : array (n_samples,)
        """
        print("\n" + "="*80)
        print("SAMPLING TRAINING DATA (3D Space-Time)")
        print("="*80)
        
        t, h, w, c = covariates.shape
        total_pixels = h * w
        n_samples_per_time = int(total_pixels * self.train_sample_ratio)
        
        print(f"Sampling strategy:")
        print(f"  Total pixels per timestep: {total_pixels:,}")
        print(f"  Sample ratio: {self.train_sample_ratio*100}%")
        print(f"  Samples per timestep: {n_samples_per_time:,}")
        print(f"  Total timesteps: {t}")
        print(f"  Total training samples: {t * n_samples_per_time:,}")
        
        np.random.seed(self.random_state)
        
        X_train_list = []
        y_train_list = []
        
        print("\nSampling pixels...")
        for ti in tqdm(range(t), desc="Sampling timesteps"):
            # Randomly sample pixel indices
            sampled_indices = np.random.choice(total_pixels, n_samples_per_time, 
                                              replace=False)
            
            # Convert flat indices to 2D coordinates
            sampled_i = sampled_indices // w
            sampled_j = sampled_indices % w
            
            # Extract samples (memory-mapped array reads only needed data)
            X_train_list.append(covariates[ti, sampled_i, sampled_j, :])
            y_train_list.append(target[ti, sampled_i, sampled_j, 0])
        
        # Concatenate all samples
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        print(f"\nTraining data assembled:")
        print(f"  X_train shape: {X_train.shape}")
        print(f"  y_train shape: {y_train.shape}")
        print(f"  Memory usage: {(X_train.nbytes + y_train.nbytes) / 1e9:.2f} GB")
        print(f"\nPrecipitation statistics:")
        print(f"  Range: [{y_train.min():.3f}, {y_train.max():.3f}] mm/day")
        print(f"  Mean: {y_train.mean():.3f} mm/day")
        print(f"  Wet day frequency: {(y_train > self.wet_threshold).mean()*100:.2f}%")
        
        return X_train, y_train
    
    def fit(self, X_train, y_train):
        """
        Fit Double Random Forest (moderate + extreme precipitation)
        Following He et al. (2016) methodology
        
        Parameters:
        -----------
        X_train : array (n_samples, 34)
        y_train : array (n_samples,)
        """
        print("\n" + "="*80)
        print("FITTING DOUBLE RANDOM FOREST")
        print("="*80)
        
        # Calculate extreme precipitation threshold
        wet_vals = y_train[y_train > self.wet_threshold]
        if len(wet_vals) > 0:
            self.extreme_threshold = np.percentile(wet_vals, self.extreme_percentile)
        else:
            self.extreme_threshold = np.percentile(y_train, 99)
        
        print(f"Precipitation thresholds:")
        print(f"  Wet day threshold: {self.wet_threshold} mm/day")
        print(f"  Extreme threshold (>{self.extreme_percentile}%ile): {self.extreme_threshold:.2f} mm/day")
        
        # Separate moderate and extreme samples
        moderate_mask = y_train <= self.extreme_threshold
        extreme_mask = y_train > self.extreme_threshold
        
        print(f"\nSample distribution:")
        print(f"  Dry/moderate: {moderate_mask.sum():,} samples ({moderate_mask.mean()*100:.2f}%)")
        print(f"  Extreme:      {extreme_mask.sum():,} samples ({extreme_mask.mean()*100:.2f}%)")
        
        # Fit Random Forest for moderate precipitation
        print(f"\n{'='*60}")
        print(f"[1/2] Training RF for MODERATE precipitation")
        print(f"{'='*60}")
        print(f"Samples: {moderate_mask.sum():,}")
        print(f"Trees: {self.n_estimators}")
        print(f"Workers: {self.n_jobs}")
        
        start_time = time.time()
        
        self.rf_moderate = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_features='sqrt',  # sqrt(34) ≈ 6 features per split
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=1,
            min_samples_split=10,
            min_samples_leaf=5
        )
        
        self.rf_moderate.fit(X_train[moderate_mask], y_train[moderate_mask])
        
        elapsed = time.time() - start_time
        print(f"\n✓ Moderate RF trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Fit Random Forest for extreme precipitation
        if extreme_mask.sum() > 100:
            print(f"\n{'='*60}")
            print(f"[2/2] Training RF for EXTREME precipitation")
            print(f"{'='*60}")
            print(f"Samples: {extreme_mask.sum():,}")
            print(f"Trees: {self.n_estimators}")
            
            start_time = time.time()
            
            self.rf_extreme = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_features='sqrt',
                n_jobs=self.n_jobs,
                random_state=self.random_state,
                verbose=1,
                min_samples_split=5,
                min_samples_leaf=2
            )
            
            self.rf_extreme.fit(X_train[extreme_mask], y_train[extreme_mask])
            
            elapsed = time.time() - start_time
            print(f"\n✓ Extreme RF trained in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        else:
            print(f"\n{'='*60}")
            print(f"[2/2] Skipping extreme RF (insufficient samples: {extreme_mask.sum()})")
            print(f"{'='*60}")
            self.rf_extreme = None
        
        print("\n" + "="*80)
        print("✓ DOUBLE RF TRAINING COMPLETED!")
        print("="*80)
        
        return self
    
    def predict(self, covariates):
        """
        Predict precipitation using trained Double RF
        
        Parameters:
        -----------
        covariates : memory-mapped array (t, 320, 320, 34)
        
        Returns:
        --------
        predictions : array (t, 320, 320)
        """
        print("\n" + "="*80)
        print("PREDICTING WITH DOUBLE RF")
        print("="*80)
        
        t, h, w, c = covariates.shape
        predictions = np.zeros((t, h, w), dtype=np.float32)
        
        total_pixels = t * h * w
        print(f"Total predictions: {total_pixels:,}")
        print(f"Processing in temporal chunks...")
        
        # Predict in temporal chunks to manage memory
        chunk_size = 100
        
        for chunk_start in tqdm(range(0, t, chunk_size), desc="Predicting chunks"):
            chunk_end = min(chunk_start + chunk_size, t)
            
            # Load chunk from memory-mapped array
            X_chunk = covariates[chunk_start:chunk_end].reshape(-1, c)
            
            # Predict with moderate RF
            pred_moderate = self.rf_moderate.predict(X_chunk)
            
            # Apply extreme RF where predictions are high
            if self.rf_extreme is not None:
                extreme_mask = pred_moderate > (self.extreme_threshold * 0.8)
                n_extreme = extreme_mask.sum()
                
                if n_extreme > 0:
                    pred_extreme = self.rf_extreme.predict(X_chunk[extreme_mask])
                    pred_moderate[extreme_mask] = pred_extreme
            
            # Reshape and store
            predictions[chunk_start:chunk_end] = pred_moderate.reshape(
                chunk_end - chunk_start, h, w
            )
        
        # Ensure non-negative precipitation
        predictions = np.maximum(predictions, 0)
        
        print(f"\nPrediction statistics:")
        print(f"  Range: [{predictions.min():.3f}, {predictions.max():.3f}] mm/day")
        print(f"  Mean: {predictions.mean():.3f} mm/day")
        print(f"  Median: {np.median(predictions):.3f} mm/day")
        print(f"  Wet frequency (>{self.wet_threshold}): {(predictions > self.wet_threshold).mean()*100:.2f}%")
        
        return predictions
    
    def get_feature_importance(self):
        """
        Get and display feature importance from trained RF models
        
        Returns:
        --------
        dict with importance arrays for moderate and extreme RFs
        """
        feature_names = (
            ['precip_central'] + 
            ['precip_up', 'precip_down', 'precip_left', 'precip_right'] +
            ['dry_drift'] +
            [f'z850', 'z700', 'z500', 'z250',
             'q850', 'q700', 'q500', 'q250',
             't850', 't700', 't500', 't250',
             'u850', 'u700', 'u500', 'u250',
             'v850', 'v700', 'v500', 'v250',
             'w850', 'w700', 'w500', 'w250',
             'msl', 't2m', 'cape'] +
            ['elevation']
        )
        
        print("\n" + "="*80)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        importance_moderate = self.rf_moderate.feature_importances_
        
        print("\n[MODERATE RF] Top 15 most important features:")
        print("-" * 60)
        sorted_idx = np.argsort(importance_moderate)[::-1]
        for rank, idx in enumerate(sorted_idx[:15], 1):
            print(f"  {rank:2d}. {feature_names[idx]:25s}: {importance_moderate[idx]:.6f}")
        
        result = {'moderate': importance_moderate}
        
        if self.rf_extreme is not None:
            importance_extreme = self.rf_extreme.feature_importances_
            
            print("\n[EXTREME RF] Top 15 most important features:")
            print("-" * 60)
            sorted_idx = np.argsort(importance_extreme)[::-1]
            for rank, idx in enumerate(sorted_idx[:15], 1):
                print(f"  {rank:2d}. {feature_names[idx]:25s}: {importance_extreme[idx]:.6f}")
            
            result['extreme'] = importance_extreme
        
        return result
    
    def save_predictions(self, predictions, target, save_path):
        """Save predictions and targets"""
        print(f"\nSaving predictions to {save_path}...")
        np.savez_compressed(
            save_path, 
            predictions=predictions,
            target=target.squeeze()
        )
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"✓ Saved! File size: {size_mb:.1f} MB")


def load_data_from_files(data_path):
    """
    Load atmospheric, precipitation, and elevation data
    
    Parameters:
    -----------
    data_path : str - path to data directory
    
    Returns:
    --------
    X : array (t, 32, 32, 27) - atmospheric fields (z-score standardized)
    y : array (t, 320, 320, 1) - precipitation (mm/day)
    S : array (t or 1, 320, 320, 1) - elevation
    """
    print("\n" + "="*80)
    print("LOADING DATA FROM FILES")
    print("="*80)
    print(f"Data path: {data_path}")
    
    # Atmospheric fields (27 channels, z-score standardized)
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
    
    print(f"\nLoading {len(inputs_channels)} atmospheric channels (z-score standardized)...")
    X_list = []
    for i, (var_name, file_path) in enumerate(inputs_channels.items()):
        print(f"  [{i+1:2d}/{len(inputs_channels)}] {var_name:6s}...", end=' ')
        data = np.load(file_path)
        X_list.append(data)
        print(f"✓ shape: {data.shape}")
    
    X = np.stack(X_list, axis=-1)
    print(f"\n✓ Stacked atmospheric fields: {X.shape}")
    
    print(f"\nLoading elevation from {os.path.basename(static_file)}...")
    S = np.load(static_file)
    if S.ndim == 3:
        S = np.expand_dims(S, axis=-1)
    print(f"✓ Elevation shape: {S.shape}")
    
    print(f"\nLoading precipitation (raw mm/day) from {os.path.basename(target_file)}...")
    y = np.load(target_file)
    if y.ndim == 3:
        y = np.expand_dims(y, axis=-1)
    print(f"✓ Precipitation shape: {y.shape}")
    
    print("\n" + "-"*80)
    print("DATA SUMMARY:")
    print(f"  X (atmospheric):  {X.shape}, range=[{X.min():.4f}, {X.max():.4f}], mean={X.mean():.4f}")
    print(f"  S (elevation):    {S.shape}, range=[{S.min():.4f}, {S.max():.4f}]")
    print(f"  y (precipitation): {y.shape}, range=[{y.min():.4f}, {y.max():.4f}], mean={y.mean():.4f} mm/day")
    print("-"*80)
    
    return X, y, S


def main_pipeline(data_path, save_dir, train_size=11323, val_size=3287):
    """
    Complete Prec-DWARF pipeline for Indian precipitation downscaling
    
    Parameters:
    -----------
    data_path : str - path to input data directory
    save_dir : str - path to output directory
    train_size : int - number of training timesteps (1979-2009: 11323 days)
    val_size : int - number of validation timesteps (2010-2023: 3287 days)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("PREC-DWARF: RANDOM FOREST PRECIPITATION DOWNSCALING")
    print("Fast implementation with scipy.zoom optimization")
    print("Following He et al. (2016) Water Resources Research")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path:    {data_path}")
    print(f"  Save path:    {save_dir}")
    print(f"  Training:     {train_size} days (1979-2009)")
    print(f"  Validation:   {val_size} days (2010-2023)")
    
    # Load data
    X_low, y_high, S_high = load_data_from_files(data_path)
    
    # Split into train/val
    print("\n" + "="*80)
    print("SPLITTING DATA")
    print("="*80)
    
    X_train = X_low[:train_size]
    y_train = y_high[:train_size]
    S_train = S_high[:1] if S_high.shape[0] == 1 else S_high[:train_size]
    
    X_val = X_low[train_size:train_size+val_size]
    y_val = y_high[train_size:train_size+val_size]
    S_val = S_high[:1] if S_high.shape[0] == 1 else S_high[train_size:train_size+val_size]
    
    print(f"Training set:")
    print(f"  X: {X_train.shape}")
    print(f"  y: {y_train.shape}")
    print(f"  S: {S_train.shape}")
    print(f"\nValidation set:")
    print(f"  X: {X_val.shape}")
    print(f"  y: {y_val.shape}")
    print(f"  S: {S_val.shape}")
    
    # Initialize downscaler
    downscaler = PrecDWARF_Fast(
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
    
    # Fit Double RF
    downscaler.fit(X_train_sampled, y_train_sampled)
    
    # Get feature importance
    feature_importance = downscaler.get_feature_importance()
    
    # Prepare validation covariates
    covariates_val, target_val = downscaler.prepare_covariates_memmap(
        X_val, y_val, S_val, save_dir, prefix='val'
    )
    
    # Predict
    predictions = downscaler.predict(covariates_val)
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    pred_path = os.path.join(save_dir, 'rf_predictions.npz')
    downscaler.save_predictions(predictions, target_val, pred_path)
    
    # Save feature importance
    importance_path = os.path.join(save_dir, 'feature_importance.npz')
    np.savez_compressed(importance_path, **feature_importance)
    print(f"✓ Saved feature importance to: {importance_path}")
    
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  Predictions:        {pred_path}")
    print(f"  Feature importance: {importance_path}")
    print(f"\nMemory-mapped files (can be deleted after use):")
    print(f"  {save_dir}/train_*.dat")
    print(f"  {save_dir}/val_*.dat")
    
    return downscaler, predictions


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        print("="*80)
        print("RUNNING PREC-DWARF FROM COMMAND LINE")
        print("="*80)
        print(f"Arguments:")
        print(f"  DATA_PATH: {sys.argv[1]}")
        print(f"  SAVE_DIR:  {sys.argv[2]}")
        
        downscaler, predictions = main_pipeline(
            data_path=sys.argv[1],
            save_dir=sys.argv[2],
            train_size=11323,
            val_size=3287
        )
        
        print("\n" + "="*80)
        print("ALL DONE!")
        print("="*80)
    else:
        print("Prec-DWARF downscaler ready to use!")
        print("\nUsage:")
        print("  python rf_downscaler_fast.py <DATA_PATH> <SAVE_DIR>")
        print("\nExample:")
        print("  python rf_downscaler_fast.py /path/to/data /path/to/output")
