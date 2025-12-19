# v1.0: Simple RF Perfect Prognosis (PP) with SHAP analysis
# Train at coarse resolution (32x32), upscale to fine (320x320)

import numpy as np
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt
import warnings
import os
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


class SimplePerfectPrognosisRF:
    """
    Simple Random Forest for Perfect Prognosis precipitation downscaling
    - Train at coarse resolution (32x32)
    - Pixel-wise regression
    - SHAP feature importance
    - Bilinear upscaling to fine resolution (320x320)
    """
    
    def __init__(self, n_estimators=100, n_jobs=-1, max_depth=20, 
                 min_samples_split=10, random_state=42):
        """
        Parameters:
        -----------
        n_estimators : int
            Number of trees in Random Forest
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        max_depth : int
            Maximum depth of trees
        min_samples_split : int
            Minimum samples required to split
        random_state : int
            Random seed
        """
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.rf_model = None
        self.feature_names = None
        
        print(f"Initialized SimplePerfectPrognosisRF with {self.n_jobs} CPU cores")
    
    def prepare_data(self, X_low, y_low, save_dir, prefix='train'):
        """
        Prepare data: flatten spatial dimensions for pixel-wise training
        
        Parameters:
        -----------
        X_low : array (t, 32, 32, 27) - atmospheric predictors
        y_low : array (t, 32, 32, 1) - precipitation target
        save_dir : str - directory to save data
        prefix : str - 'train' or 'val'
        
        Returns:
        --------
        X_flat : array (t*32*32, 27)
        y_flat : array (t*32*32,)
        """
        print("\n" + "="*80)
        print(f"PREPARING DATA ({prefix.upper()})")
        print("="*80)
        
        t, h, w, c = X_low.shape
        
        print(f"Input shape:  X={X_low.shape}, y={y_low.shape}")
        
        # Flatten spatial dimensions (keep time and channels)
        X_flat = X_low.reshape(-1, c)  # (t*h*w, 27)
        y_flat = y_low.reshape(-1)     # (t*h*w,)
        
        print(f"Flattened:    X={X_flat.shape}, y={y_flat.shape}")
        print(f"Total samples: {X_flat.shape[0]:,}")
        
        # Save to disk
        print(f"\nSaving {prefix} data...")
        X_save_path = os.path.join(save_dir, f'{prefix}_X.npz')
        y_save_path = os.path.join(save_dir, f'{prefix}_y.npz')
        
        np.savez_compressed(X_save_path, X=X_flat)
        np.savez_compressed(y_save_path, y=y_flat)
        
        print(f"  ✓ Saved X to: {X_save_path} ({os.path.getsize(X_save_path)/1e6:.1f} MB)")
        print(f"  ✓ Saved y to: {y_save_path} ({os.path.getsize(y_save_path)/1e6:.1f} MB)")
        
        # Statistics
        print(f"\nData statistics ({prefix}):")
        print(f"  X range: [{X_flat.min():.4f}, {X_flat.max():.4f}]")
        print(f"  X mean:  {X_flat.mean():.4f}")
        print(f"  y range: [{y_flat.min():.4f}, {y_flat.max():.4f}]")
        print(f"  y mean:  {y_flat.mean():.4f} mm/day")
        print(f"  Dry frequency (<1mm): {(y_flat < 1.0).mean()*100:.2f}%")
        print(f"  Wet frequency (≥1mm): {(y_flat >= 1.0).mean()*100:.2f}%")
        
        return X_flat, y_flat
    
    def fit(self, X_train, y_train):
        """
        Fit Random Forest model
        
        Parameters:
        -----------
        X_train : array (n_samples, 27)
        y_train : array (n_samples,)
        """
        print("\n" + "="*80)
        print("TRAINING RANDOM FOREST")
        print("="*80)
        
        print(f"Training samples: {X_train.shape[0]:,}")
        print(f"Features: {X_train.shape[1]}")
        print(f"\nRandom Forest configuration:")
        print(f"  n_estimators:       {self.n_estimators}")
        print(f"  max_depth:          {self.max_depth}")
        print(f"  min_samples_split:  {self.min_samples_split}")
        print(f"  max_features:       'sqrt' (≈{int(np.sqrt(X_train.shape[1]))} features/split)")
        print(f"  n_jobs:             {self.n_jobs}")
        
        start_time = time.time()
        
        self.rf_model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features='sqrt',
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=2
        )
        
        print("\nFitting Random Forest...")
        self.rf_model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Training score
        train_score = self.rf_model.score(X_train, y_train)
        print(f"  Training R² score: {train_score:.4f}")
        
        return self
    
    def predict(self, X_test, original_shape):
        """
        Predict precipitation at coarse resolution
        
        Parameters:
        -----------
        X_test : array (n_samples, 27)
        original_shape : tuple (t, h, w) - to reshape output
        
        Returns:
        --------
        predictions : array (t, h, w)
        """
        print("\n" + "="*80)
        print("PREDICTION AT COARSE RESOLUTION")
        print("="*80)
        
        print(f"Predicting for {X_test.shape[0]:,} samples...")
        
        start_time = time.time()
        y_pred_flat = self.rf_model.predict(X_test)
        elapsed = time.time() - start_time
        
        print(f"✓ Prediction completed in {elapsed:.1f}s")
        
        # Reshape to original spatial dimensions
        t, h, w = original_shape
        y_pred = y_pred_flat.reshape(t, h, w)
        
        # Ensure non-negative precipitation
        y_pred = np.maximum(y_pred, 0)
        
        print(f"\nPrediction statistics:")
        print(f"  Shape: {y_pred.shape}")
        print(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}] mm/day")
        print(f"  Mean:  {y_pred.mean():.4f} mm/day")
        print(f"  Wet frequency (≥1mm): {(y_pred >= 1.0).mean()*100:.2f}%")
        
        return y_pred
    
    def upscale_to_fine(self, y_coarse, target_shape=(320, 320), save_path=None):
        """
        Bilinear upscaling from 32x32 to 320x320
        
        Parameters:
        -----------
        y_coarse : array (t, 32, 32) - coarse resolution predictions
        target_shape : tuple (h_high, w_high)
        save_path : str - path to save upscaled predictions
        
        Returns:
        --------
        y_fine : array (t, 320, 320)
        """
        print("\n" + "="*80)
        print("BILINEAR UPSCALING TO FINE RESOLUTION")
        print("="*80)
        
        t, h_low, w_low = y_coarse.shape
        h_high, w_high = target_shape
        
        print(f"Upscaling from {h_low}×{w_low} to {h_high}×{w_high}")
        print(f"Scale factor: {h_high/h_low:.1f}×")
        
        # Calculate zoom factors
        zoom_factors = (1, h_high/h_low, w_high/w_low)
        
        print(f"\nUpscaling {t} timesteps using scipy.zoom...")
        start_time = time.time()
        
        y_fine = zoom(y_coarse, zoom_factors, order=1, mode='nearest')
        
        elapsed = time.time() - start_time
        print(f"✓ Upscaling completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        print(f"\nUpscaled data:")
        print(f"  Shape: {y_fine.shape}")
        print(f"  Range: [{y_fine.min():.4f}, {y_fine.max():.4f}] mm/day")
        print(f"  Mean:  {y_fine.mean():.4f} mm/day")
        
        # Save if path provided
        if save_path is not None:
            print(f"\nSaving upscaled predictions to: {save_path}")
            np.save(save_path, y_fine)
            size_mb = os.path.getsize(save_path) / 1e6
            print(f"  ✓ Saved! File size: {size_mb:.1f} MB")
        
        return y_fine
    
    def shap_analysis(self, X_sample, feature_names, save_dir, 
                      max_samples=1000, plot_type='beeswarm'):
        """
        SHAP feature importance analysis with beeswarm plot
        
        Parameters:
        -----------
        X_sample : array (n_samples, 27) - sample data for SHAP
        feature_names : list - feature names
        save_dir : str - directory to save plots
        max_samples : int - max samples for SHAP (memory limit)
        plot_type : str - 'beeswarm', 'bar', or 'both'
        """
        print("\n" + "="*80)
        print("SHAP FEATURE IMPORTANCE ANALYSIS")
        print("="*80)
        
        self.feature_names = feature_names
        
        # Subsample if too many samples
        if X_sample.shape[0] > max_samples:
            print(f"Subsampling {max_samples} samples from {X_sample.shape[0]:,} for SHAP analysis...")
            np.random.seed(self.random_state)
            indices = np.random.choice(X_sample.shape[0], max_samples, replace=False)
            X_shap = X_sample[indices]
        else:
            X_shap = X_sample
        
        print(f"Computing SHAP values for {X_shap.shape[0]:,} samples...")
        print(f"This may take several minutes...")
        
        start_time = time.time()
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(self.rf_model)
        shap_values = explainer.shap_values(X_shap, check_additivity=False)
        
        elapsed = time.time() - start_time
        print(f"✓ SHAP values computed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Save SHAP values
        shap_save_path = os.path.join(save_dir, 'shap_values.npz')
        print(f"\nSaving SHAP values to: {shap_save_path}")
        np.savez_compressed(
            shap_save_path,
            shap_values=shap_values,
            base_value=explainer.expected_value,
            feature_names=feature_names
        )
        print(f"  ✓ Saved! ({os.path.getsize(shap_save_path)/1e6:.1f} MB)")
        
        # Generate plots
        print("\n" + "-"*80)
        print("GENERATING SHAP PLOTS")
        print("-"*80)
        
        # Beeswarm plot (summary plot)
        if plot_type in ['beeswarm', 'both']:
            print("\n[1] Generating beeswarm plot...")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(
                shap_values, 
                X_shap, 
                feature_names=feature_names,
                show=False,
                max_display=27
            )
            plt.title("SHAP Feature Importance - Beeswarm Plot", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            beeswarm_path = os.path.join(save_dir, 'shap_beeswarm.png')
            plt.savefig(beeswarm_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to: {beeswarm_path}")
            plt.close()
        
        # Bar plot (mean absolute SHAP values)
        if plot_type in ['bar', 'both']:
            print("\n[2] Generating bar plot...")
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                X_shap, 
                feature_names=feature_names,
                plot_type='bar',
                show=False,
                max_display=27
            )
            plt.title("SHAP Feature Importance - Bar Plot", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            bar_path = os.path.join(save_dir, 'shap_bar.png')
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            print(f"  ✓ Saved to: {bar_path}")
            plt.close()
        
        # Feature importance summary
        print("\n" + "-"*80)
        print("TOP 15 MOST IMPORTANT FEATURES (by mean |SHAP|):")
        print("-"*80)
        
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        sorted_idx = np.argsort(mean_abs_shap)[::-1]
        
        for rank, idx in enumerate(sorted_idx[:15], 1):
            print(f"  {rank:2d}. {feature_names[idx]:15s}: {mean_abs_shap[idx]:.6f}")
        
        return shap_values, explainer.expected_value
    
    def get_sklearn_importance(self, feature_names, save_dir):
        """
        Get sklearn built-in feature importance (Gini importance)
        
        Parameters:
        -----------
        feature_names : list
        save_dir : str
        """
        print("\n" + "="*80)
        print("SKLEARN FEATURE IMPORTANCE (Gini)")
        print("="*80)
        
        importance = self.rf_model.feature_importances_
        
        # Save
        importance_path = os.path.join(save_dir, 'sklearn_importance.npz')
        np.savez_compressed(
            importance_path,
            importance=importance,
            feature_names=feature_names
        )
        print(f"Saved to: {importance_path}")
        
        # Print top 15
        print("\nTOP 15 MOST IMPORTANT FEATURES (by Gini importance):")
        print("-"*60)
        sorted_idx = np.argsort(importance)[::-1]
        for rank, idx in enumerate(sorted_idx[:15], 1):
            print(f"  {rank:2d}. {feature_names[idx]:15s}: {importance[idx]:.6f}")
        
        # Plot
        plt.figure(figsize=(10, 8))
        sorted_importance = importance[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]
        
        plt.barh(range(len(sorted_features)), sorted_importance)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Gini Importance', fontsize=12)
        plt.title('Feature Importance (Sklearn Random Forest)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'sklearn_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {plot_path}")
        plt.close()
        
        return importance


def upscale_precipitation_coarse(y_high, coarse_shape=(32, 32)):
    """
    Upscale high-res precipitation to coarse using box averaging
    
    Parameters:
    -----------
    y_high : array (t, 320, 320, 1) - fine resolution
    coarse_shape : tuple (32, 32)
    
    Returns:
    --------
    y_coarse : array (t, 32, 32, 1)
    """
    print("\n" + "="*80)
    print("UPSCALING PRECIPITATION TO COARSE RESOLUTION")
    print("="*80)
    
    t, h_high, w_high, c = y_high.shape
    h_coarse, w_coarse = coarse_shape
    
    factor_h = h_high // h_coarse
    factor_w = w_high // w_coarse
    
    print(f"Upscaling from {h_high}×{w_high} to {h_coarse}×{w_coarse}")
    print(f"Averaging over {factor_h}×{factor_w} boxes")
    
    y_coarse = np.zeros((t, h_coarse, w_coarse, c), dtype=np.float32)
    
    for ti in tqdm(range(t), desc="Box averaging"):
        for i in range(h_coarse):
            for j in range(w_coarse):
                y_coarse[ti, i, j, 0] = np.mean(
                    y_high[ti, i*factor_h:(i+1)*factor_h, 
                           j*factor_w:(j+1)*factor_w, 0]
                )
    
    print(f"\nResults:")
    print(f"  Fine mean:   {y_high.mean():.4f} mm/day")
    print(f"  Coarse mean: {y_coarse.mean():.4f} mm/day")
    print(f"  Mass preserved: {np.abs(y_high.mean() - y_coarse.mean()) < 0.001}")
    
    return y_coarse


def load_data(data_path):
    """Load atmospheric and precipitation data"""
    
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    print(f"Data path: {data_path}")
    
    # Atmospheric fields (27 channels, RAW - no standardization for RF)
    inputs_channels = {
        'z850': f'{data_path}/IND32M_ERA5_100_Z850_DAY_1979_2023_RBIL.npy',
        'z700': f'{data_path}/IND32M_ERA5_100_Z700_DAY_1979_2023_RBIL.npy',
        'z500': f'{data_path}/IND32M_ERA5_100_Z500_DAY_1979_2023_RBIL.npy',
        'z250': f'{data_path}/IND32M_ERA5_100_Z250_DAY_1979_2023_RBIL.npy',
        'q850': f'{data_path}/IND32M_ERA5_100_Q850_DAY_1979_2023_RBIL.npy',
        'q700': f'{data_path}/IND32M_ERA5_100_Q700_DAY_1979_2023_RBIL.npy',
        'q500': f'{data_path}/IND32M_ERA5_100_Q500_DAY_1979_2023_RBIL.npy',
        'q250': f'{data_path}/IND32M_ERA5_100_Q250_DAY_1979_2023_RBIL.npy',
        't850': f'{data_path}/IND32M_ERA5_100_T850_DAY_1979_2023_RBIL.npy',
        't700': f'{data_path}/IND32M_ERA5_100_T700_DAY_1979_2023_RBIL.npy',
        't500': f'{data_path}/IND32M_ERA5_100_T500_DAY_1979_2023_RBIL.npy',
        't250': f'{data_path}/IND32M_ERA5_100_T250_DAY_1979_2023_RBIL.npy',
        'u850': f'{data_path}/IND32M_ERA5_100_U850_DAY_1979_2023_RBIL.npy',
        'u700': f'{data_path}/IND32M_ERA5_100_U700_DAY_1979_2023_RBIL.npy',
        'u500': f'{data_path}/IND32M_ERA5_100_U500_DAY_1979_2023_RBIL.npy',
        'u250': f'{data_path}/IND32M_ERA5_100_U250_DAY_1979_2023_RBIL.npy',
        'v850': f'{data_path}/IND32M_ERA5_100_V850_DAY_1979_2023_RBIL.npy',
        'v700': f'{data_path}/IND32M_ERA5_100_V700_DAY_1979_2023_RBIL.npy',
        'v500': f'{data_path}/IND32M_ERA5_100_V500_DAY_1979_2023_RBIL.npy',
        'v250': f'{data_path}/IND32M_ERA5_100_V250_DAY_1979_2023_RBIL.npy',
        'w850': f'{data_path}/IND32M_ERA5_100_W850_DAY_1979_2023_RBIL.npy',
        'w700': f'{data_path}/IND32M_ERA5_100_W700_DAY_1979_2023_RBIL.npy',
        'w500': f'{data_path}/IND32M_ERA5_100_W500_DAY_1979_2023_RBIL.npy',
        'w250': f'{data_path}/IND32M_ERA5_100_W250_DAY_1979_2023_RBIL.npy',
        'msl': f'{data_path}/IND32M_ERA5_100_MSL_DAY_1979_2023_RBIL.npy',
        't2m': f'{data_path}/IND32M_ERA5_100_T2M_DAY_1979_2023_RBIL.npy',
        'cape': f'{data_path}/IND32M_ERA5_100_CAPE_DAY_1979_2023_RBIL.npy',
    }
    
    target_file = f'{data_path}/IND32M_MSWX_010_PREC_DAY_1979_2023.npy'
    
    print(f"\nLoading {len(inputs_channels)} atmospheric channels (RAW, no standardization)...")
    X_list = []
    feature_names = []
    for i, (var_name, file_path) in enumerate(inputs_channels.items()):
        print(f"  [{i+1:2d}/{len(inputs_channels)}] {var_name:6s}...", end=' ')
        X_list.append(np.load(file_path))
        feature_names.append(var_name)
        print(f"✓")
    
    X_low = np.stack(X_list, axis=-1)
    
    print(f"\nLoading precipitation (fine resolution)...")
    y_high = np.load(target_file)
    if y_high.ndim == 3:
        y_high = np.expand_dims(y_high, axis=-1)
    
    print(f"\n✓ Data loaded:")
    print(f"  X_low (atmospheric): {X_low.shape}")
    print(f"  y_high (precip):     {y_high.shape}")
    
    return X_low, y_high, feature_names


def main_pipeline(data_path, save_dir, train_size=11323):
    """
    Complete Simple Perfect Prognosis RF pipeline
    
    Parameters:
    -----------
    data_path : str
    save_dir : str
    train_size : int - training timesteps (rest is validation)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("SIMPLE PERFECT PROGNOSIS RANDOM FOREST")
    print("Train at coarse (32×32), upscale to fine (320×320)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path:    {data_path}")
    print(f"  Save path:    {save_dir}")
    print(f"  Training:     {train_size} days")
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    X_low, y_high, feature_names = load_data(data_path)
    
    total_size = X_low.shape[0]
    val_size = total_size - train_size
    
    print(f"\nData split:")
    print(f"  Total:      {total_size} days")
    print(f"  Training:   {train_size} days")
    print(f"  Validation: {val_size} days")
    
    # =========================================================================
    # STEP 2: UPSCALE TARGET TO COARSE RESOLUTION (32×32)
    # =========================================================================
    y_low = upscale_precipitation_coarse(y_high, coarse_shape=(32, 32))
    
    # Save coarse precipitation
    y_low_save = os.path.join(save_dir, 'precipitation_coarse_32x32.npy')
    np.save(y_low_save, y_low)
    print(f"\n✓ Saved coarse precipitation to: {y_low_save}")
    
    # =========================================================================
    # STEP 3: SPLIT TRAIN/VAL
    # =========================================================================
    X_train = X_low[:train_size]
    y_train = y_low[:train_size]
    
    X_val = X_low[train_size:]
    y_val = y_low[train_size:]
    
    # =========================================================================
    # STEP 4: INITIALIZE MODEL
    # =========================================================================
    rf_model = SimplePerfectPrognosisRF(
        n_estimators=100,
        n_jobs=-1,
        max_depth=20,
        min_samples_split=10,
        random_state=42
    )
    
    # =========================================================================
    # STEP 5: PREPARE DATA (FLATTEN SPATIAL DIMENSIONS)
    # =========================================================================
    X_train_flat, y_train_flat = rf_model.prepare_data(
        X_train, y_train, save_dir, prefix='train'
    )
    
    X_val_flat, y_val_flat = rf_model.prepare_data(
        X_val, y_val, save_dir, prefix='val'
    )
    
    # =========================================================================
    # STEP 6: TRAIN RANDOM FOREST
    # =========================================================================
    rf_model.fit(X_train_flat, y_train_flat)
    
    # =========================================================================
    # STEP 7: SKLEARN FEATURE IMPORTANCE
    # =========================================================================
    sklearn_importance = rf_model.get_sklearn_importance(feature_names, save_dir)
    
    # =========================================================================
    # STEP 8: SHAP ANALYSIS
    # =========================================================================
    print("\n" + "="*80)
    print("PERFORMING SHAP ANALYSIS")
    print("="*80)
    
    # Use validation data for SHAP (representative sample)
    shap_values, base_value = rf_model.shap_analysis(
        X_val_flat,
        feature_names,
        save_dir,
        max_samples=1000,  # Limit for memory
        plot_type='both'   # Both beeswarm and bar plots
    )
    
    # =========================================================================
    # STEP 9: PREDICT AT COARSE RESOLUTION
    # =========================================================================
    # Training predictions
    y_train_pred = rf_model.predict(X_train_flat, X_train.shape[:3])
    
    # Validation predictions
    y_val_pred = rf_model.predict(X_val_flat, X_val.shape[:3])
    
    # Save coarse predictions
    train_pred_coarse = os.path.join(save_dir, 'train_predictions_coarse_32x32.npy')
    val_pred_coarse = os.path.join(save_dir, 'val_predictions_coarse_32x32.npy')
    
    np.save(train_pred_coarse, y_train_pred)
    np.save(val_pred_coarse, y_val_pred)
    
    print(f"\n✓ Saved coarse predictions:")
    print(f"  Training:   {train_pred_coarse}")
    print(f"  Validation: {val_pred_coarse}")
    
    # =========================================================================
    # STEP 10: UPSCALE VALIDATION PREDICTIONS TO FINE RESOLUTION (320×320)
    # =========================================================================
    val_pred_fine_path = os.path.join(save_dir, 'val_predictions_upscaled_320x320.npy')
    
    y_val_pred_fine = rf_model.upscale_to_fine(
        y_val_pred, 
        target_shape=(320, 320),
        save_path=val_pred_fine_path
    )
    
    # =========================================================================
    # STEP 11: SAVE TARGETS FOR COMPARISON
    # =========================================================================
    # Save validation target at fine resolution
    y_val_fine = y_high[train_size:]
    val_target_fine = os.path.join(save_dir, 'val_target_fine_320x320.npy')
    np.save(val_target_fine, y_val_fine.squeeze())
    print(f"\n✓ Saved fine-resolution target: {val_target_fine}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    print(f"\nOutput files in {save_dir}:")
    print("\n[Data Files]")
    print(f"  precipitation_coarse_32x32.npy         - Coarse target")
    print(f"  train_X.npz, train_y.npz               - Training data (flattened)")
    print(f"  val_X.npz, val_y.npz                   - Validation data (flattened)")
    
    print("\n[Predictions]")
    print(f"  train_predictions_coarse_32x32.npy     - Training predictions (32×32)")
    print(f"  val_predictions_coarse_32x32.npy       - Validation predictions (32×32)")
    print(f"  val_predictions_upscaled_320x320.npy   - Validation upscaled (320×320) ← FINAL")
    print(f"  val_target_fine_320x320.npy            - Validation target (320×320)")
    
    print("\n[Feature Importance]")
    print(f"  sklearn_importance.npz                 - Gini importance")
    print(f"  sklearn_importance.png                 - Gini plot")
    print(f"  shap_values.npz                        - SHAP values")
    print(f"  shap_beeswarm.png                      - SHAP beeswarm plot")
    print(f"  shap_bar.png                           - SHAP bar plot")
    
    print("\n" + "="*80)
    
    return rf_model, y_val_pred_fine


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        train_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 11323
        
        rf_model, predictions = main_pipeline(
            data_path=sys.argv[1],
            save_dir=sys.argv[2],
            train_size=train_size
        )
    else:
        print("Usage: python rf_simple_pp.py <DATA_PATH> <SAVE_DIR> [TRAIN_SIZE]")
        print("Example: python rf_simple_pp.py /data /output 11323")
