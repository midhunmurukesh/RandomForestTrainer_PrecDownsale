# v2.0: RF Perfect Prognosis with Hyperparameter Tuning
# Train at coarse (32x32), validate against MSWX, find best model

import numpy as np
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
import os
from tqdm import tqdm
import time
import pickle

warnings.filterwarnings('ignore')


class RFWithTuning:
    """
    Random Forest with Hyperparameter Tuning for Precipitation Downscaling
    
    Methodology:
    1. Train at coarse resolution (32×32) using atmospheric predictors
    2. Validate against coarse-averaged MSWX precipitation
    3. Tune hyperparameters to find best model
    4. Upscale best model predictions to fine resolution (320×320)
    """
    
    def __init__(self, n_jobs=-1, random_state=42):
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.tuning_results = None
        
        print(f"Initialized RF with {self.n_jobs} CPU cores")
    
    def prepare_data(self, X_low, y_low, save_dir, prefix='train'):
        """Flatten spatial dimensions for pixel-wise regression"""
        print(f"\n{'='*80}")
        print(f"PREPARING DATA ({prefix.upper()})")
        print('='*80)
        
        t, h, w, c = X_low.shape
        
        # Flatten: (t, h, w, c) → (t*h*w, c)
        X_flat = X_low.reshape(-1, c)
        y_flat = y_low.reshape(-1)
        
        print(f"Original shape: X={X_low.shape}, y={y_low.shape}")
        print(f"Flattened:      X={X_flat.shape}, y={y_flat.shape}")
        print(f"Total samples:  {X_flat.shape[0]:,}")
        
        # Save
        X_path = os.path.join(save_dir, f'{prefix}_X.npz')
        y_path = os.path.join(save_dir, f'{prefix}_y.npz')
        
        np.savez_compressed(X_path, X=X_flat)
        np.savez_compressed(y_path, y=y_flat)
        
        print(f"\n✓ Saved:")
        print(f"  X: {X_path} ({os.path.getsize(X_path)/1e6:.1f} MB)")
        print(f"  y: {y_path} ({os.path.getsize(y_path)/1e6:.1f} MB)")
        
        # Statistics
        print(f"\nData statistics:")
        print(f"  X range: [{X_flat.min():.4f}, {X_flat.max():.4f}]")
        print(f"  y range: [{y_flat.min():.4f}, {y_flat.max():.4f}] mm/day")
        print(f"  y mean:  {y_flat.mean():.4f} mm/day")
        print(f"  Wet %:   {(y_flat >= 1.0).mean()*100:.2f}%")
        
        return X_flat, y_flat
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, 
                             n_iter=20, cv=3):
        """
        Hyperparameter tuning using RandomizedSearchCV
        
        Parameters:
        -----------
        X_train : array - training features
        y_train : array - training targets
        X_val : array - validation features (for final evaluation)
        y_val : array - validation targets (for final evaluation)
        n_iter : int - number of parameter settings sampled
        cv : int - cross-validation folds
        """
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING")
        print('='*80)
        
        # Define parameter grid
        param_distributions = {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [10, 20, 50, 100],
            'min_samples_leaf': [5, 10, 20],
            'max_features': ['sqrt', 'log2', 0.3, 0.5],
            'bootstrap': [True],
            'max_samples': [0.7, 0.8, 0.9, 1.0]
        }
        
        print(f"\nParameter search space:")
        for param, values in param_distributions.items():
            print(f"  {param:20s}: {values}")
        
        print(f"\nSearch strategy:")
        print(f"  Method:           RandomizedSearchCV")
        print(f"  Iterations:       {n_iter}")
        print(f"  Cross-validation: {cv}-fold")
        print(f"  Scoring:          R² (coefficient of determination)")
        print(f"  Total fits:       {n_iter * cv}")
        
        # Subsample training data for faster tuning (optional)
        max_tuning_samples = 50000
        if X_train.shape[0] > max_tuning_samples:
            print(f"\n⚠ Subsampling to {max_tuning_samples:,} samples for faster tuning...")
            np.random.seed(self.random_state)
            indices = np.random.choice(X_train.shape[0], max_tuning_samples, replace=False)
            X_tune = X_train[indices]
            y_tune = y_train[indices]
        else:
            X_tune = X_train
            y_tune = y_train
        
        print(f"\nTuning on {X_tune.shape[0]:,} samples...")
        
        # Base estimator
        rf_base = RandomForestRegressor(
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=0
        )
        
        # Randomized search
        search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=1,  # Parallel at tree level, not CV level
            verbose=2,
            random_state=self.random_state,
            return_train_score=True
        )
        
        print("\nStarting hyperparameter search...")
        start_time = time.time()
        
        search.fit(X_tune, y_tune)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Tuning completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Store results
        self.best_params = search.best_params_
        self.tuning_results = search.cv_results_
        
        # Print best parameters
        print(f"\n{'='*80}")
        print("BEST PARAMETERS FOUND")
        print('='*80)
        for param, value in self.best_params.items():
            print(f"  {param:20s}: {value}")
        
        print(f"\nCross-validation scores:")
        print(f"  Best CV R²:     {search.best_score_:.4f}")
        print(f"  CV R² std:      {self.tuning_results['std_test_score'][search.best_index_]:.4f}")
        
        # Evaluate best model on validation set
        print(f"\n{'='*80}")
        print("VALIDATION PERFORMANCE")
        print('='*80)
        
        y_val_pred = search.best_estimator_.predict(X_val)
        
        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)
        
        print(f"  Validation R²:    {val_r2:.4f}")
        print(f"  Validation RMSE:  {val_rmse:.4f} mm/day")
        print(f"  Validation MAE:   {val_mae:.4f} mm/day")
        
        # Store best estimator
        self.best_model = search.best_estimator_
        
        return self.best_params, search.best_score_
    
    def train_best_model(self, X_train, y_train, best_params):
        """
        Train final model with best parameters on full training data
        
        Parameters:
        -----------
        X_train : array
        y_train : array
        best_params : dict - best hyperparameters from tuning
        """
        print(f"\n{'='*80}")
        print("TRAINING FINAL MODEL WITH BEST PARAMETERS")
        print('='*80)
        
        print(f"\nTraining on full dataset: {X_train.shape[0]:,} samples")
        print(f"\nBest parameters:")
        for param, value in best_params.items():
            print(f"  {param:20s}: {value}")
        
        start_time = time.time()
        
        self.best_model = RandomForestRegressor(
            **best_params,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=2
        )
        
        self.best_model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Training score
        train_score = self.best_model.score(X_train, y_train)
        print(f"  Training R²: {train_score:.4f}")
        
        return self.best_model
    
    def predict(self, X_test, original_shape):
        """Predict and reshape to original spatial dimensions"""
        print(f"\n{'='*80}")
        print("PREDICTION")
        print('='*80)
        
        print(f"Predicting for {X_test.shape[0]:,} samples...")
        
        start_time = time.time()
        y_pred_flat = self.best_model.predict(X_test)
        elapsed = time.time() - start_time
        
        print(f"✓ Completed in {elapsed:.1f}s")
        
        # Reshape
        t, h, w = original_shape
        y_pred = y_pred_flat.reshape(t, h, w)
        y_pred = np.maximum(y_pred, 0)  # Non-negative
        
        print(f"\nPrediction statistics:")
        print(f"  Shape: {y_pred.shape}")
        print(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}] mm/day")
        print(f"  Mean:  {y_pred.mean():.4f} mm/day")
        
        return y_pred
    
    def evaluate(self, y_true, y_pred, prefix='Validation'):
        """Calculate evaluation metrics"""
        print(f"\n{'='*80}")
        print(f"{prefix.upper()} METRICS")
        print('='*80)
        
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        r2 = r2_score(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)
        
        # Precipitation-specific metrics
        wet_threshold = 1.0
        wet_mask_true = y_true_flat >= wet_threshold
        wet_mask_pred = y_pred_flat >= wet_threshold
        
        # Wet day detection
        true_positives = np.sum(wet_mask_true & wet_mask_pred)
        false_positives = np.sum(~wet_mask_true & wet_mask_pred)
        false_negatives = np.sum(wet_mask_true & ~wet_mask_pred)
        
        pod = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        far = false_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        print(f"\nOverall metrics:")
        print(f"  R² (coefficient of determination): {r2:.4f}")
        print(f"  RMSE (mm/day):                     {rmse:.4f}")
        print(f"  MAE (mm/day):                      {mae:.4f}")
        
        print(f"\nWet day detection (threshold = {wet_threshold} mm/day):")
        print(f"  POD (Probability of Detection):    {pod:.4f}")
        print(f"  FAR (False Alarm Ratio):           {far:.4f}")
        print(f"  True wet days:                     {wet_mask_true.sum():,}")
        print(f"  Predicted wet days:                {wet_mask_pred.sum():,}")
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae, 'pod': pod, 'far': far}
    
    def plot_feature_importance(self, feature_names, save_dir):
        """Plot Gini importance for best model"""
        print(f"\n{'='*80}")
        print("FEATURE IMPORTANCE (GINI)")
        print('='*80)
        
        importance = self.best_model.feature_importances_
        
        # Sort by importance
        sorted_idx = np.argsort(importance)[::-1]
        sorted_importance = importance[sorted_idx]
        sorted_features = [feature_names[i] for i in sorted_idx]
        
        # Print top 15
        print("\nTop 15 most important features:")
        print("-" * 60)
        for rank, (feat, imp) in enumerate(zip(sorted_features[:15], sorted_importance[:15]), 1):
            print(f"  {rank:2d}. {feat:15s}: {imp:.6f}")
        
        # Save to file
        importance_path = os.path.join(save_dir, 'gini_importance.npz')
        np.savez_compressed(
            importance_path,
            importance=importance,
            feature_names=np.array(feature_names)
        )
        print(f"\n✓ Saved importance to: {importance_path}")
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all features
        y_pos = np.arange(len(sorted_features))
        ax.barh(y_pos, sorted_importance, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_features, fontsize=10)
        ax.set_xlabel('Gini Importance', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance - Random Forest (Best Model)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Invert y-axis so most important is on top
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        plot_path = os.path.join(save_dir, 'gini_importance.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to: {plot_path}")
        plt.close()
        
        return importance
    
    def upscale_to_fine(self, y_coarse, target_shape=(320, 320), save_path=None):
        """Bilinear upscaling from 32×32 to 320×320"""
        print(f"\n{'='*80}")
        print("UPSCALING TO FINE RESOLUTION")
        print('='*80)
        
        t, h_low, w_low = y_coarse.shape
        h_high, w_high = target_shape
        
        print(f"Upscaling from {h_low}×{w_low} to {h_high}×{w_high}")
        
        zoom_factors = (1, h_high/h_low, w_high/w_low)
        
        start_time = time.time()
        y_fine = zoom(y_coarse, zoom_factors, order=1, mode='nearest')
        elapsed = time.time() - start_time
        
        print(f"✓ Completed in {elapsed:.1f}s")
        print(f"  Shape: {y_fine.shape}")
        print(f"  Mean:  {y_fine.mean():.4f} mm/day")
        
        if save_path:
            np.save(save_path, y_fine)
            print(f"✓ Saved to: {save_path}")
        
        return y_fine
    
    def save_model(self, save_path):
        """Save trained model"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Model saved to: {save_path}")


def upscale_precipitation_coarse(y_high, coarse_shape=(32, 32)):
    """Box averaging to coarse resolution"""
    print(f"\n{'='*80}")
    print("COARSENING PRECIPITATION (BOX AVERAGING)")
    print('='*80)
    
    t, h_high, w_high, c = y_high.shape
    h_coarse, w_coarse = coarse_shape
    
    factor_h = h_high // h_coarse
    factor_w = w_high // w_coarse
    
    print(f"From {h_high}×{w_high} to {h_coarse}×{w_coarse}")
    print(f"Box size: {factor_h}×{factor_w}")
    
    y_coarse = np.zeros((t, h_coarse, w_coarse, c), dtype=np.float32)
    
    for ti in tqdm(range(t), desc="Coarsening"):
        for i in range(h_coarse):
            for j in range(w_coarse):
                y_coarse[ti, i, j, 0] = np.mean(
                    y_high[ti, i*factor_h:(i+1)*factor_h, 
                           j*factor_w:(j+1)*factor_w, 0]
                )
    
    print(f"\n✓ Mass preserved:")
    print(f"  Fine mean:   {y_high.mean():.4f} mm/day")
    print(f"  Coarse mean: {y_coarse.mean():.4f} mm/day")
    
    return y_coarse


def load_data(data_path):
    """Load atmospheric and precipitation data"""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print('='*80)
    print(f"Data path: {data_path}")
    
    # 27 atmospheric channels (RAW - no standardization for RF)
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
    
    print(f"\nLoading {len(inputs_channels)} atmospheric channels...")
    X_list = []
    feature_names = []
    for i, (var_name, file_path) in enumerate(inputs_channels.items()):
        print(f"  [{i+1:2d}/{len(inputs_channels)}] {var_name:6s}...", end=' ')
        X_list.append(np.load(file_path))
        feature_names.append(var_name)
        print(f"✓")
    
    X_low = np.stack(X_list, axis=-1)
    
    print(f"\nLoading precipitation...")
    y_high = np.load(target_file)
    if y_high.ndim == 3:
        y_high = np.expand_dims(y_high, axis=-1)
    
    print(f"\n✓ Data loaded:")
    print(f"  X (atmospheric): {X_low.shape}")
    print(f"  y (precip):      {y_high.shape}")
    
    return X_low, y_high, feature_names


def main_pipeline(data_path, save_dir, train_size=11323, tune_model=True, n_iter=20):
    """
    Complete RF pipeline with hyperparameter tuning
    
    Parameters:
    -----------
    data_path : str
    save_dir : str
    train_size : int - training timesteps
    tune_model : bool - whether to do hyperparameter tuning
    n_iter : int - number of tuning iterations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("RANDOM FOREST WITH HYPERPARAMETER TUNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path:       {data_path}")
    print(f"  Save path:       {save_dir}")
    print(f"  Training size:   {train_size} days")
    print(f"  Tuning enabled:  {tune_model}")
    if tune_model:
        print(f"  Tuning iters:    {n_iter}")
    
    # Load data
    X_low, y_high, feature_names = load_data(data_path)
    
    total_size = X_low.shape[0]
    val_size = total_size - train_size
    
    print(f"\nData split:")
    print(f"  Total:      {total_size} days")
    print(f"  Training:   {train_size} days")
    print(f"  Validation: {val_size} days")
    
    # Coarsen precipitation to 32×32
    y_low = upscale_precipitation_coarse(y_high, coarse_shape=(32, 32))
    
    # Save coarsened target
    y_low_path = os.path.join(save_dir, 'precipitation_coarse_32x32.npy')
    np.save(y_low_path, y_low)
    print(f"\n✓ Saved: {y_low_path}")
    
    # Split data
    X_train = X_low[:train_size]
    y_train = y_low[:train_size]
    
    X_val = X_low[train_size:]
    y_val = y_low[train_size:]
    
    # Initialize model
    rf = RFWithTuning(n_jobs=-1, random_state=42)
    
    # Prepare data
    X_train_flat, y_train_flat = rf.prepare_data(X_train, y_train, save_dir, 'train')
    X_val_flat, y_val_flat = rf.prepare_data(X_val, y_val, save_dir, 'val')
    
    # Hyperparameter tuning or use default parameters
    if tune_model:
        best_params, best_cv_score = rf.hyperparameter_tuning(
            X_train_flat, y_train_flat, 
            X_val_flat, y_val_flat,
            n_iter=n_iter, cv=3
        )
        
        # Save tuning results
        tuning_path = os.path.join(save_dir, 'tuning_results.npz')
        np.savez_compressed(tuning_path, 
                           best_params=best_params,
                           best_cv_score=best_cv_score)
        print(f"\n✓ Saved tuning results: {tuning_path}")
        
        # Train final model on full training data
        rf.train_best_model(X_train_flat, y_train_flat, best_params)
    else:
        # Use default parameters
        print("\n⚠ Skipping tuning, using default parameters...")
        default_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'bootstrap': True
        }
        rf.train_best_model(X_train_flat, y_train_flat, default_params)
    
    # Save model
    model_path = os.path.join(save_dir, 'best_rf_model.pkl')
    rf.save_model(model_path)
    
    # Predictions at coarse resolution
    y_train_pred = rf.predict(X_train_flat, X_train.shape[:3])
    y_val_pred = rf.predict(X_val_flat, X_val.shape[:3])
    
    # Save coarse predictions
    np.save(os.path.join(save_dir, 'train_predictions_32x32.npy'), y_train_pred)
    np.save(os.path.join(save_dir, 'val_predictions_32x32.npy'), y_val_pred)
    
    # Evaluate at coarse resolution
    train_metrics = rf.evaluate(y_train.squeeze(), y_train_pred, 'Training')
    val_metrics = rf.evaluate(y_val.squeeze(), y_val_pred, 'Validation')
    
    # Save metrics
    metrics_path = os.path.join(save_dir, 'metrics.npz')
    np.savez_compressed(metrics_path, 
                       train=train_metrics,
                       val=val_metrics)
    print(f"\n✓ Saved metrics: {metrics_path}")
    
    # Feature importance (Gini)
    importance = rf.plot_feature_importance(feature_names, save_dir)
    
    # Upscale validation predictions to 320×320
    val_pred_fine_path = os.path.join(save_dir, 'val_predictions_upscaled_320x320.npy')
    y_val_pred_fine = rf.upscale_to_fine(y_val_pred, (320, 320), val_pred_fine_path)
    
    # Save fine-resolution target
    y_val_fine = y_high[train_size:]
    val_target_path = os.path.join(save_dir, 'val_target_fine_320x320.npy')
    np.save(val_target_path, y_val_fine.squeeze())
    print(f"✓ Saved target: {val_target_path}")
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETED!")
    print('='*80)
    print(f"\nKey outputs in {save_dir}:")
    print(f"  best_rf_model.pkl                      - Trained model")
    print(f"  gini_importance.npz & .png             - Feature importance")
    print(f"  val_predictions_upscaled_320x320.npy   - Final predictions")
    print(f"  metrics.npz                            - Performance metrics")
    if tune_model:
        print(f"  tuning_results.npz                     - Hyperparameter search")
    print('='*80)
    
    return rf, y_val_pred_fine, val_metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        train_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 11323
        tune = sys.argv[4].lower() == 'true' if len(sys.argv) >= 5 else True
        
        rf, predictions, metrics = main_pipeline(
            data_path=sys.argv[1],
            save_dir=sys.argv[2],
            train_size=train_size,
            tune_model=tune,
            n_iter=20
        )
    else:
        print("Usage: python rf_tuned.py <DATA_PATH> <SAVE_DIR> [TRAIN_SIZE] [TUNE]")
        print("Example: python rf_tuned.py /data /output 11323 true")
