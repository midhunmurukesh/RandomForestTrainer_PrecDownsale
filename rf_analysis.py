# v2.4: RF with Mean-Pooled Coarsening & NetCDF Export
# Fixed coarsening to use MEAN (not SUM) + NetCDF output generation

import numpy as np
from scipy.ndimage import zoom
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import os
from tqdm import tqdm
import time
import pickle
import json
import gc
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("RFPipeline")


class RFWithTuning:
    """
    Random Forest with Extended Hyperparameter Tuning for Precipitation Downscaling
    Memory-optimized with batched prediction
    """

    def __init__(self, n_jobs=-1, random_state=42):
        self.n_jobs = n_jobs if n_jobs > 0 else os.cpu_count()
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.tuning_results = None
        self.cv_results_df = None

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
        print(f"  y range: [{y_flat.min():.4f}, {y_flat.max():.4f}] mm/day (mean per cell)")
        print(f"  y mean:  {y_flat.mean():.4f} mm/day (mean per cell)")

        return X_flat, y_flat

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val,
                              n_iter=50, cv=3, save_dir=None):
        """
        Extended hyperparameter tuning with REGULARIZED parameters
        """
        print(f"\n{'='*80}")
        print("REGULARIZED HYPERPARAMETER TUNING")
        print('='*80)

        # REGULARIZED parameter grid (prevents overfitting and huge models)
        param_distributions = {
            'n_estimators': [200, 300, 400, 500],
            'max_depth': [20, 25, 30, 35],
            'min_samples_split': [50, 100, 200, 500],
            'min_samples_leaf': [20, 50, 100, 200],
            'max_features': ['sqrt', 0.2, 0.25, 0.3],
            'bootstrap': [True],
            'max_samples': [0.6, 0.7, 0.8]
        }

        print(f"\nRegularized parameter search space (anti-overfitting):")
        for param, values in param_distributions.items():
            print(f"  {param:20s}: {values}")

        total_combinations = 1
        for values in param_distributions.values():
            total_combinations *= len(values)

        print(f"\nSearch strategy:")
        print(f"  Total combinations:   {total_combinations:,}")
        print(f"  Sampling:             {n_iter} iterations")
        print(f"  Coverage:             {n_iter/total_combinations*100:.2f}%")
        print(f"  Cross-validation:     {cv}-fold")
        print(f"  Scoring:              R²")
        print(f"  Total model fits:     {n_iter * cv:,}")

        # Subsample training data for faster tuning
        max_tuning_samples = 100000
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
            verbose=0,
            warm_start=False
        )

        # Randomized search
        search = RandomizedSearchCV(
            rf_base,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='r2',
            n_jobs=1,
            verbose=2,
            random_state=self.random_state,
            return_train_score=True,
            refit=True
        )

        print("\nStarting hyperparameter search...")
        print(f"Estimated time: {n_iter * cv * 2 / 60:.1f} - {n_iter * cv * 5 / 60:.1f} minutes")
        start_time = time.time()

        try:
            search.fit(X_tune, y_tune)
        except Exception as e:
            print(f"\nError during hyperparameter tuning: {e}")
            raise

        elapsed = time.time() - start_time
        print(f"\n✓ Tuning completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Store results
        self.best_params = search.best_params_
        self.tuning_results = search.cv_results_

        # Create detailed results DataFrame
        self.cv_results_df = pd.DataFrame(self.tuning_results)

        # Print best parameters
        print(f"\n{'='*80}")
        print("BEST PARAMETERS FOUND")
        print('='*80)
        for param, value in self.best_params.items():
            print(f"  {param:20s}: {value}")

        print(f"\nCross-validation performance:")
        print(f"  Best CV R²:          {search.best_score_:.6f}")
        print(f"  CV R² std:           {self.tuning_results['std_test_score'][search.best_index_]:.6f}")
        print(f"  Best CV rank:        {search.best_index_ + 1}/{n_iter}")

        # Get top 10 configurations
        print(f"\n{'='*80}")
        print("TOP 10 CONFIGURATIONS")
        print('='*80)

        top_10_idx = np.argsort(self.tuning_results['mean_test_score'])[-10:][::-1]

        print(f"\n{'Rank':<6} {'R²':<10} {'±std':<10} {'Trees':<8} {'Depth':<8} {'Features':<12}")
        print("-" * 80)
        for rank, idx in enumerate(top_10_idx, 1):
            r2_mean = self.tuning_results['mean_test_score'][idx]
            r2_std = self.tuning_results['std_test_score'][idx]
            params = self.tuning_results['params'][idx]

            print(f"{rank:<6} {r2_mean:<10.6f} {r2_std:<10.6f} "
                  f"{params['n_estimators']:<8} {str(params['max_depth']):<8} "
                  f"{str(params['max_features']):<12}")

        # Evaluate best model on validation set
        print(f"\n{'='*80}")
        print("VALIDATION PERFORMANCE")
        print('='*80)

        y_val_pred = search.best_estimator_.predict(X_val)

        val_r2 = r2_score(y_val, y_val_pred)
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_mae = mean_absolute_error(y_val, y_val_pred)

        print(f"  Validation R²:       {val_r2:.6f}")
        print(f"  Validation RMSE:     {val_rmse:.4f} mm/day")
        print(f"  Validation MAE:      {val_mae:.4f} mm/day")

        # Check for overfitting
        train_r2 = self.tuning_results['mean_train_score'][search.best_index_]
        overfit_gap = train_r2 - search.best_score_

        print(f"\n  Train R² (CV):       {train_r2:.6f}")
        print(f"  Test R² (CV):        {search.best_score_:.6f}")
        print(f"  Overfit gap:         {overfit_gap:.6f}")
        if overfit_gap > 0.05:
            print(f" [WARNING]: Overfitting detected (gap > 0.05)")
            print(f"  → Model may not generalize well to new data")
        else:
            print(f"  ✓ No significant overfitting")

        # Store best estimator
        self.best_model = search.best_estimator_

        # Save tuning results if directory provided
        if save_dir:
            self.save_tuning_results(save_dir, elapsed)

        return self.best_params, search.best_score_

    def save_tuning_results(self, save_dir, elapsed_time):
        """Save comprehensive tuning results"""
        print(f"\n{'='*80}")
        print("SAVING TUNING RESULTS")
        print('='*80)

        # 1. Save best parameters (JSON)
        best_params_path = os.path.join(save_dir, 'best_params.json')
        with open(best_params_path, 'w') as f:
            params_serializable = {}
            for k, v in self.best_params.items():
                if isinstance(v, (np.integer, np.floating)):
                    params_serializable[k] = v.item()
                else:
                    params_serializable[k] = v
            json.dump(params_serializable, f, indent=2)
        print(f"✓ Best parameters (JSON):     {best_params_path}")

        # 2. Save all CV results (CSV)
        cv_results_path = os.path.join(save_dir, 'cv_results_all.csv')
        self.cv_results_df.to_csv(cv_results_path, index=False)
        print(f"✓ All CV results (CSV):       {cv_results_path}")

        # 3. Save top 50 configurations (CSV)
        top_50_path = os.path.join(save_dir, 'cv_results_top50.csv')
        top_50_idx = np.argsort(self.cv_results_df['mean_test_score'])[-50:][::-1]
        top_50_df = self.cv_results_df.iloc[top_50_idx]
        top_50_df.to_csv(top_50_path, index=False)
        print(f"✓ Top 50 configs (CSV):       {top_50_path}")

        # 4. Save summary statistics (TXT)
        summary_path = os.path.join(save_dir, 'tuning_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING SUMMARY\n")
            f.write("="*80 + "\n\n")

            f.write(f"Tuning time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)\n")
            f.write(f"Total iterations: {len(self.cv_results_df)}\n\n")

            f.write("Best parameters:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param:20s}: {value}\n")

            f.write(f"\nBest CV R²: {self.cv_results_df['mean_test_score'].max():.6f}\n")

        print(f"✓ Tuning summary (TXT):       {summary_path}")

        # 5. Save raw CV results (NPZ)
        cv_results_npz_path = os.path.join(save_dir, 'cv_results_raw.npz')
        np.savez_compressed(cv_results_npz_path, **self.tuning_results)
        print(f"✓ Raw CV results (NPZ):       {cv_results_npz_path}")

        # 6. Create visualization plots
        try:
            self.plot_tuning_results(save_dir)
        except Exception as e:
            print(f"⚠ Warning: Could not create plots: {e}")

    def plot_tuning_results(self, save_dir):
        """Create visualization of tuning results"""
        print(f"\nGenerating tuning visualizations...")

        # Performance over iterations
        fig, ax = plt.subplots(figsize=(12, 6))

        iterations = np.arange(len(self.cv_results_df))
        mean_scores = self.cv_results_df['mean_test_score']
        std_scores = self.cv_results_df['std_test_score']

        ax.plot(iterations, mean_scores, 'o-', color='steelblue',
                markersize=4, linewidth=1, alpha=0.6, label='Test R²')
        ax.fill_between(iterations,
                        mean_scores - std_scores,
                        mean_scores + std_scores,
                        alpha=0.2, color='steelblue')

        best_idx = mean_scores.idxmax()
        ax.plot(best_idx, mean_scores[best_idx], 'r*',
                markersize=20, label=f'Best (R²={mean_scores[best_idx]:.4f})')

        ax.set_xlabel('Iteration', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cross-Validation R²', fontweight='bold', fontsize=12)
        ax.set_title('Hyperparameter Search Progress', fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(alpha=0.3, linestyle='--')

        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'search_progress.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Search progress plot:       {plot_path}")
        plt.close()

    def train_best_model(self, X_train, y_train, best_params):
        """Train final model with best parameters"""
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

        try:
            self.best_model.fit(X_train, y_train)
        except Exception as e:
            print(f"\nError during model training: {e}")
            raise

        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Training score
        train_score = self.best_model.score(X_train, y_train)
        print(f"  Training R²: {train_score:.6f}")

        return self.best_model

    def predict_batched(self, X_test, original_shape, batch_size=32768):
        """
        BATCHED PREDICTION (Memory-efficient)
        Prevents memory errors with large models
        """
        print(f"\n{'='*80}")
        print("BATCHED PREDICTION AT COARSE RESOLUTION")
        print('='*80)

        n_samples = X_test.shape[0]
        t, h, w = original_shape

        print(f"Total samples:    {n_samples:,}")
        print(f"Batch size:       {batch_size:,}")
        print(f"Number of batches: {(n_samples + batch_size - 1) // batch_size}")

        # Pre-allocate output array
        y_pred_flat = np.zeros(n_samples, dtype=np.float32)

        start_time = time.time()

        # Predict in batches
        n_batches = (n_samples + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(n_batches), desc="Batch prediction"):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)

            # Predict batch
            X_batch = X_test[start_idx:end_idx]
            y_batch_pred = self.best_model.predict(X_batch)
            y_pred_flat[start_idx:end_idx] = y_batch_pred

            # Free memory
            del X_batch, y_batch_pred

            # Periodic garbage collection
            if batch_idx % 10 == 0:
                gc.collect()

        elapsed = time.time() - start_time
        print(f"\n✓ Prediction completed in {elapsed:.1f}s")

        # Reshape to spatial
        y_pred = y_pred_flat.reshape(t, h, w)
        y_pred = np.maximum(y_pred, 0)  # Non-negative

        print(f"\nPrediction statistics:")
        print(f"  Shape: {y_pred.shape}")
        print(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}] mm/day (mean per cell)")
        print(f"  Mean:  {y_pred.mean():.4f} mm/day (mean per cell)")

        return y_pred

    def evaluate(self, y_true, y_pred, prefix='Validation'):
        """Calculate evaluation metrics"""
        print(f"\n{'='*80}")
        print(f"{prefix.upper()} METRICS (COARSE RESOLUTION)")
        print('='*80)

        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        r2 = r2_score(y_true_flat, y_pred_flat)
        rmse = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
        mae = mean_absolute_error(y_true_flat, y_pred_flat)

        print(f"\nOverall metrics:")
        print(f"  R²:   {r2:.6f}")
        print(f"  RMSE: {rmse:.4f} mm/day")
        print(f"  MAE:  {mae:.4f} mm/day")

        return {'r2': r2, 'rmse': rmse, 'mae': mae}

    def plot_feature_importance(self, feature_names, save_dir):
        """Plot Gini importance"""
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
        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            y_pos = np.arange(len(sorted_features))
            ax.barh(y_pos, sorted_importance, color='steelblue', edgecolor='black', linewidth=0.5)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sorted_features, fontsize=10)
            ax.set_xlabel('Gini Importance', fontsize=12, fontweight='bold')
            ax.set_title('Feature Importance - Random Forest', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3, linestyle='--')
            ax.invert_yaxis()

            plt.tight_layout()

            plot_path = os.path.join(save_dir, 'gini_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved plot to: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"⚠ Warning: Could not create plot: {e}")

        return importance

    def bilinear_upscale(self, y_coarse, target_shape=(320, 320), save_path=None):
        """Simple bilinear interpolation upscaling"""
        print(f"\n{'='*80}")
        print("BILINEAR UPSCALING TO FINE RESOLUTION")
        print('='*80)

        t, h_low, w_low = y_coarse.shape
        h_high, w_high = target_shape

        factor_h = h_high // h_low
        factor_w = w_high // w_low

        print(f"Upscaling from {h_low}×{w_low} to {h_high}×{w_high}")
        print(f"Upscaling factor: {factor_h}×{factor_w}")

        # Step 1: Bilinear interpolation
        print(f"\nApplying bilinear interpolation...")
        start_time = time.time()
        zoom_factors = (1, h_high/h_low, w_high/w_low)
        y_fine_interp = zoom(y_coarse, zoom_factors, order=1, mode='nearest')
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.1f}s")

        # For mean pooling, bilinear interpolation naturally preserves mean values
        y_fine_mmday = y_fine_interp

        print(f"\nFinal upscaled predictions:")
        print(f"  Shape:  {y_fine_mmday.shape}")
        print(f"  Range:  [{y_fine_mmday.min():.4f}, {y_fine_mmday.max():.4f}] mm/day")
        print(f"  Mean:   {y_fine_mmday.mean():.4f} mm/day")

        if save_path:
            np.save(save_path, y_fine_mmday)
            print(f"✓ Saved to: {save_path}")
            print(f"  File size: {os.path.getsize(save_path) / 1e6:.2f} MB")

        return y_fine_mmday

    def save_model(self, save_path):
        """Save trained model"""
        print(f"\n✓ Saving model to: {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(self.best_model, f, protocol=4)

        model_size_gb = os.path.getsize(save_path) / 1e9
        print(f"  Model size: {model_size_gb:.2f} GB")

        if model_size_gb > 20:
            print(f"  ⚠ WARNING: Model is very large ({model_size_gb:.1f} GB)")
            print(f"  → Consider using more regularization (lower max_depth, higher min_samples_leaf)")


def mean_pool_coarsen(y_high, coarse_shape=(32, 32)):
    """Mean pooling coarsening - appropriate for intensity variables (mm/day)"""
    print(f"\n{'='*80}")
    print("MEAN POOLING COARSENING")
    print('='*80)

    t, h_high, w_high, c = y_high.shape
    h_coarse, w_coarse = coarse_shape

    factor_h = h_high // h_coarse
    factor_w = w_high // w_coarse

    print(f"From {h_high}×{w_high} to {h_coarse}×{w_coarse}")
    print(f"Pool size: {factor_h}×{factor_w}")

    start_time = time.time()

    # Use MEAN for intensity variables (mm/day)
    y_reshaped = y_high[:, :, :, 0].reshape(t, h_coarse, factor_h, w_coarse, factor_w)
    y_coarse_3d = y_reshaped.mean(axis=(2, 4))
    y_coarse = y_coarse_3d.reshape(t, h_coarse, w_coarse, 1).astype(np.float32)

    elapsed = time.time() - start_time
    print(f"\n✓ Coarsening completed in {elapsed:.1f}s")

    fine_mean = y_high.mean()
    coarse_mean = y_coarse.mean()

    print(f"\n✓ Mean preservation check:")
    print(f"  Fine mean:   {fine_mean:.4f} mm/day")
    print(f"  Coarse mean: {coarse_mean:.4f} mm/day")
    print(f"  Difference:  {abs(fine_mean - coarse_mean) / fine_mean * 100:.8f}%")

    return y_coarse


def create_netcdf_file(data, ref_ds_path, varname, start_date, end_date, 
                       calendar, save_dir, filename, delete_existing_file=True):
    """
    Create NetCDF file from numpy array using ai4klima utility
    
    Parameters
    ----------
    data : numpy.ndarray
        Input array to convert to NetCDF
    ref_ds_path : Path or str
        Path to reference dataset for spatial metadata
    varname : str
        Variable name in NetCDF file
    start_date : str
        Start date (YYYY-MM-DD)
    end_date : str
        End date (YYYY-MM-DD)
    calendar : str
        Calendar type (e.g., 'standard')
    save_dir : str
        Directory to save NetCDF file
    filename : str
        Output filename (including .nc extension)
    delete_existing_file : bool
        Whether to overwrite existing file
    """
    try:
        from ai4klima.utils import build_netcdf_from_array
        
        logger.info(f"Creating NetCDF: {filename}")
        logger.info(f"  Data shape: {data.shape}")
        logger.info(f"  Date range: {start_date} to {end_date}")
        
        build_netcdf_from_array(
            data,
            ref_ds=str(ref_ds_path),
            varname=varname,
            start_date=start_date,
            end_date=end_date,
            calendar=calendar,
            save_dir=save_dir,
            filename=filename,
            delete_existing_file=delete_existing_file,
        )
        logger.info(f"✓ NetCDF file created: {filename}")
        
    except ImportError:
        logger.warning("ai4klima not available - skipping NetCDF creation")
    except Exception as e:
        logger.exception(f"NetCDF creation failed: {e}")
        raise


def load_data(data_path):
    """Load atmospheric and precipitation data"""
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print('='*80)
    print(f"Data path: {data_path}")

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
        print("✓")

    X_low = np.stack(X_list, axis=-1)

    print(f"\nLoading precipitation...")
    y_high = np.load(target_file)
    if y_high.ndim == 3:
        y_high = np.expand_dims(y_high, axis=-1)

    print(f"\n✓ Data loaded:")
    print(f"  X (atmospheric): {X_low.shape}")
    print(f"  y (precip):      {y_high.shape}")

    return X_low, y_high, feature_names


def main_pipeline(data_path, save_dir, train_size=11323, tune_model=True, n_iter=50):
    """
    Complete RF pipeline with mean pooling and NetCDF export
    """
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*80)
    print("RF PERFECT PROGNOSIS - MEAN POOLING + NETCDF PIPELINE")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path:          {data_path}")
    print(f"  Save path:          {save_dir}")
    print(f"  Training size:      {train_size} days")
    print(f"  Tuning enabled:     {tune_model}")
    if tune_model:
        print(f"  Tuning iterations:  {n_iter}")

    # Load data
    X_low, y_high, feature_names = load_data(data_path)

    total_size = X_low.shape[0]
    val_size = total_size - train_size

    print(f"\nData split:")
    print(f"  Total:      {total_size} days")
    print(f"  Training:   {train_size} days")
    print(f"  Validation: {val_size} days")

    # MEAN POOLING coarsening (appropriate for intensity variables)
    y_low = mean_pool_coarsen(y_high, coarse_shape=(32, 32))

    # Save coarsened target
    y_low_path = os.path.join(save_dir, 'precipitation_coarse_32x32_mean.npy')
    np.save(y_low_path, y_low)
    print(f"\n✓ Saved coarse precipitation: {y_low_path}")

    # Split data
    X_train = X_low[:train_size]
    y_train = y_low[:train_size]

    X_val = X_low[train_size:]
    y_val = y_low[train_size:]
    y_val_high = y_high[train_size:]

    # Initialize model
    rf = RFWithTuning(n_jobs=-1, random_state=42)

    # Prepare data
    X_train_flat, y_train_flat = rf.prepare_data(X_train, y_train, save_dir, 'train')
    X_val_flat, y_val_flat = rf.prepare_data(X_val, y_val, save_dir, 'val')

    # Hyperparameter tuning
    if tune_model:
        best_params, best_cv_score = rf.hyperparameter_tuning(
            X_train_flat, y_train_flat,
            X_val_flat, y_val_flat,
            n_iter=n_iter, cv=3,
            save_dir=save_dir
        )
        rf.train_best_model(X_train_flat, y_train_flat, best_params)
    else:
        print("\n⚠ Skipping tuning, using default regularized parameters...")
        default_params = {
            'n_estimators': 300,
            'max_depth': 25,
            'min_samples_split': 100,
            'min_samples_leaf': 50,
            'max_features': 'sqrt',
            'bootstrap': True,
            'max_samples': 0.8
        }
        rf.train_best_model(X_train_flat, y_train_flat, default_params)

    # Save model
    model_path = os.path.join(save_dir, 'best_rf_model.pkl')
    rf.save_model(model_path)

    # BATCHED PREDICTIONS (memory-safe)
    print(f"\n{'='*80}")
    print("GENERATING PREDICTIONS (BATCHED)")
    print('='*80)

    y_train_pred = rf.predict_batched(X_train_flat, X_train.shape[:3], batch_size=32768)
    y_val_pred = rf.predict_batched(X_val_flat, X_val.shape[:3], batch_size=32768)

    # Save coarse predictions (numpy)
    print(f"\nSaving coarse predictions...")
    np.save(os.path.join(save_dir, 'train_predictions_32x32.npy'), y_train_pred)
    np.save(os.path.join(save_dir, 'val_predictions_32x32.npy'), y_val_pred)
    print("✓ Saved numpy arrays")

    # Evaluate at coarse resolution
    train_metrics = rf.evaluate(y_train.squeeze(), y_train_pred, 'Training')
    val_metrics_coarse = rf.evaluate(y_val.squeeze(), y_val_pred, 'Validation')

    # Feature importance
    importance = rf.plot_feature_importance(feature_names, save_dir)

    # Bilinear upscaling to fine resolution
    val_pred_fine_path = os.path.join(save_dir, 'val_predictions_upscaled_320x320.npy')
    y_val_pred_fine = rf.bilinear_upscale(y_val_pred, (320, 320), val_pred_fine_path)

    # Evaluate at fine resolution
    print(f"\n{'='*80}")
    print("FINE RESOLUTION EVALUATION")
    print('='*80)

    y_val_high_flat = y_val_high.flatten()
    y_val_pred_fine_flat = y_val_pred_fine.flatten()

    r2_fine = r2_score(y_val_high_flat, y_val_pred_fine_flat)
    rmse_fine = np.sqrt(mean_squared_error(y_val_high_flat, y_val_pred_fine_flat))
    mae_fine = mean_absolute_error(y_val_high_flat, y_val_pred_fine_flat)

    print(f"\nFine-resolution metrics (320×320):")
    print(f"  R²:   {r2_fine:.6f}")
    print(f"  RMSE: {rmse_fine:.4f} mm/day")
    print(f"  MAE:  {mae_fine:.4f} mm/day")

    val_metrics_fine = {'r2': r2_fine, 'rmse': rmse_fine, 'mae': mae_fine}

    # Save all metrics
    metrics_path = os.path.join(save_dir, 'metrics.npz')
    np.savez_compressed(metrics_path,
                        train_coarse=train_metrics,
                        val_coarse=val_metrics_coarse,
                        val_fine=val_metrics_fine)
    print(f"\n✓ Saved metrics: {metrics_path}")

    # Save fine-resolution target
    val_target_path = os.path.join(save_dir, 'val_target_fine_320x320.npy')
    np.save(val_target_path, y_val_high.squeeze())
    print(f"✓ Saved target: {val_target_path}")

    # =========================================================================
    # CREATE NETCDF FILES
    # =========================================================================
    print(f"\n{'='*80}")
    print("CREATING NETCDF FILES")
    print('='*80)

    ref_ds_path = Path("/home/midhunm/NIMBUS/protem/midhun/AI4KLIM/DATASETS/DATASET_IND32M/MASTER/grids/IND32M_GTOP_010_ELEV.nc")
    
    # NetCDF for validation predictions
    try:
        create_netcdf_file(
            y_val_pred_fine,
            ref_ds_path=ref_ds_path,
            varname="prec",
            start_date="2010-01-01",
            end_date="2023-12-31",
            calendar="standard",
            save_dir=save_dir,
            filename="val_predictions_upscaled_320x320.nc",
            delete_existing_file=True
        )
    except Exception as e:
        logger.warning(f"Could not create NetCDF for validation predictions: {e}")

    # NetCDF for validation target
    try:
        create_netcdf_file(
            y_val_high.squeeze(),
            ref_ds_path=ref_ds_path,
            varname="prec",
            start_date="2010-01-01",
            end_date="2023-12-31",
            calendar="standard",
            save_dir=save_dir,
            filename="val_target_fine_320x320.nc",
            delete_existing_file=True
        )
    except Exception as e:
        logger.warning(f"Could not create NetCDF for validation target: {e}")

    # Final summary
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print('='*80)
    print(f"\nKey outputs in {save_dir}:")
    print(f"  best_rf_model.pkl                         - Trained model")
    print(f"  train_predictions_32x32.npy               - Training predictions")
    print(f"  val_predictions_32x32.npy                 - Validation predictions (coarse)")
    print(f"  val_predictions_upscaled_320x320.npy      - Upscaled predictions (fine, numpy)")
    print(f"  val_predictions_upscaled_320x320.nc       - Upscaled predictions (fine, NetCDF)")
    print(f"  val_target_fine_320x320.npy               - Ground truth target (numpy)")
    print(f"  val_target_fine_320x320.nc                - Ground truth target (NetCDF)")
    print(f"  metrics.npz                               - Performance metrics")
    print(f"  gini_importance.npz & .png                - Feature importance")
    print(f"  best_params.json                          - Best hyperparameters")
    print('='*80)
    print(f"\nPerformance Summary:")
    print(f"  Coarse (32×32):                    R² = {val_metrics_coarse['r2']:.6f}")
    print(f"  Fine (320×320):                    R² = {val_metrics_fine['r2']:.6f}, RMSE = {rmse_fine:.4f} mm/day")
    print('='*80)

    return rf, y_val_pred_fine, val_metrics_fine


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        train_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 11323
        tune = sys.argv[4].lower() == 'true' if len(sys.argv) >= 5 else True
        n_iter = int(sys.argv[5]) if len(sys.argv) >= 6 else 50

        try:
            rf, predictions, metrics = main_pipeline(
                data_path=sys.argv[1],
                save_dir=sys.argv[2],
                train_size=train_size,
                tune_model=tune,
                n_iter=n_iter
            )
        except Exception as e:
            print(f"\nFATAL ERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("Usage: python rf_mean_pooling_netcdf.py <DATA_PATH> <SAVE_DIR> [TRAIN_SIZE] [TUNE] [N_ITER]")
        print("Example: python rf_mean_pooling_netcdf.py /data /output 11323 true 50")