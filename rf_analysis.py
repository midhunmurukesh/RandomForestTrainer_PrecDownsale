# v2.2: RF with Extended Hyperparameter Tuning & Results Saving
# Expanded search space up to 1000 trees with detailed logging


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

warnings.filterwarnings('ignore')

class RFWithTuning:
    """
    Random Forest with Extended Hyperparameter Tuning for Precipitation Downscaling
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
        print(f"  y range: [{y_flat.min():.4f}, {y_flat.max():.4f}] mm (total per cell)")
        print(f"  y mean:  {y_flat.mean():.4f} mm (total per cell)")
        
        return X_flat, y_flat
    
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val, 
                             n_iter=50, cv=3, save_dir=None):
        """
        Extended hyperparameter tuning with comprehensive results saving
        
        Parameters:
        -----------
        X_train : array - training features
        y_train : array - training targets
        X_val : array - validation features
        y_val : array - validation targets
        n_iter : int - number of parameter settings sampled
        cv : int - cross-validation folds
        save_dir : str - directory to save tuning results
        """
        print(f"\n{'='*80}")
        print("EXTENDED HYPERPARAMETER TUNING")
        print('='*80)
        
        # EXTENDED parameter grid (up to 1000 trees)
        param_distributions = {
            'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            'max_depth': [15, 20, 25, 30, 35, 40, 45, 50, None],
            'min_samples_split': [5, 10, 15, 20, 30, 50, 100],
            'min_samples_leaf': [1, 2, 5, 7, 10, 15, 20],
            'max_features': [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 'sqrt', 'log2'],
            'bootstrap': [True],
            'max_samples': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
        }
        
        print(f"\nExtended parameter search space:")
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
        max_tuning_samples = 100000  # Increased for better estimates
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
            n_jobs=1,  # Parallel at tree level
            verbose=2,
            random_state=self.random_state,
            return_train_score=True,
            refit=True
        )
        
        print("\nStarting hyperparameter search...")
        print(f"Estimated time: {n_iter * cv * 2 / 60:.1f} - {n_iter * cv * 5 / 60:.1f} minutes")
        start_time = time.time()
        
        search.fit(X_tune, y_tune)
        
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
        print(f"  Validation RMSE:     {val_rmse:.4f} mm")
        print(f"  Validation MAE:      {val_mae:.4f} mm")
        
        # Check for overfitting
        train_r2 = self.tuning_results['mean_train_score'][search.best_index_]
        overfit_gap = train_r2 - search.best_score_
        
        print(f"\n  Train R² (CV):       {train_r2:.6f}")
        print(f"  Test R² (CV):        {search.best_score_:.6f}")
        print(f"  Overfit gap:         {overfit_gap:.6f}")
        if overfit_gap > 0.05:
            print(f"  ⚠ WARNING: Possible overfitting detected!")
        else:
            print(f"  ✓ No significant overfitting")
        
        # Store best estimator
        self.best_model = search.best_estimator_
        
        # Save tuning results if directory provided
        if save_dir:
            self.save_tuning_results(save_dir, elapsed)
        
        return self.best_params, search.best_score_
    
    def save_tuning_results(self, save_dir, elapsed_time):
        """
        Save comprehensive tuning results to multiple formats
        """
        print(f"\n{'='*80}")
        print("SAVING TUNING RESULTS")
        print('='*80)
        
        # 1. Save best parameters (JSON)
        best_params_path = os.path.join(save_dir, 'best_params.json')
        with open(best_params_path, 'w') as f:
            # Convert numpy types to Python types for JSON
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
            f.write(f"Total iterations: {len(self.cv_results_df)}\n")
            f.write(f"CV folds: {self.cv_results_df['split0_test_score'].notna().sum()}\n\n")
            
            f.write("Best parameters:\n")
            for param, value in self.best_params.items():
                f.write(f"  {param:20s}: {value}\n")
            
            f.write(f"\nBest CV R²: {self.cv_results_df['mean_test_score'].max():.6f}\n")
            f.write(f"Best CV std: {self.cv_results_df.loc[self.cv_results_df['mean_test_score'].idxmax(), 'std_test_score']:.6f}\n")
            
            f.write(f"\nParameter importance (range of R² across values):\n")
            for param in self.best_params.keys():
                if param in ['n_estimators', 'max_depth', 'max_features']:
                    param_col = f'param_{param}'
                    if param_col in self.cv_results_df.columns:
                        grouped = self.cv_results_df.groupby(param_col)['mean_test_score']
                        r2_range = grouped.max().max() - grouped.max().min()
                        f.write(f"  {param:20s}: Δ R² = {r2_range:.6f}\n")
        
        print(f"✓ Tuning summary (TXT):       {summary_path}")
        
        # 5. Save raw CV results (NPZ)
        cv_results_npz_path = os.path.join(save_dir, 'cv_results_raw.npz')
        np.savez_compressed(cv_results_npz_path, **self.tuning_results)
        print(f"✓ Raw CV results (NPZ):       {cv_results_npz_path}")
        
        # 6. Create visualization plots
        self.plot_tuning_results(save_dir)
        
        print(f"\n✓ All tuning results saved to: {save_dir}")
    
    def plot_tuning_results(self, save_dir):
        """
        Create comprehensive visualization of tuning results
        """
        print(f"\nGenerating tuning visualizations...")
        
        # Figure 1: Hyperparameter importance
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Hyperparameter Tuning Results', fontsize=16, fontweight='bold')
        
        params_to_plot = ['n_estimators', 'max_depth', 'max_features', 
                         'min_samples_split', 'min_samples_leaf', 'max_samples']
        
        for idx, param in enumerate(params_to_plot):
            ax = axes[idx // 3, idx % 3]
            param_col = f'param_{param}'
            
            if param_col in self.cv_results_df.columns:
                # Group by parameter value
                grouped = self.cv_results_df.groupby(param_col)['mean_test_score']
                
                param_values = []
                r2_means = []
                r2_stds = []
                
                for val in grouped.groups.keys():
                    param_values.append(str(val))
                    r2_means.append(grouped.get_group(val).mean())
                    r2_stds.append(grouped.get_group(val).std())
                
                x_pos = np.arange(len(param_values))
                ax.bar(x_pos, r2_means, yerr=r2_stds, alpha=0.7, 
                      color='steelblue', edgecolor='black', linewidth=0.5,
                      error_kw={'linewidth': 1, 'ecolor': 'red'})
                ax.set_xticks(x_pos)
                ax.set_xticklabels(param_values, rotation=45, ha='right')
                ax.set_xlabel(param.replace('_', ' ').title(), fontweight='bold')
                ax.set_ylabel('Mean Test R²', fontweight='bold')
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                
                # Highlight best value
                best_idx = np.argmax(r2_means)
                ax.axvline(best_idx, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        plt.tight_layout()
        plot1_path = os.path.join(save_dir, 'hyperparameter_importance.png')
        plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
        print(f"✓ Hyperparameter importance:  {plot1_path}")
        plt.close()
        
        # Figure 2: Performance over iterations
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
        
        # Highlight best
        best_idx = mean_scores.idxmax()
        ax.plot(best_idx, mean_scores[best_idx], 'r*', 
               markersize=20, label=f'Best (R²={mean_scores[best_idx]:.4f})')
        
        ax.set_xlabel('Iteration', fontweight='bold', fontsize=12)
        ax.set_ylabel('Cross-Validation R²', fontweight='bold', fontsize=12)
        ax.set_title('Hyperparameter Search Progress', fontweight='bold', fontsize=14)
        ax.legend(loc='best')
        ax.grid(alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plot2_path = os.path.join(save_dir, 'search_progress.png')
        plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
        print(f"✓ Search progress:            {plot2_path}")
        plt.close()
        
        # Figure 3: n_estimators vs R² (key parameter)
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if 'param_n_estimators' in self.cv_results_df.columns:
            n_est_groups = self.cv_results_df.groupby('param_n_estimators')['mean_test_score']
            
            n_estimators = []
            r2_means = []
            r2_stds = []
            
            for n_est in sorted(n_est_groups.groups.keys()):
                n_estimators.append(n_est)
                scores = n_est_groups.get_group(n_est)
                r2_means.append(scores.mean())
                r2_stds.append(scores.std())
            
            n_estimators = np.array(n_estimators)
            r2_means = np.array(r2_means)
            r2_stds = np.array(r2_stds)
            
            ax.errorbar(n_estimators, r2_means, yerr=r2_stds, 
                       fmt='o-', color='steelblue', markersize=8,
                       linewidth=2, capsize=5, capthick=2, 
                       ecolor='red', label='Mean ± Std')
            
            ax.set_xlabel('Number of Trees (n_estimators)', fontweight='bold', fontsize=12)
            ax.set_ylabel('Cross-Validation R²', fontweight='bold', fontsize=12)
            ax.set_title('Effect of Number of Trees on Performance', 
                        fontweight='bold', fontsize=14)
            ax.legend(loc='best')
            ax.grid(alpha=0.3, linestyle='--')
            ax.set_xscale('log')
            
            plt.tight_layout()
            plot3_path = os.path.join(save_dir, 'n_estimators_effect.png')
            plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
            print(f"✓ Trees effect:               {plot3_path}")
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
        
        self.best_model.fit(X_train, y_train)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Training score
        train_score = self.best_model.score(X_train, y_train)
        print(f"  Training R²: {train_score:.6f}")
        
        return self.best_model
    
    def predict(self, X_test, original_shape):
        """Predict and reshape to original spatial dimensions"""
        print(f"\n{'='*80}")
        print("PREDICTION AT COARSE RESOLUTION")
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
        print(f"  Range: [{y_pred.min():.4f}, {y_pred.max():.4f}] mm (total per cell)")
        print(f"  Mean:  {y_pred.mean():.4f} mm (total per cell)")
        
        return y_pred
    
    def evaluate(self, y_true, y_pred, prefix='Validation'):
        """Calculate evaluation metrics at coarse resolution"""
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
        print(f"  RMSE: {rmse:.4f} mm")
        print(f"  MAE:  {mae:.4f} mm")
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
    
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
    
    def mass_conserving_upscale(self, y_coarse, target_shape=(320, 320), save_path=None):
        """Mass-conserving bilinear upscaling from 32×32 to 320×320"""
        print(f"\n{'='*80}")
        print("MASS-CONSERVING UPSCALING TO FINE RESOLUTION")
        print('='*80)
        
        t, h_low, w_low = y_coarse.shape
        h_high, w_high = target_shape
        
        factor_h = h_high // h_low
        factor_w = w_high // w_low
        
        print(f"Upscaling from {h_low}×{w_low} to {h_high}×{w_high}")
        print(f"Upscaling factor: {factor_h}×{factor_w}")
        print(f"Method: Bilinear interpolation + mass conservation correction")
        
        # Step 1: Bilinear interpolation
        print(f"\nStep 1: Bilinear interpolation...")
        start_time = time.time()
        zoom_factors = (1, h_high/h_low, w_high/w_low)
        y_fine_interp = zoom(y_coarse, zoom_factors, order=1, mode='nearest')
        elapsed = time.time() - start_time
        print(f"✓ Completed in {elapsed:.1f}s")
        
        print(f"\nBefore mass correction:")
        print(f"  Coarse total: {y_coarse.sum():,.2f} mm")
        print(f"  Fine total:   {y_fine_interp.sum():,.2f} mm")
        print(f"  Mass error:   {abs(y_coarse.sum() - y_fine_interp.sum())/y_coarse.sum()*100:.2f}%")
        
        # Step 2: Mass conservation correction
        print(f"\nStep 2: Applying mass conservation correction...")
        y_fine_corrected = np.zeros_like(y_fine_interp)
        
        start_time = time.time()
        for ti in tqdm(range(t), desc="Mass correction"):
            for i in range(h_low):
                for j in range(w_low):
                    target_sum = y_coarse[ti, i, j]
                    fine_box = y_fine_interp[ti, 
                                            i*factor_h:(i+1)*factor_h,
                                            j*factor_w:(j+1)*factor_w]
                    current_sum = fine_box.sum()
                    
                    if current_sum > 0:
                        correction_factor = target_sum / current_sum
                        y_fine_corrected[ti,
                                        i*factor_h:(i+1)*factor_h,
                                        j*factor_w:(j+1)*factor_w] = fine_box * correction_factor
                    else:
                        uniform_val = target_sum / (factor_h * factor_w)
                        y_fine_corrected[ti,
                                        i*factor_h:(i+1)*factor_h,
                                        j*factor_w:(j+1)*factor_w] = uniform_val
        
        elapsed = time.time() - start_time
        print(f"✓ Correction completed in {elapsed:.1f}s")
        
        print(f"\nAfter mass correction:")
        print(f"  Fine total:   {y_fine_corrected.sum():,.2f} mm")
        print(f"  Mass error:   {abs(y_coarse.sum() - y_fine_corrected.sum())/y_coarse.sum()*100:.8f}%")
        
        # Convert from total mm to mm/day
        y_fine_mmday = y_fine_corrected / (factor_h * factor_w)
        
        print(f"\nFinal fine-resolution predictions:")
        print(f"  Shape:  {y_fine_mmday.shape}")
        print(f"  Range:  [{y_fine_mmday.min():.4f}, {y_fine_mmday.max():.4f}] mm/day")
        print(f"  Mean:   {y_fine_mmday.mean():.4f} mm/day")
        print(f"  99th:   {np.percentile(y_fine_mmday, 99):.4f} mm/day")
        
        if save_path:
            np.save(save_path, y_fine_mmday)
            print(f"✓ Saved to: {save_path}")
        
        return y_fine_mmday
    
    def save_model(self, save_path):
        """Save trained model"""
        with open(save_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        print(f"✓ Model saved to: {save_path}")



def conservative_coarsen(y_high, coarse_shape=(32, 32)):
    """Conservative remapping using SUM (vectorized)"""
    print(f"\n{'='*80}")
    print("CONSERVATIVE COARSENING (SUM)")
    print('='*80)
    
    t, h_high, w_high, c = y_high.shape
    h_coarse, w_coarse = coarse_shape
    
    factor_h = h_high // h_coarse
    factor_w = w_high // w_coarse
    
    print(f"From {h_high}×{w_high} to {h_coarse}×{w_coarse}")
    print(f"Box size: {factor_h}×{factor_w} = {factor_h*factor_w} pixels per coarse cell")
    print(f"Method: CONSERVATIVE SUM (mass-preserving)")
    
    start_time = time.time()
    
    y_reshaped = y_high[:, :, :, 0].reshape(
        t, 
        h_coarse, factor_h,
        w_coarse, factor_w
    )
    
    y_coarse_3d = y_reshaped.sum(axis=(2, 4))
    y_coarse = y_coarse_3d.reshape(t, h_coarse, w_coarse, 1).astype(np.float32)
    
    elapsed = time.time() - start_time
    print(f"\n✓ Coarsening completed in {elapsed:.1f}s")
    
    fine_total = y_high.sum()
    coarse_total = y_coarse.sum()
    mass_error = abs(fine_total - coarse_total) / fine_total * 100
    
    print(f"\n✓ Mass conservation:")
    print(f"  Fine total:     {fine_total:,.2f} mm")
    print(f"  Coarse total:   {coarse_total:,.2f} mm")
    print(f"  Error:          {mass_error:.8f}%")
    
    print(f"\n✓ Value preservation:")
    print(f"  Fine mean:      {y_high.mean():.4f} mm/day")
    print(f"  Coarse mean:    {y_coarse.mean():.4f} mm (total per cell)")
    print(f"  Fine max:       {y_high.max():.2f} mm/day")
    print(f"  Coarse max:     {y_coarse.max():.2f} mm (total per cell)")
    
    return y_coarse



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



def main_pipeline(data_path, save_dir, train_size=11323, tune_model=True, n_iter=50):
    """
    Complete RF pipeline with extended hyperparameter tuning
    
    Parameters:
    -----------
    data_path : str
    save_dir : str
    train_size : int - training timesteps
    tune_model : bool - whether to do hyperparameter tuning
    n_iter : int - number of tuning iterations (recommended: 50-100)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("RF PERFECT PROGNOSIS WITH EXTENDED TUNING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Data path:          {data_path}")
    print(f"  Save path:          {save_dir}")
    print(f"  Training size:      {train_size} days")
    print(f"  Tuning enabled:     {tune_model}")
    if tune_model:
        print(f"  Tuning iterations:  {n_iter}")
        print(f"  Max trees:          1000")
    
    # Load data
    X_low, y_high, feature_names = load_data(data_path)
    
    total_size = X_low.shape[0]
    val_size = total_size - train_size
    
    print(f"\nData split:")
    print(f"  Total:      {total_size} days")
    print(f"  Training:   {train_size} days")
    print(f"  Validation: {val_size} days")
    
    # Conservative coarsening
    y_low = conservative_coarsen(y_high, coarse_shape=(32, 32))
    
    # Save coarsened target
    y_low_path = os.path.join(save_dir, 'precipitation_coarse_32x32_sum.npy')
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
            save_dir=save_dir  # Pass save_dir to save results
        )
        
        # Train final model
        rf.train_best_model(X_train_flat, y_train_flat, best_params)
    else:
        print("\n⚠ Skipping tuning, using default parameters...")
        default_params = {
            'n_estimators': 300,
            'max_depth': 30,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'bootstrap': True,
            'max_samples': 0.8
        }
        rf.train_best_model(X_train_flat, y_train_flat, default_params)
    
    # Save model
    model_path = os.path.join(save_dir, 'best_rf_model.pkl')
    rf.save_model(model_path)
    
    # Predictions
    y_train_pred = rf.predict(X_train_flat, X_train.shape[:3])
    y_val_pred = rf.predict(X_val_flat, X_val.shape[:3])
    
    # Save coarse predictions
    np.save(os.path.join(save_dir, 'train_predictions_32x32.npy'), y_train_pred)
    np.save(os.path.join(save_dir, 'val_predictions_32x32.npy'), y_val_pred)
    
    # Evaluate at coarse resolution
    train_metrics = rf.evaluate(y_train.squeeze(), y_train_pred, 'Training')
    val_metrics_coarse = rf.evaluate(y_val.squeeze(), y_val_pred, 'Validation')
    
    # Feature importance
    importance = rf.plot_feature_importance(feature_names, save_dir)
    
    # Mass-conserving upscaling
    val_pred_fine_path = os.path.join(save_dir, 'val_predictions_upscaled_320x320.npy')
    y_val_pred_fine = rf.mass_conserving_upscale(y_val_pred, (320, 320), val_pred_fine_path)
    
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
    
    # Final summary
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETED!")
    print('='*80)
    print(f"\nKey outputs in {save_dir}:")
    print(f"  best_rf_model.pkl                         - Trained model")
    print(f"  best_params.json                          - Best hyperparameters")
    print(f"  cv_results_all.csv                        - All CV results")
    print(f"  cv_results_top50.csv                      - Top 50 configurations")
    print(f"  tuning_summary.txt                        - Tuning summary")
    print(f"  hyperparameter_importance.png             - Parameter visualization")
    print(f"  search_progress.png                       - Search progress")
    print(f"  n_estimators_effect.png                   - Trees effect")
    print(f"  gini_importance.npz & .png                - Feature importance")
    print(f"  val_predictions_upscaled_320x320.npy      - Final predictions")
    print(f"  metrics.npz                               - Performance metrics")
    print('='*80)
    print(f"\nPerformance Summary:")
    print(f"  Coarse (32×32):  R² = {val_metrics_coarse['r2']:.6f}")
    print(f"  Fine (320×320):  R² = {val_metrics_fine['r2']:.6f}, RMSE = {rmse_fine:.4f} mm/day")
    print('='*80)
    
    return rf, y_val_pred_fine, val_metrics_fine

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) >= 3:
        train_size = int(sys.argv[3]) if len(sys.argv) >= 4 else 11323
        tune = sys.argv[4].lower() == 'true' if len(sys.argv) >= 5 else True
        n_iter = int(sys.argv[5]) if len(sys.argv) >= 6 else 50
        
        rf, predictions, metrics = main_pipeline(
            data_path=sys.argv[1],
            save_dir=sys.argv[2],
            train_size=train_size,
            tune_model=tune,
            n_iter=n_iter
        )
    else:
        print("Usage: python rf_extended_tuning.py <DATA_PATH> <SAVE_DIR> [TRAIN_SIZE] [TUNE] [N_ITER]")
        print("Example: python rf_extended_tuning.py /data /output 11323 true 50")
