# solar_energy_predictor.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class SolarEnergyPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the solar energy dataset"""
        print("Loading and preprocessing data...")
        
        # Load data from NASA POWER API
        df = pd.read_csv(file_path)
        
        # Display basic info
        print(f"Dataset shape: {df.shape}")
        print("\nColumns:", df.columns.tolist())
        print("\nFirst few rows:")
        print(df.head())
        
        # Data cleaning
        df_clean = self._clean_data(df)
        
        # Feature engineering
        df_processed = self._engineer_features(df_clean)
        
        return df_processed
    
    def _clean_data(self, df):
        """Clean the raw dataset"""
        # Create a copy
        df_clean = df.copy()
        
        # Convert date column
        if 'YYYYMMDD' in df_clean.columns:
            df_clean['DATE'] = pd.to_datetime(df_clean['YYYYMMDD'], format='%Y%m%d')
        elif 'YEAR' in df_clean.columns and 'MO' in df_clean.columns and 'DY' in df_clean.columns:
            df_clean['DATE'] = pd.to_datetime(df_clean[['YEAR', 'MO', 'DY']])
        
        # Set date as index
        df_clean.set_index('DATE', inplace=True)
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df_clean.isnull().sum())
        
        # Remove rows with missing target variable
        df_clean = df_clean.dropna(subset=['ALLSKY_SFC_SW_DWN'])
        
        # Fill missing features with forward fill
        df_clean = df_clean.ffill()
        
        print(f"Data after cleaning: {df_clean.shape}")
        return df_clean
    
    def _engineer_features(self, df):
        """Create additional features for better prediction"""
        df_eng = df.copy()
        
        # Original features from NASA POWER
        # ALLSKY_SFC_SW_DWN - Solar radiation (target)
        # T2M - Temperature at 2m
        # RH2M - Relative humidity at 2m
        # WS2M - Wind speed at 2m
        # PRECTOTCORR - Precipitation
        
        # Time-based features
        df_eng['DAY_OF_YEAR'] = df_eng.index.dayofyear
        df_eng['MONTH'] = df_eng.index.month
        df_eng['SEASON'] = (df_eng.index.month % 12 + 3) // 3  # 1:Winter, 2:Spring, etc.
        
        # Cyclical encoding for seasonal patterns
        df_eng['DAY_SIN'] = np.sin(2 * np.pi * df_eng['DAY_OF_YEAR'] / 365)
        df_eng['DAY_COS'] = np.cos(2 * np.pi * df_eng['DAY_OF_YEAR'] / 365)
        
        # Weather interaction features
        df_eng['TEMP_HUMIDITY'] = df_eng['T2M'] * df_eng['RH2M']
        df_eng['WIND_TEMP'] = df_eng['WS2M'] * df_eng['T2M']
        
        # Lag features for temporal patterns
        df_eng['SOLAR_LAG_1'] = df_eng['ALLSKY_SFC_SW_DWN'].shift(1)
        df_eng['SOLAR_LAG_7'] = df_eng['ALLSKY_SFC_SW_DWN'].shift(7)
        
        # Rolling statistics
        df_eng['SOLAR_7D_AVG'] = df_eng['ALLSKY_SFC_SW_DWN'].rolling(7).mean()
        
        # Remove rows with NaN from lag features
        df_eng = df_eng.dropna()
        
        print(f"Data after feature engineering: {df_eng.shape}")
        return df_eng
    
    def prepare_features_target(self, df):
        """Prepare features and target variable"""
        # Target variable
        target = 'ALLSKY_SFC_SW_DWN'
        
        # Feature columns (excluding target and date-related columns for basic model)
        feature_columns = ['T2M', 'RH2M', 'WS2M', 'PRECTOTCORR', 
                          'DAY_SIN', 'DAY_COS', 'TEMP_HUMIDITY', 'WIND_TEMP',
                          'SOLAR_LAG_1', 'SOLAR_LAG_7', 'SOLAR_7D_AVG']
        
        # Only use columns that exist in dataframe
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        X = df[feature_columns]
        y = df[target]
        
        print(f"Features: {feature_columns}")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        
        return X, y, feature_columns
    
    def split_and_scale_data(self, X, y):
        """Split data and scale features"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Testing set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple ML models"""
        print("\nTraining models...")
        
        # Traditional ML Models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            self.models[name] = {
                'model': model,
                'predictions': y_pred,
                'metrics': self._calculate_metrics(y_test, y_pred)
            }
            print(f"{name} trained - MAE: {self.models[name]['metrics']['mae']:.4f}")
        
        # Neural Network
        nn_model = self._build_neural_network(X_train.shape[1])
        history = nn_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            verbose=0
        )
        
        y_pred_nn = nn_model.predict(X_test).flatten()
        
        self.models['Neural Network'] = {
            'model': nn_model,
            'predictions': y_pred_nn,
            'metrics': self._calculate_metrics(y_test, y_pred_nn),
            'history': history
        }
        
        print(f"Neural Network trained - MAE: {self.models['Neural Network']['metrics']['mae']:.4f}")
    
    def _build_neural_network(self, input_dim):
        """Build a neural network model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1)
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred)
        }
    
    def evaluate_models(self, X_test, y_test):
        """Comprehensive model evaluation"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        results = []
        for name, model_info in self.models.items():
            metrics = model_info['metrics']
            results.append({
                'Model': name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2']
            })
        
        results_df = pd.DataFrame(results)
        print(results_df.round(4))
        
        # Store results
        self.results['model_comparison'] = results_df
        
        # Identify best model
        best_model_name = results_df.loc[results_df['MAE'].idxmin(), 'Model']
        print(f"\nBest Model: {best_model_name}")
        
        return best_model_name
    
    def plot_results(self, X_test, y_test, feature_names):
        """Create comprehensive visualization of results"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Solar Energy Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted for best model
        best_model_name = self.results['model_comparison'].loc[
            self.results['model_comparison']['MAE'].idxmin(), 'Model'
        ]
        best_predictions = self.models[best_model_name]['predictions']
        
        axes[0,0].scatter(y_test, best_predictions, alpha=0.6)
        axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('Actual Solar Radiation (kWh/m²/day)')
        axes[0,0].set_ylabel('Predicted Solar Radiation (kWh/m²/day)')
        axes[0,0].set_title(f'Actual vs Predicted - {best_model_name}')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Model comparison
        models = self.results['model_comparison']['Model']
        mae_scores = self.results['model_comparison']['MAE']
        
        bars = axes[0,1].bar(models, mae_scores, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'])
        axes[0,1].set_ylabel('MAE (Lower is Better)')
        axes[0,1].set_title('Model Performance Comparison (MAE)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, mae_scores):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Feature importance
        rf_model = self.models['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        axes[0,2].barh(range(len(indices)), importances[indices], align='center')
        axes[0,2].set_yticks(range(len(indices)))
        axes[0,2].set_yticklabels([feature_names[i] for i in indices])
        axes[0,2].set_xlabel('Feature Importance')
        axes[0,2].set_title('Random Forest Feature Importance')
        
        # 4. Time series plot (sample of test data)
        sample_size = min(100, len(y_test))
        test_dates = range(sample_size)
        
        axes[1,0].plot(test_dates, y_test.values[:sample_size], label='Actual', linewidth=2)
        axes[1,0].plot(test_dates, best_predictions[:sample_size], label='Predicted', alpha=0.8)
        axes[1,0].set_xlabel('Time Index')
        axes[1,0].set_ylabel('Solar Radiation (kWh/m²/day)')
        axes[1,0].set_title('Time Series: Actual vs Predicted')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Residual plot
        residuals = y_test - best_predictions
        axes[1,1].scatter(best_predictions, residuals, alpha=0.6)
        axes[1,1].axhline(y=0, color='red', linestyle='--')
        axes[1,1].set_xlabel('Predicted Values')
        axes[1,1].set_ylabel('Residuals')
        axes[1,1].set_title('Residual Analysis')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Error distribution
        axes[1,2].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1,2].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1,2].set_xlabel('Prediction Error')
        axes[1,2].set_ylabel('Frequency')
        axes[1,2].set_title('Distribution of Prediction Errors')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('solar_energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def ethical_analysis(self):
        """Analyze ethical considerations and biases"""
        print("\n" + "="*50)
        print("ETHICAL ANALYSIS & SUSTAINABILITY IMPACT")
        print("="*50)
        
        ethical_considerations = {
            "Data Bias Risks": [
                "Geographic bias: Model trained on Nairobi data may not generalize to other regions",
                "Temporal bias: Limited to 2018-2023, may not capture long-term climate patterns",
                "Weather station bias: Single location may not represent entire region"
            ],
            "Mitigation Strategies": [
                "Incorporate data from multiple geographic locations",
                "Use transfer learning for new regions with limited data",
                "Implement uncertainty quantification in predictions",
                "Regular model retraining with new data"
            ],
            "Sustainability Impact": [
                "Enables better solar resource assessment for clean energy planning",
                "Supports grid stability through improved renewable energy forecasting",
                "Reduces reliance on fossil fuels by optimizing solar integration",
                "Promotes energy access in underserved communities"
            ],
            "SDG 7 Alignment": [
                "Affordable and Clean Energy: Optimizes solar power generation",
                "Climate Action: Supports transition to renewable energy",
                "Sustainable Cities: Enables smart grid operations",
                "Industry Innovation: Advances predictive analytics for energy"
            ]
        }
        
        for category, points in ethical_considerations.items():
            print(f"\n{category}:")
            for point in points:
                print(f"  • {point}")

def main():
    """Main execution function"""
    # Initialize the predictor
    predictor = SolarEnergyPredictor()
    
    # Since we can't download the file directly in this environment,
    # I'll create a sample dataset that mimics the NASA POWER data structure
    def create_sample_data():
        """Create sample data matching NASA POWER API structure"""
        dates = pd.date_range('2018-01-01', '2023-12-31', freq='D')
        
        # Simulate realistic seasonal patterns for Nairobi, Kenya
        n_days = len(dates)
        
        # Base solar radiation with seasonal pattern
        day_of_year = dates.dayofyear
        base_radiation = 4 + 2 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        # Add weather effects
        temperature = 20 + 8 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2, n_days)
        humidity = 60 + 20 * np.sin(2 * np.pi * (day_of_year + 100) / 365) + np.random.normal(0, 10, n_days)
        wind_speed = 3 + np.random.exponential(1, n_days)
        precipitation = np.random.exponential(0.5, n_days)
        
        # Add noise and correlations
        radiation = (base_radiation + 
                    0.1 * (temperature - 20) - 
                    0.02 * humidity + 
                    0.5 * wind_speed - 
                    2 * precipitation + 
                    np.random.normal(0, 0.5, n_days))
        
        # Ensure positive values
        radiation = np.maximum(radiation, 0)
        
        df = pd.DataFrame({
            'YYYYMMDD': dates.strftime('%Y%m%d'),
            'ALLSKY_SFC_SW_DWN': radiation,
            'T2M': temperature,
            'RH2M': humidity,
            'WS2M': wind_speed,
            'PRECTOTCORR': precipitation,
            'YEAR': dates.year,
            'MO': dates.month,
            'DY': dates.day
        })
        
        return df
    
    print("Creating sample solar energy dataset...")
    sample_df = create_sample_data()
    sample_df.to_csv('nasa_power_solar_data.csv', index=False)
    
    # Load and preprocess data
    df_processed = predictor.load_and_preprocess_data('nasa_power_solar_data.csv')
    
    # Prepare features and target
    X, y, feature_names = predictor.prepare_features_target(df_processed)
    
    # Split and scale data
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = predictor.split_and_scale_data(X, y)
    
    # Train models
    predictor.train_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Evaluate models
    best_model = predictor.evaluate_models(X_test_scaled, y_test)
    
    # Plot results
    predictor.plot_results(X_test_scaled, y_test, feature_names)
    
    # Ethical analysis
    predictor.ethical_analysis()
    
    # Print summary for report
    print("\n" + "="*50)
    print("PROJECT SUMMARY")
    print("="*50)
    print("SDG Problem: SDG 7 - Affordable and Clean Energy")
    print("ML Approach: Multi-model regression for solar energy prediction")
    print(f"Best Model: {best_model}")
    best_metrics = predictor.models[best_model]['metrics']
    print(f"Best Model Performance:")
    print(f"  MAE: {best_metrics['mae']:.4f} kWh/m²/day")
    print(f"  RMSE: {best_metrics['rmse']:.4f} kWh/m²/day")
    print(f"  R²: {best_metrics['r2']:.4f}")
    print("\nKey Impact: Enables better solar planning and grid integration")
    print("Sustainability: Supports transition to renewable energy sources")

if __name__ == "__main__":
    main()