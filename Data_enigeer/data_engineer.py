import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta

class MarketDataEngineer:
    def __init__(self, fred_api_key):
        self.fred = Fred(api_key=fred_api_key)
        self.start_date = '2000-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.series_ids = {
            'VIXCLS': 'vix_level',           # VIX Volatility Index
            'SP500': 'sp500_level',          # S&P 500 Index
            'FEDFUNDS': 'fed_rate',          # Federal Funds Rate
            'DGS10': 'treasury_10y',         # 10-Year Treasury Rate
            'UNRATE': 'unemployment'         # Unemployment Rate
        }

    def get_all_data(self):
        """
        Collects and processes data from FRED.
        Returns a DataFrame with aligned and cleaned financial data.
        """
        raw_data = pd.DataFrame()
        
        # First, collect all the data series
        for series_id, column_name in self.series_ids.items():
            try:
                # Fetch the data from FRED
                series = self.fred.get_series(series_id, 
                                            observation_start=self.start_date,
                                            observation_end=self.end_date)
                
                series.index = pd.to_datetime(series.index)
                
                # Convert to daily frequency if needed
                if not series.index.freq:
                    series = series.asfreq('D')
                
                # Forward fill missing values with a 5-day limit
                series = series.fillna(method='ffill', limit=5)
                
                # Add the processed series to our DataFrame
                raw_data[column_name] = series
                
                print(f"Successfully retrieved {column_name} data with {len(series)} observations")
                print(f"Date range for {column_name}: {series.index.min()} to {series.index.max()}")
                
            except Exception as e:
                print(f"Error fetching {series_id}: {str(e)}")
        
        if not raw_data.empty:
            # Find the overlapping date range across all series
            all_dates = pd.DataFrame(index=pd.date_range(start=raw_data.index.min(),
                                                        end=raw_data.index.max(),
                                                        freq='D'))
            
            # Merge with data to ensure consistent daily dates
            raw_data = all_dates.join(raw_data)
            
            print("\nData Quality Summary:")
            print(f"Total rows: {len(raw_data)}")
            print(f"Date range: {raw_data.index.min()} to {raw_data.index.max()}")
            print("\nMissing values per column:")
            print(raw_data.isnull().sum())
            
            # Print sample of the data to verify its structure
            print("\nSample of the first few rows:")
            print(raw_data.head())
        else:
            print("Warning: No data was collected!")
        
        return raw_data
    def engineer_market_features(self, data):
        """
        Create market-specific features with improved handling of missing values.
        """
        market_features = pd.DataFrame(index=data.index)
        
        if 'vix_level' in data.columns:
            # First fill missing values
            vix_data = data['vix_level'].ffill()
            
            market_features['vix_level'] = vix_data
            market_features['vix_daily_change'] = vix_data.pct_change()
            market_features['vix_weekly_avg'] = vix_data.rolling(window=5).mean() #short-term trends 
            market_features['vix_monthly_avg'] = vix_data.rolling(window=21).mean() # identify regime changes
            market_features['vix_volatility'] = vix_data.rolling(window=21).std() # volatility of the VIX
            
            try:
                market_features['vix_regime'] = pd.qcut(
                    vix_data,
                    q=5,
                    labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                    duplicates='drop'
                )
            except Exception as e:
                print(f"Could not create VIX regime categories: {str(e)}")
        
        print("\nMarket Features Summary:")
        print(f"Created {len(market_features.columns)} market features")
        print("Features created:", market_features.columns.tolist())
        
        return market_features

    def engineer_economic_features(self, data):
        """
        Create economic features with improved error handling.
        """
        economic_features = pd.DataFrame(index=data.index)
        
        # Basic economic indicators with forward filling
        for col in ['fed_rate', 'treasury_10y', 'unemployment']:
            if col in data.columns:
                economic_features[col] = data[col].ffill()
        
        # Derived economic features
        if all(col in data.columns for col in ['fed_rate', 'treasury_10y']):
            economic_features['yield_curve'] = (
                data['treasury_10y'].ffill() - data['fed_rate'].ffill()
            )
        
        if 'unemployment' in data.columns:
            unemployment_filled = data['unemployment'].ffill()
            economic_features['unemployment_trend'] = unemployment_filled.diff(periods=90)
        
        print("\nEconomic Features Summary:")
        print(f"Created {len(economic_features.columns)} economic features")
        print("Features created:", economic_features.columns.tolist())
        
        return economic_features

    def create_final_feature_set(self):
        """
        Combine all features with enhanced error checking and reporting.
        """
        # Get raw data
        print("Fetching raw data...")
        raw_data = self.get_all_data()
        
        if raw_data.empty:
            print("Error: No raw data was collected")
            return pd.DataFrame()
        
        # Engineer features
        print("\nEngineering market features...")
        market_features = self.engineer_market_features(raw_data)
        
        print("\nEngineering economic features...")
        economic_features = self.engineer_economic_features(raw_data)
        
        # Combine features
        final_features = pd.concat([market_features, economic_features], axis=1)
        
        # Create lag features
        if 'vix_level' in final_features.columns:
            for period in [1, 5, 21]:
                final_features[f'vix_lag_{period}d'] = final_features['vix_level'].shift(period)
        
        # Clean up missing values
        final_features = final_features.ffill().dropna()
        
        print("\nFinal Feature Set Summary:")
        print(f"Total features created: {len(final_features.columns)}")
        print(f"Total rows of data: {len(final_features)}")
        print("\nFeature names:", final_features.columns.tolist())
        
        return final_features

def main():
    fred_api_key = "bb1fcb27bdf63518da789633f96e8a23"
    data_engineer = MarketDataEngineer(fred_api_key)
    
    # Get engineered features
    features = data_engineer.create_final_feature_set()
    
    if not features.empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export to CSV
        csv_filename = f'engineered_market_data_{timestamp}.csv'
        features.to_csv(csv_filename)
        print(f"\nData exported to {csv_filename}")    
     
    else:
        print("No data to export - please check the error messages above")

if __name__ == "__main__":
    features = main()