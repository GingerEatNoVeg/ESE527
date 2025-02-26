import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os 

class MarketDataAnalyzer:
    def __init__(self, csv_path):
        """
        Initializes market data analyzer with proper date handling and data validation.
        The analyzer will process  engineered data for both statistical analysis and modeling.   
        """
        # Read the data and handle the specific date format (MM/DD/YY)
        self.data = pd.read_csv(csv_path)
        self.data['Date'] = pd.to_datetime(self.data.iloc[:, 0], format='%m/%d/%y')
        self.data.set_index('Date', inplace=True)
        
        # Identify column types for appropriate handling
        self.numerical_columns = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.categorical_columns = self.data.select_dtypes(include=['object']).columns
        
        print(f"Data loaded successfully. Time range: {self.data.index.min()} to {self.data.index.max()}")

    def _calculate_rsi(self, prices, period=14):
        """
        Calculates the Relative Strength Index (RSI), which gives the speed and magnitude of recent price changes
          to measure momentum and identify overbought/oversold conditions.
        """
        delta = prices.diff() # day over day price changes
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss #see for overbought/oversold conditions
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """
        Calculates Bollinger Bands to identify volatility levels and potential price reversals.
        """
        middle_band = prices.rolling(window=window).mean() #value around which prices fluctuate
        rolling_std = prices.rolling(window=window).std()
        
        upper_band = middle_band + (rolling_std * num_std) #trespassing upper band indicates overbought conditions
        lower_band = middle_band - (rolling_std * num_std)
        
        return upper_band, lower_band

    def _calculate_momentum(self, prices, period=14):
        """
        Calculates momentum indicator to measure the rate of change in prices.
        """
        return prices.diff(period)

    def _add_technical_indicators(self, df):
        """
        Adds a comprehensive set of technical indicators for volatility prediction.
        These indicators help capture different aspects of market behavior.
        """
        # RSI calculation
        df['RSI'] = self._calculate_rsi(df['vix_level']) #high RSI indicates peak fear conditions
        
        # Bollinger Bands
        df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['vix_level'])
        
        # Moving Averages
        df['MA_5'] = df['vix_level'].rolling(window=5).mean() #moving average of 5 days
        df['MA_21'] = df['vix_level'].rolling(window=21).mean()
        
        # Momentum indicators
        df['momentum_5d'] = self._calculate_momentum(df['vix_level'], period=5) #acceleration of market fear
        df['momentum_21d'] = self._calculate_momentum(df['vix_level'], period=21)
        
        # Volatility indicators
        df['volatility_21d'] = df['vix_level'].rolling(window=21).std()
        
        # Ratio indicators
        df['MA_ratio'] = df['MA_5'] / df['MA_21'] #short term vs long term trends
        
        print("\nTechnical indicators added:")
        print("- RSI (Relative Strength Index)")
        print("- Bollinger Bands (Upper and Lower)")
        print("- Moving Averages (5-day and 21-day)")
        print("- Momentum (5-day and 21-day)")
        print("- 21-day Volatility")
        print("- Moving Average Ratio")
        
        return df

    def _calculate_volatility_regime(self, series):
        """
        Implements regime detection using rolling volatility to identify market states.
        """
        rolling_vol = series.rolling(window=21).std()
        regimes = pd.qcut(
            rolling_vol, 
            q=3, 
            labels=['Low', 'Medium', 'High'],
            duplicates='drop'
        )
        return regimes

    def perform_statistical_analysis(self):
        """
        Performs comprehensive statistical analysis including time series decomposition,
        volatility clustering analysis, and correlation studies.
        """
        analysis_results = {}
        
        # Calculate basic statistics for numerical features
        analysis_results['basic_stats'] = self.data[self.numerical_columns].describe()
        
        # Analyze VIX specifically if available
        if 'vix_level' in self.data.columns:
            # Time series decomposition
            decomposition = seasonal_decompose(
                self.data['vix_level'].fillna(method='ffill'), 
                period=21
            )
            
            # Test for volatility clustering
            arch_effects = acorr_ljungbox(
                self.data['vix_level'].dropna(),
                lags=21,
                return_df=True
            )
            
            analysis_results['vix_decomposition'] = decomposition
            analysis_results['arch_effects'] = arch_effects
            
            # Create visualization
            self._plot_vix_analysis(decomposition)

        # Calculate and visualize correlations
        correlation = self.data[self.numerical_columns].corr()
        analysis_results['correlation'] = correlation
        self._plot_correlation_matrix(correlation)
        
        return analysis_results
    def _handle_missing_values(self, df):
        """
        Handles missing values using appropriate methods for different types of features.
        
        For financial data, we need to be careful about how we handle missing values:
        - Technical indicators often have missing values at the start due to calculation windows
        - Forward-looking targets will have missing values at the end of the series
        - Economic indicators might have gaps due to reporting frequencies
        """
        original_shape = df.shape
        
        # 1. Handle technical indicators (forward fill with limited window)
        technical_columns = ['RSI', 'BB_upper', 'BB_lower', 'MA_5', 'MA_21', 
                            'momentum_5d', 'momentum_21d', 'volatility_21d', 'MA_ratio']
        for col in technical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill', limit=5)
        
        # 2. Handle economic indicators (forward fill for longer periods)
        economic_columns = ['yield_curve', 'unemployment', 'fed_rate', 'treasury_10y']
        for col in economic_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # 3. Handle target variables (we'll keep these NaN as they're at the end of series)
        target_columns = [col for col in df.columns if col.startswith('target_vix_')]
        
        # 4. Remove any remaining rows with missing values in key features
        key_features = ['vix_level', 'RSI', 'MA_ratio']  # Add other critical features here
        df = df.dropna(subset=key_features)
        
        print(f"\nMissing value handling summary:")
        print(f"Original shape: {original_shape}")
        print(f"Shape after handling missing values: {df.shape}")
        print(f"Rows removed: {original_shape[0] - df.shape[0]}")
        
        return df


    def prepare_modeling_data(self):
        """
        Prepares the final dataset for modeling with comprehensive missing value handling.
        """
        modeling_data = self.data.copy()
        
        # First, let's analyze where our missing values are coming from
        print("\nAnalyzing missing values before processing:")
        print(modeling_data.isnull().sum())
        
        # Add technical indicators
        modeling_data = self._add_technical_indicators(modeling_data)
        
        print("\nAnalyzing missing values after adding technical indicators:")
        print(modeling_data.isnull().sum())
        
        # Create target variables for different prediction horizons
        for horizon in [1, 5, 21]:
            modeling_data[f'target_vix_{horizon}d'] = modeling_data['vix_level'].shift(-horizon)
        
        # Add volatility regime indicators
        modeling_data['volatility_regime'] = self._calculate_volatility_regime(
            modeling_data['vix_level']
        )
        
        # Create interaction features
        if all(col in modeling_data.columns for col in ['yield_curve', 'unemployment']):
            modeling_data['vix_yield_interaction'] = (
                modeling_data['vix_level'] * modeling_data['yield_curve']
            )
            modeling_data['vix_unemployment_interaction'] = (
                modeling_data['vix_level'] * modeling_data['unemployment']
            )
        
        # Handle missing values appropriately
        modeling_data = self._handle_missing_values(modeling_data)
        
        # Scale numerical features
        scaler = StandardScaler()
        modeling_data[self.numerical_columns] = scaler.fit_transform(
            modeling_data[self.numerical_columns]
        )
        
        print("\nFinal data shape:", modeling_data.shape)
        print("Missing values in final dataset:")
        print(modeling_data.isnull().sum())
        
        return modeling_data


    def _plot_vix_analysis(self, decomposition):
        """Creates visualization of VIX analysis components."""
        plt.figure(figsize=(15, 12))
        decomposition.plot()
        plt.tight_layout()
        plt.savefig('vix_analysis.png')
        plt.close()

    def _plot_correlation_matrix(self, correlation):
        """Creates visualization of feature correlations."""
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png')
        plt.close()

def main():
    """
    Executes the complete analysis pipeline and prepares modeling data.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    analyzer = MarketDataAnalyzer('engineered_market_data.csv')
    
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    analysis_results = analyzer.perform_statistical_analysis()
    
    # Prepare modeling data
    print("\nPreparing data for modeling...")
    modeling_data = analyzer.prepare_modeling_data()
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save modeling data
    modeling_data.to_csv(f'modeling_ready_data_{timestamp}.csv')
    print(f"\nModeling data saved to modeling_ready_data_{timestamp}.csv")
    
    # Generate analysis report
    report = f"""
    Market Volatility Analysis Report
    Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    1. Data Overview
    ---------------
    Time Period: {modeling_data.index.min()} to {modeling_data.index.max()}
    Total Observations: {len(modeling_data)}
    
    2. Basic Statistics
    ------------------
    {analysis_results['basic_stats']}
    
    3. Feature Engineering Summary
    ----------------------------
    Total Features: {len(modeling_data.columns)}
    Added Technical Indicators: RSI, Bollinger Bands, Momentum
    Added Interaction Features: VIX-Yield, VIX-Unemployment
    
    Note: Visualizations saved as separate PNG files
    """
    
    # Save report
    with open(f'analysis_report_{timestamp}.txt', 'w') as f:
        f.write(report)
    
    print(f"\nAnalysis report saved to analysis_report_{timestamp}.txt")
    print("\nAnalysis pipeline complete!")
    
    return modeling_data, analysis_results

if __name__ == "__main__":
    modeling_data, analysis_results = main()