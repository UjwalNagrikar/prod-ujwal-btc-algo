"""
Trend Following Strategy - Professional Quantitative Implementation
Strategy: Dual Moving Average Crossover with ATR-based Position Sizing
Exchange: OKX (simulated data)
Author: Quant Developer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os

class TrendFollowingStrategy:
    """
    A professional trend-following strategy using:
    - Dual Moving Average (Fast: 20, Slow: 50) for trend identification
    - ATR (Average True Range) for volatility-based position sizing
    - Risk management with stop-loss and take-profit levels
    """
    
    def __init__(self, initial_capital=10000, commission=0.001, leverage=3):
        """
        Initialize strategy parameters
        
        Args:
            initial_capital: Starting capital in USD
            commission: Trading fee (0.1% = 0.001)
            leverage: Leverage multiplier
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.leverage = leverage
        
        # Strategy parameters
        self.fast_ma = 20
        self.slow_ma = 50
        self.atr_period = 14
        self.risk_per_trade = 0.05  # 2% risk per trade
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 3.0
        
        # Tracking variables
        self.trades = []
        self.equity_curve = []
        
    def calculate_atr(self, df):
        """Calculate Average True Range for volatility measurement"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=self.atr_period).mean()
        
        return atr
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators"""
        # Moving Averages
        df['ma_fast'] = df['close'].rolling(window=self.fast_ma).mean()
        df['ma_slow'] = df['close'].rolling(window=self.slow_ma).mean()
        
        # ATR for position sizing and risk management
        df['atr'] = self.calculate_atr(df)
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['ma_fast'] > df['ma_slow'], 'signal'] = 1  # Bullish
        df.loc[df['ma_fast'] < df['ma_slow'], 'signal'] = -1  # Bearish
        
        # Detect crossovers
        df['position'] = df['signal'].diff()
        
        return df
    
    def calculate_position_size(self, capital, price, atr):
        """
        Calculate position size based on ATR and risk management
        
        Risk = (Entry Price - Stop Loss) * Position Size
        Position Size = (Account Risk) / (Entry Price - Stop Loss)
        """
        risk_amount = capital * self.risk_per_trade
        stop_distance = atr * self.stop_loss_atr_multiplier
        
        if stop_distance > 0:
            position_size = risk_amount / stop_distance
            position_value = position_size * price
            
            # Apply leverage
            max_position_value = capital * self.leverage
            if position_value > max_position_value:
                position_size = max_position_value / price
        else:
            position_size = 0
            
        return position_size
    
    def backtest(self, df, print_trades=True):
        """
        Execute backtest on historical data
        
        Args:
            df: DataFrame with OHLCV data
            print_trades: If True, print each trade as it's executed
            
        Returns:
            DataFrame with results and performance metrics
        """
        df = self.calculate_indicators(df)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        position_size = 0
        trade_number = 0
        
        if print_trades:
            print("\n" + "="*100)
            print("TRADE EXECUTION LOG")
            print("="*100)
            print(f"{'#':<5} {'Date':<20} {'Market':<10} {'Direction':<10} {'Entry':<12} {'Exit':<12} {'Contracts':<10} {'P/L':<12} {'Return %':<10}")
            print("-"*100)
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Skip if not enough data for indicators
            if pd.isna(row['ma_slow']) or pd.isna(row['atr']):
                self.equity_curve.append(capital)
                continue
            
            current_price = row['close']
            current_atr = row['atr']
            
            # Check stop loss and take profit
            if position != 0:
                if position > 0:  # Long position
                    if current_price <= stop_loss or current_price >= take_profit:
                        # Close position
                        pnl = (current_price - entry_price) * position_size
                        commission_cost = current_price * position_size * self.commission
                        capital += pnl - commission_cost
                        
                        exit_reason = "TP" if current_price >= take_profit else "SL"
                        trade_number += 1
                        
                        trade_data = {
                            'trade_num': trade_number,
                            'entry_date': df.iloc[entry_idx]['timestamp'],
                            'exit_date': row['timestamp'],
                            'type': 'LONG',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'size': position_size,
                            'pnl': pnl - commission_cost,
                            'return_pct': ((current_price - entry_price) / entry_price) * 100,
                            'exit_reason': exit_reason
                        }
                        
                        self.trades.append(trade_data)
                        
                        if print_trades:
                            self._print_trade(trade_data)
                        
                        position = 0
                        position_size = 0
                
                elif position < 0:  # Short position
                    if current_price >= stop_loss or current_price <= take_profit:
                        # Close position
                        pnl = (entry_price - current_price) * abs(position_size)
                        commission_cost = current_price * abs(position_size) * self.commission
                        capital += pnl - commission_cost
                        
                        exit_reason = "TP" if current_price <= take_profit else "SL"
                        trade_number += 1
                        
                        trade_data = {
                            'trade_num': trade_number,
                            'entry_date': df.iloc[entry_idx]['timestamp'],
                            'exit_date': row['timestamp'],
                            'type': 'SHORT',
                            'entry_price': entry_price,
                            'exit_price': current_price,
                            'size': abs(position_size),
                            'pnl': pnl - commission_cost,
                            'return_pct': ((entry_price - current_price) / entry_price) * 100,
                            'exit_reason': exit_reason
                        }
                        
                        self.trades.append(trade_data)
                        
                        if print_trades:
                            self._print_trade(trade_data)
                        
                        position = 0
                        position_size = 0
            
            # Entry signals
            if position == 0 and row['position'] == 2:  # Bullish crossover
                position = 1
                entry_price = current_price
                entry_idx = i
                position_size = self.calculate_position_size(capital, current_price, current_atr)
                
                # Set stop loss and take profit
                stop_loss = entry_price - (current_atr * self.stop_loss_atr_multiplier)
                take_profit = entry_price + (current_atr * self.take_profit_atr_multiplier)
                
                # Deduct entry commission
                commission_cost = current_price * position_size * self.commission
                capital -= commission_cost
            
            elif position == 0 and row['position'] == -2:  # Bearish crossover
                position = -1
                entry_price = current_price
                entry_idx = i
                position_size = self.calculate_position_size(capital, current_price, current_atr)
                
                # Set stop loss and take profit
                stop_loss = entry_price + (current_atr * self.stop_loss_atr_multiplier)
                take_profit = entry_price - (current_atr * self.take_profit_atr_multiplier)
                
                # Deduct entry commission
                commission_cost = current_price * position_size * self.commission
                capital -= commission_cost
            
            # Track equity
            unrealized_pnl = 0
            if position > 0:
                unrealized_pnl = (current_price - entry_price) * position_size
            elif position < 0:
                unrealized_pnl = (entry_price - current_price) * abs(position_size)
            
            self.equity_curve.append(capital + unrealized_pnl)
        
        # Close any remaining position
        if position != 0:
            current_price = df.iloc[-1]['close']
            if position > 0:
                pnl = (current_price - entry_price) * position_size
            else:
                pnl = (entry_price - current_price) * abs(position_size)
            
            commission_cost = current_price * abs(position_size) * self.commission
            capital += pnl - commission_cost
            
            trade_number += 1
            
            trade_data = {
                'trade_num': trade_number,
                'entry_date': df.iloc[entry_idx]['timestamp'],
                'exit_date': df.iloc[-1]['timestamp'],
                'type': 'LONG' if position > 0 else 'SHORT',
                'entry_price': entry_price,
                'exit_price': current_price,
                'size': abs(position_size),
                'pnl': pnl - commission_cost,
                'return_pct': ((current_price - entry_price) / entry_price * (1 if position > 0 else -1)) * 100,
                'exit_reason': 'EOD'
            }
            
            self.trades.append(trade_data)
            
            if print_trades:
                self._print_trade(trade_data)
        
        if print_trades:
            print("-"*100)
            print(f"Total Trades: {trade_number}")
            print("="*100 + "\n")
        
        df['equity'] = self.equity_curve
        
        return df
    
    def _print_trade(self, trade):
        """Helper function to print a single trade"""
        entry_date = trade['entry_date'].strftime('%Y-%m-%d %H:%M')
        pnl_color = '+' if trade['pnl'] >= 0 else ''
        
        print(f"{trade['trade_num']:<5} "
              f"{entry_date:<20} "
              f"{'BTC/USDT':<10} "
              f"{trade['type']:<10} "
              f"${trade['entry_price']:<11.2f} "
              f"${trade['exit_price']:<11.2f} "
              f"{trade['size']:<10.4f} "
              f"{pnl_color}${trade['pnl']:<11.2f} "
              f"{pnl_color}{trade['return_pct']:<9.2f}%")
    
    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {
                'error': 'No trades executed',
                'total_trades': 0
            }
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Profit factor
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Returns
        final_capital = self.equity_curve[-1]
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Drawdown
        equity_series = pd.Series(self.equity_curve)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # Sharpe Ratio (simplified - assuming daily returns)
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(365)) if returns.std() > 0 else 0
        
        # Best and worst trades
        best_trade = trades_df.loc[trades_df['pnl'].idxmax()]
        worst_trade = trades_df.loc[trades_df['pnl'].idxmin()]
        
        # Consecutive wins/losses
        trades_df['is_win'] = trades_df['pnl'] > 0
        trades_df['streak'] = (trades_df['is_win'] != trades_df['is_win'].shift()).cumsum()
        win_streaks = trades_df[trades_df['is_win']].groupby('streak').size()
        loss_streaks = trades_df[~trades_df['is_win']].groupby('streak').size()
        
        max_win_streak = win_streaks.max() if len(win_streaks) > 0 else 0
        max_loss_streak = loss_streaks.max() if len(loss_streaks) > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown_pct': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'best_trade': best_trade['pnl'],
            'worst_trade': worst_trade['pnl'],
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }
        
        return metrics
    
    def plot_results(self, df, save_path='strategy_results.png'):
        """Generate comprehensive visualization of backtest results"""
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        # Plot 1: Price and Moving Averages
        ax1 = axes[0]
        ax1.plot(df.index, df['close'], label='Close Price', linewidth=1.5, color='black')
        ax1.plot(df.index, df['ma_fast'], label=f'MA{self.fast_ma}', linewidth=1, color='blue', alpha=0.7)
        ax1.plot(df.index, df['ma_slow'], label=f'MA{self.slow_ma}', linewidth=1, color='red', alpha=0.7)
        
        # Mark entry points
        buys = df[df['position'] == 2]
        sells = df[df['position'] == -2]
        ax1.scatter(buys.index, buys['close'], marker='^', color='green', s=100, label='Buy Signal', zorder=5)
        ax1.scatter(sells.index, sells['close'], marker='v', color='red', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Trend Following Strategy - Price & Signals', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price (USD)', fontsize=11)
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Equity Curve
        ax2 = axes[1]
        ax2.plot(df.index, df['equity'], label='Portfolio Value', linewidth=2, color='darkgreen')
        ax2.axhline(y=self.initial_capital, color='gray', linestyle='--', label='Initial Capital', linewidth=1)
        ax2.fill_between(df.index, self.initial_capital, df['equity'], 
                         where=(df['equity'] >= self.initial_capital), alpha=0.3, color='green')
        ax2.fill_between(df.index, self.initial_capital, df['equity'], 
                         where=(df['equity'] < self.initial_capital), alpha=0.3, color='red')
        
        ax2.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Portfolio Value (USD)', fontsize=11)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Drawdown
        ax3 = axes[2]
        equity_series = pd.Series(self.equity_curve, index=df.index)
        running_max = equity_series.cummax()
        drawdown = (equity_series - running_max) / running_max * 100
        
        ax3.fill_between(df.index, 0, drawdown, color='red', alpha=0.3)
        ax3.plot(df.index, drawdown, color='darkred', linewidth=1.5)
        ax3.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)', fontsize=11)
        ax3.set_xlabel('Date', fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Chart saved to: {save_path}")
        
        return fig


def generate_sample_data(symbol='BTC/USDT', days=365):
    """
    Generate realistic cryptocurrency price data for backtesting
    Simulates 1 year of hourly OHLCV data from OKX
    """
    np.random.seed(42)
    
    # Generate hourly timestamps for 1 year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, end=end_date, freq='h')
    
    # Simulate realistic BTC price movement
    initial_price = 45000
    n_periods = len(timestamps)
    
    # Generate returns with drift and volatility
    drift = 0.0001  # Slight upward drift
    volatility = 0.02
    returns = np.random.normal(drift, volatility, n_periods)
    
    # Add some trending behavior
    trend = np.sin(np.linspace(0, 4*np.pi, n_periods)) * 0.001
    returns += trend
    
    # Calculate prices
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = []
    for i, (timestamp, close) in enumerate(zip(timestamps, close_prices)):
        # Generate realistic OHLC
        volatility_factor = np.random.uniform(0.005, 0.015)
        high = close * (1 + volatility_factor)
        low = close * (1 - volatility_factor)
        open_price = np.random.uniform(low, high)
        
        # Ensure OHLC relationship is valid
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    return df


def main():
    """Main execution function"""
    print("\n" + "=" * 70)
    print("TREND FOLLOWING STRATEGY BACKTEST")
    print("Exchange: OKX (Simulated Data)")
    print("Symbol: BTC/USDT")
    print("Timeframe: 1 Hour")
    print("Period: 1 Year")
    print("=" * 70)
    
    # Generate sample data (simulating OKX exchange data)
    print("\n📊 Loading market data...")
    df = generate_sample_data(symbol='BTC/USDT', days=365)
    print(f"✓ Data loaded: {len(df)} candles")
    print(f"  From: {df['timestamp'].min()}")
    print(f"  To:   {df['timestamp'].max()}")
    
    # Initialize strategy
    strategy = TrendFollowingStrategy(
        initial_capital=10000,
        commission=0.001,  # 0.1% (typical OKX fee)
        leverage=1
    )
    
    print("\n⚙️  Strategy Parameters:")
    print(f"  • Fast MA: {strategy.fast_ma}")
    print(f"  • Slow MA: {strategy.slow_ma}")
    print(f"  • ATR Period: {strategy.atr_period}")
    print(f"  • Risk per Trade: {strategy.risk_per_trade * 100}%")
    print(f"  • Initial Capital: ${strategy.initial_capital:,.2f}")
    print(f"  • Commission: {strategy.commission * 100}%")
    
    # Run backtest with trade printing enabled
    df = strategy.backtest(df, print_trades=True)
    
    # Calculate metrics
    metrics = strategy.calculate_metrics()
    
    # Display results summary
    print("\n" + "=" * 70)
    print("📈 BACKTEST SUMMARY")
    print("=" * 70)
    print()
    print("TRADING METRICS:")
    print(f"  Total Trades:        {metrics['total_trades']}")
    print(f"  Winning Trades:      {metrics['winning_trades']}")
    print(f"  Losing Trades:       {metrics['losing_trades']}")
    print(f"  Win Rate:            {metrics['win_rate']:.2f}%")
    print(f"  Max Win Streak:      {metrics['max_win_streak']}")
    print(f"  Max Loss Streak:     {metrics['max_loss_streak']}")
    print()
    print("PERFORMANCE METRICS:")
    print(f"  Initial Capital:     ${metrics['initial_capital']:,.2f}")
    print(f"  Final Capital:       ${metrics['final_capital']:,.2f}")
    print(f"  Total PnL:           ${metrics['total_pnl']:,.2f}")
    print(f"  Total Return:        {metrics['total_return_pct']:.2f}%")
    print(f"  Best Trade:          ${metrics['best_trade']:,.2f}")
    print(f"  Worst Trade:         ${metrics['worst_trade']:,.2f}")
    print(f"  Average Win:         ${metrics['avg_win']:,.2f}")
    print(f"  Average Loss:        ${metrics['avg_loss']:,.2f}")
    print(f"  Profit Factor:       {metrics['profit_factor']:.2f}")
    print(f"  Max Drawdown:        {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.2f}")
    print()
    
    # Save detailed trade log
    if strategy.trades:
        trades_df = pd.DataFrame(strategy.trades)
        trades_df.to_csv('trade_log_detailed.csv', index=False)
        print("✓ Detailed trade log saved to: trade_log_detailed.csv")
    
    # Generate visualization
    print("📊 Generating charts...")
    strategy.plot_results(df)
    
    # Save metrics to JSON
    with open('backtest_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4, default=str)
    print("✓ Metrics saved to: backtest_metrics.json")
    
    print()
    print("=" * 70)
    print("✅ Backtest completed successfully!")
    print("=" * 70 + "\n")
    
    return strategy, df, metrics


if __name__ == "__main__":
    strategy, df, metrics = main()
