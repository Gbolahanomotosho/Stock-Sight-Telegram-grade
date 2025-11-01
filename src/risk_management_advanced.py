# src/risk_management_advanced.py
"""
Professional Risk Management System
Includes: Position sizing, Kelly Criterion, Risk of Ruin, Sharpe optimization
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class RiskParameters:
    """Risk management parameters"""
    account_balance: float = 10000.0
    max_risk_per_trade: float = 0.02  # 2% max risk
    max_daily_loss: float = 0.06  # 6% daily loss limit
    max_drawdown: float = 0.20  # 20% max drawdown
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    target_sharpe: float = 2.0  # Target Sharpe ratio
    win_rate: float = 0.55  # Historical win rate
    avg_win_loss_ratio: float = 1.8  # Average win/loss ratio
    
class AdvancedRiskManager:
    """
    Institutional-grade risk management system
    """
    
    def __init__(self, params: RiskParameters = None):
        self.params = params or RiskParameters()
        self.trade_history = []
        self.daily_pnl = []
        self.current_drawdown = 0.0
        self.peak_balance = self.params.account_balance
        
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float,
        volatility: float,
        regime: Dict
    ) -> Dict:
        """
        Calculate optimal position size using multiple methods
        """
        try:
            # Method 1: Fixed fractional (basic)
            fixed_frac = self._fixed_fractional_sizing(entry_price, stop_loss)
            
            # Method 2: Kelly Criterion (optimal growth)
            kelly_size = self._kelly_criterion_sizing(signal_confidence)
            
            # Method 3: Volatility-adjusted
            vol_adjusted = self._volatility_adjusted_sizing(volatility, regime)
            
            # Method 4: Risk of ruin adjusted
            risk_of_ruin_adj = self._risk_of_ruin_adjustment()
            
            # Method 5: Sharpe-optimized
            sharpe_optimized = self._sharpe_optimized_sizing()
            
            # COMBINE METHODS (weighted average)
            weights = {
                'fixed_frac': 0.25,
                'kelly': 0.20,
                'volatility': 0.25,
                'risk_ruin': 0.15,
                'sharpe': 0.15
            }
            
            combined_size = (
                fixed_frac * weights['fixed_frac'] +
                kelly_size * weights['kelly'] +
                vol_adjusted * weights['volatility'] +
                risk_of_ruin_adj * weights['risk_ruin'] +
                sharpe_optimized * weights['sharpe']
            )
            
            # Apply confidence scaling
            confidence_scaled = combined_size * self._confidence_multiplier(signal_confidence)
            
            # Apply regime scaling
            regime_scaled = confidence_scaled * self._regime_multiplier(regime)
            
            # Apply drawdown scaling
            drawdown_scaled = regime_scaled * self._drawdown_multiplier()
            
            # Apply correlation adjustment (if multiple positions)
            correlation_adjusted = drawdown_scaled * self._correlation_adjustment()
            
            # Final safety checks
            final_size = self._apply_safety_limits(correlation_adjusted, entry_price, stop_loss)
            
            # Calculate lot size for forex
            lot_size = self._calculate_lot_size(final_size, entry_price)
            
            # Risk metrics
            risk_amount = final_size * abs(entry_price - stop_loss) / entry_price
            risk_percent = (risk_amount / self.params.account_balance) * 100
            
            return {
                'position_size_usd': float(final_size),
                'lot_size': float(lot_size),
                'risk_amount': float(risk_amount),
                'risk_percent': float(risk_percent),
                'leverage_used': float(final_size / self.params.account_balance),
                'methods': {
                    'fixed_fractional': float(fixed_frac),
                    'kelly_criterion': float(kelly_size),
                    'volatility_adjusted': float(vol_adjusted),
                    'risk_of_ruin': float(risk_of_ruin_adj),
                    'sharpe_optimized': float(sharpe_optimized)
                },
                'adjustments': {
                    'confidence_mult': float(self._confidence_multiplier(signal_confidence)),
                    'regime_mult': float(self._regime_multiplier(regime)),
                    'drawdown_mult': float(self._drawdown_multiplier()),
                    'correlation_mult': float(self._correlation_adjustment())
                },
                'safety_status': 'ok' if risk_percent <= self.params.max_risk_per_trade * 100 else 'reduced'
            }
            
        except Exception as e:
            print(f"[ERROR] Position sizing calculation failed: {e}")
            return self._get_conservative_position()
    
    def _fixed_fractional_sizing(self, entry_price: float, stop_loss: float) -> float:
        """Fixed fractional position sizing"""
        if entry_price <= 0 or stop_loss <= 0:
            return 0.0
        
        # Risk amount
        risk_amount = self.params.account_balance * self.params.max_risk_per_trade
        
        # Stop loss distance
        sl_distance = abs(entry_price - stop_loss) / entry_price
        
        if sl_distance == 0:
            return 0.0
        
        # Position size
        position_size = risk_amount / sl_distance
        
        return float(position_size)
    
    def _kelly_criterion_sizing(self, confidence: float) -> float:
        """
        Kelly Criterion for optimal position sizing
        f* = (p*b - q) / b
        where p = win probability, q = loss probability, b = win/loss ratio
        """
        try:
            # Adjust win rate based on signal confidence
            adjusted_win_rate = self.params.win_rate * (0.7 + 0.3 * confidence)
            adjusted_win_rate = min(0.75, max(0.45, adjusted_win_rate))  # Clamp
            
            loss_rate = 1.0 - adjusted_win_rate
            win_loss_ratio = self.params.avg_win_loss_ratio
            
            # Kelly formula
            kelly_fraction = (adjusted_win_rate * win_loss_ratio - loss_rate) / win_loss_ratio
            
            # Use fractional Kelly (safer)
            fractional_kelly = max(0.0, kelly_fraction * 0.25)  # 25% of full Kelly
            
            # Convert to position size
            position_size = self.params.account_balance * fractional_kelly
            
            return float(position_size)
            
        except Exception as e:
            return self.params.account_balance * 0.02
    
    def _volatility_adjusted_sizing(self, volatility: float, regime: Dict) -> float:
        """Adjust position size based on volatility"""
        try:
            # Base position
            base_size = self.params.account_balance * 0.02
            
            # Volatility adjustment (inverse relationship)
            vol_percentile = regime.get('volatility_percentile', 50.0)
            vol_mult = 1.0 - (vol_percentile / 100.0) * 0.5
            
            # Additional adjustment for extreme volatility
            if vol_percentile > 80:
                vol_mult *= 0.7
            elif vol_percentile < 20:
                vol_mult *= 1.2
            
            return float(base_size * vol_mult)
            
        except Exception:
            return self.params.account_balance * 0.02
    
    def _risk_of_ruin_adjustment(self) -> float:
        """Adjust based on risk of ruin probability"""
        try:
            # Calculate current drawdown
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - self.params.account_balance) / self.peak_balance
            else:
                current_drawdown = 0.0
            
            # Risk of ruin approximation
            # RoR ≈ ((1-W)/W) ^ (Capital/AverageWin)
            # Simplified: reduce size as we approach max drawdown
            
            drawdown_ratio = current_drawdown / self.params.max_drawdown
            
            if drawdown_ratio > 0.8:
                # Very close to max drawdown - severely reduce
                size_mult = 0.3
            elif drawdown_ratio > 0.6:
                size_mult = 0.5
            elif drawdown_ratio > 0.4:
                size_mult = 0.7
            else:
                size_mult = 1.0
            
            base_size = self.params.account_balance * 0.02
            return float(base_size * size_mult)
            
        except Exception:
            return self.params.account_balance * 0.02
    
    def _sharpe_optimized_sizing(self) -> float:
        """Position sizing optimized for Sharpe ratio"""
        try:
            if len(self.trade_history) < 10:
                return self.params.account_balance * 0.02
            
            # Calculate recent Sharpe ratio
            recent_returns = [t['return_pct'] for t in self.trade_history[-30:]]
            
            mean_return = np.mean(recent_returns)
            std_return = np.std(recent_returns)
            
            if std_return == 0:
                return self.params.account_balance * 0.02
            
            sharpe = (mean_return - self.params.risk_free_rate / 252) / std_return
            
            # Optimal sizing for target Sharpe
            # f* = (μ - rf) / (σ² * target_sharpe)
            if sharpe > 0:
                optimal_fraction = mean_return / (std_return ** 2 * self.params.target_sharpe)
                optimal_fraction = max(0.01, min(0.05, optimal_fraction))
            else:
                optimal_fraction = 0.01
            
            return float(self.params.account_balance * optimal_fraction)
            
        except Exception:
            return self.params.account_balance * 0.02
    
    def _confidence_multiplier(self, confidence: float) -> float:
        """Scale position based on signal confidence"""
        # Non-linear scaling
        if confidence >= 0.8:
            return 1.2
        elif confidence >= 0.65:
            return 1.0
        elif confidence >= 0.5:
            return 0.8
        else:
            return 0.5
    
    def _regime_multiplier(self, regime: Dict) -> float:
        """Scale based on market regime"""
        regime_type = regime.get('regime', 'unknown')
        risk_level = regime.get('risk_level', 'moderate')
        
        # Favorable regimes
        if regime_type in ['strong_bull_trend', 'strong_bear_trend']:
            base_mult = 1.2
        elif regime_type in ['moderate_bull_trend', 'moderate_bear_trend']:
            base_mult = 1.0
        elif regime_type == 'range_bound':
            base_mult = 0.8
        elif regime_type in ['volatile_choppy', 'volatile_trending']:
            base_mult = 0.6
        else:
            base_mult = 0.7
        
        # Risk level adjustment
        risk_adjustments = {
            'very_low': 1.2,
            'low': 1.1,
            'moderate': 1.0,
            'high': 0.8,
            'very_high': 0.5
        }
        
        risk_mult = risk_adjustments.get(risk_level, 1.0)
        
        return base_mult * risk_mult
    
    def _drawdown_multiplier(self) -> float:
        """Scale based on current drawdown"""
        if self.peak_balance <= 0:
            return 1.0
        
        drawdown_pct = (self.peak_balance - self.params.account_balance) / self.peak_balance
        
        if drawdown_pct < 0.05:
            return 1.0
        elif drawdown_pct < 0.10:
            return 0.9
        elif drawdown_pct < 0.15:
            return 0.75
        else:
            return 0.5
    
    def _correlation_adjustment(self) -> float:
        """Adjust for portfolio correlation (placeholder)"""
        # In a full implementation, this would check correlation
        # between current position and existing positions
        return 1.0
    
    def _apply_safety_limits(self, size: float, entry: float, stop_loss: float) -> float:
        """Apply final safety checks"""
        # Maximum position size (e.g., 20% of account)
        max_size = self.params.account_balance * 0.20
        size = min(size, max_size)
        
        # Minimum position size
        min_size = self.params.account_balance * 0.005
        size = max(size, min_size)
        
        # Check risk doesn't exceed limit
        risk = size * abs(entry - stop_loss) / entry
        max_risk = self.params.account_balance * self.params.max_risk_per_trade
        
        if risk > max_risk:
            size = size * (max_risk / risk)
        
        return float(size)
    
    def _calculate_lot_size(self, position_size_usd: float, price: float) -> float:
        """Convert USD position size to forex lot size"""
        # Standard lot = 100,000 units
        # Mini lot = 10,000 units
        # Micro lot = 1,000 units
        
        units = position_size_usd / price
        lots = units / 100000  # Standard lots
        
        # Round to nearest 0.01 lot (micro lot)
        lots = round(lots, 2)
        
        return max(0.01, lots)  # Minimum 0.01 lot
    
    def _get_conservative_position(self) -> Dict:
        """Return conservative default position"""
        conservative_size = self.params.account_balance * 0.01
        return {
            'position_size_usd': float(conservative_size),
            'lot_size': 0.01,
            'risk_amount': float(conservative_size * 0.02),
            'risk_percent': 1.0,
            'leverage_used': 0.1,
            'methods': {},
            'adjustments': {},
            'safety_status': 'conservative_default'
        }
    
    def calculate_stop_loss_take_profit(
        self,
        entry_price: float,
        signal: str,
        atr: float,
        volatility_percentile: float,
        regime: Dict
    ) -> Tuple[float, float, float]:
        """
        Calculate dynamic stop loss and take profit levels
        """
        try:
            # Base ATR multipliers
            base_sl_mult = 1.5
            base_tp_mult = 3.0
            
            # Adjust for volatility
            if volatility_percentile > 80:
                sl_mult = base_sl_mult * 1.3
                tp_mult = base_tp_mult * 0.9
            elif volatility_percentile > 60:
                sl_mult = base_sl_mult * 1.1
                tp_mult = base_tp_mult * 0.95
            elif volatility_percentile < 20:
                sl_mult = base_sl_mult * 0.8
                tp_mult = base_tp_mult * 1.1
            else:
                sl_mult = base_sl_mult
                tp_mult = base_tp_mult
            
            # Adjust for regime
            regime_type = regime.get('regime', 'unknown')
            if 'strong' in regime_type:
                tp_mult *= 1.2
            elif 'volatile' in regime_type:
                sl_mult *= 1.2
                tp_mult *= 0.9
            
            # Calculate levels
            sl_distance = atr * sl_mult
            tp_distance = atr * tp_mult
            
            if signal == 'BUY':
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:  # SELL
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
            
            # Risk-reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            risk_reward = reward / risk if risk > 0 else 0
            
            return float(stop_loss), float(take_profit), float(risk_reward)
            
        except Exception as e:
            print(f"[ERROR] SL/TP calculation failed: {e}")
            # Conservative defaults
            if signal == 'BUY':
                return float(entry_price * 0.98), float(entry_price * 1.04), 2.0
            else:
                return float(entry_price * 1.02), float(entry_price * 0.96), 2.0
    
    def update_trade(self, trade_result: Dict):
        """Update trade history and metrics"""
        self.trade_history.append(trade_result)
        
        # Update balance
        pnl = trade_result.get('pnl', 0.0)
        self.params.account_balance += pnl
        
        # Update peak balance
        if self.params.account_balance > self.peak_balance:
            self.peak_balance = self.params.account_balance
        
        # Track daily PnL
        self.daily_pnl.append({
            'date': trade_result.get('exit_time'),
            'pnl': pnl
        })
        
        # Update drawdown
        self.current_drawdown = (self.peak_balance - self.params.account_balance) / self.peak_balance
    
    def should_trade_today(self) -> Tuple[bool, str]:
        """Check if trading should continue today"""
        if not self.daily_pnl:
            return True, "ok"
        
        # Calculate today's PnL
        today_trades = [t for t in self.daily_pnl[-20:]]  # Last 20 trades as "today"
        today_pnl = sum(t['pnl'] for t in today_trades)
        today_pnl_pct = (today_pnl / self.peak_balance) * 100
        
        # Check daily loss limit
        if today_pnl_pct < -self.params.max_daily_loss * 100:
            return False, "daily_loss_limit_reached"
        
        # Check max drawdown
        if self.current_drawdown > self.params.max_drawdown:
            return False, "max_drawdown_reached"
        
        # Check consecutive losses
        recent_trades = self.trade_history[-10:]
        if len(recent_trades) >= 10:
            recent_losses = sum(1 for t in recent_trades if t.get('pnl', 0) < 0)
            if recent_losses >= 7:  # 7 out of 10 losses
                return False, "excessive_consecutive_losses"
        
        return True, "ok"
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.trade_history:
            return {}
        
        returns = [t['return_pct'] for t in self.trade_history]
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        total_return = ((self.params.account_balance - 10000) / 10000) * 100
        
        metrics = {
            'total_trades': len(self.trade_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(returns) * 100 if returns else 0,
            'total_return_pct': float(total_return),
            'average_win': float(np.mean(wins)) if wins else 0,
            'average_loss': float(np.mean(losses)) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else 0,
            'sharpe_ratio': self._calculate_sharpe(returns),
            'sortino_ratio': self._calculate_sortino(returns),
            'max_drawdown_pct': float(self.current_drawdown * 100),
            'current_balance': float(self.params.account_balance),
            'peak_balance': float(self.peak_balance)
        }
        
        return metrics
    
    def _calculate_sharpe(self, returns: list) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe (assuming daily returns)
        sharpe = (mean_return - self.params.risk_free_rate / 252) / std_return * np.sqrt(252)
        
        return float(sharpe)
    
    def _calculate_sortino(self, returns: list) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        downside_returns = [r for r in returns if r < 0]
        
        if not downside_returns:
            return 0.0
        
        downside_std = np.std(downside_returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (mean_return - self.params.risk_free_rate / 252) / downside_std * np.sqrt(252)
        
        return float(sortino)
