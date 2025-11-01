# src/market_regime_advanced.py
"""
Advanced Market Regime Detection System
Uses multiple indicators, volatility analysis, and ML clustering
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedMarketRegimeDetector:
    """
    Sophisticated market regime detection using multiple methods:
    - Trend strength analysis
    - Volatility clustering
    - Volume analysis
    - Market microstructure
    - Correlation analysis
    """
    
    def __init__(self):
        self.regime_model = None
        self.scaler = StandardScaler()
        self.regime_history = []
        
    def detect_regime(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Comprehensive regime detection with confidence scores
        """
        try:
            if len(df) < lookback:
                lookback = len(df)
            
            recent_data = df.tail(lookback).copy()
            
            # 1. TREND ANALYSIS
            trend_metrics = self._analyze_trend(recent_data)
            
            # 2. VOLATILITY ANALYSIS
            vol_metrics = self._analyze_volatility(recent_data)
            
            # 3. MOMENTUM ANALYSIS
            momentum_metrics = self._analyze_momentum(recent_data)
            
            # 4. VOLUME ANALYSIS
            volume_metrics = self._analyze_volume(recent_data)
            
            # 5. MARKET MICROSTRUCTURE
            microstructure_metrics = self._analyze_microstructure(recent_data)
            
            # 6. CORRELATION ANALYSIS
            correlation_metrics = self._analyze_correlations(recent_data)
            
            # COMBINE ALL METRICS
            regime_score = self._calculate_regime_score(
                trend_metrics, vol_metrics, momentum_metrics,
                volume_metrics, microstructure_metrics, correlation_metrics
            )
            
            # CLASSIFY REGIME
            regime_type = self._classify_regime(regime_score)
            
            # CALCULATE CONFIDENCE
            confidence = self._calculate_confidence(regime_score)
            
            # MARKET PHASE DETECTION
            market_phase = self._detect_market_phase(recent_data)
            
            # VOLATILITY STATE
            vol_state = self._classify_volatility_state(vol_metrics)
            
            result = {
                'regime': regime_type,
                'confidence': confidence,
                'market_phase': market_phase,
                'volatility_state': vol_state,
                'trend_strength': trend_metrics['strength'],
                'volatility_percentile': vol_metrics['percentile'],
                'momentum_score': momentum_metrics['score'],
                'volume_profile': volume_metrics['profile'],
                'risk_level': self._assess_risk_level(regime_score),
                'recommended_leverage': self._calculate_optimal_leverage(regime_score, vol_metrics),
                'metrics': regime_score
            }
            
            self.regime_history.append(result)
            if len(self.regime_history) > 1000:
                self.regime_history.pop(0)
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Regime detection failed: {e}")
            return self._get_default_regime()
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Comprehensive trend analysis"""
        try:
            close = df['y'].values
            
            # Multiple moving averages
            ma_5 = pd.Series(close).rolling(5).mean().values
            ma_20 = pd.Series(close).rolling(20).mean().values
            ma_50 = pd.Series(close).rolling(50).mean().values if len(close) >= 50 else ma_20
            
            # Trend direction
            short_trend = np.mean(close[-5:]) - np.mean(close[-10:-5]) if len(close) >= 10 else 0
            long_trend = np.mean(close[-20:]) - np.mean(close[-40:-20]) if len(close) >= 40 else short_trend
            
            # ADX (Average Directional Index) approximation
            high = df['High'].values if 'High' in df.columns else close
            low = df['Low'].values if 'Low' in df.columns else close
            
            plus_dm = np.maximum(high[1:] - high[:-1], 0)
            minus_dm = np.maximum(low[:-1] - low[1:], 0)
            tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
            tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
            
            plus_di = 100 * pd.Series(plus_dm).rolling(14).mean() / pd.Series(tr).rolling(14).mean()
            minus_di = 100 * pd.Series(minus_dm).rolling(14).mean() / pd.Series(tr).rolling(14).mean()
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().fillna(0).values[-1]
            
            # Trend strength (0-100)
            trend_strength = min(100, max(0, adx))
            
            # Trend consistency
            ma_alignment = 0
            if ma_5[-1] > ma_20[-1] > ma_50[-1]:
                ma_alignment = 1  # Bullish alignment
            elif ma_5[-1] < ma_20[-1] < ma_50[-1]:
                ma_alignment = -1  # Bearish alignment
            
            return {
                'direction': 'bullish' if short_trend > 0 else 'bearish',
                'strength': float(trend_strength),
                'consistency': float(ma_alignment),
                'short_trend': float(short_trend),
                'long_trend': float(long_trend),
                'adx': float(adx)
            }
            
        except Exception as e:
            print(f"[WARN] Trend analysis error: {e}")
            return {'direction': 'neutral', 'strength': 25.0, 'consistency': 0.0, 
                    'short_trend': 0.0, 'long_trend': 0.0, 'adx': 25.0}
    
    def _analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """Advanced volatility analysis"""
        try:
            close = df['y'].values
            
            # Returns
            returns = np.diff(close) / close[:-1]
            
            # Historical volatility (annualized)
            historical_vol = np.std(returns) * np.sqrt(252) * 100
            
            # Recent volatility
            recent_vol = np.std(returns[-20:]) * np.sqrt(252) * 100 if len(returns) >= 20 else historical_vol
            
            # Volatility percentile
            vol_series = pd.Series(returns).rolling(20).std() * np.sqrt(252) * 100
            current_vol_percentile = (vol_series.rank(pct=True).iloc[-1]) * 100 if len(vol_series) > 0 else 50.0
            
            # Parkinson volatility (high-low range based)
            if 'High' in df.columns and 'Low' in df.columns:
                high = df['High'].values
                low = df['Low'].values
                parkinson_vol = np.sqrt(np.mean(np.log(high/low)**2) / (4 * np.log(2))) * np.sqrt(252) * 100
            else:
                parkinson_vol = historical_vol
            
            # Volatility regime
            if current_vol_percentile < 30:
                regime = 'low'
            elif current_vol_percentile < 70:
                regime = 'medium'
            else:
                regime = 'high'
            
            return {
                'historical': float(historical_vol),
                'recent': float(recent_vol),
                'percentile': float(current_vol_percentile),
                'parkinson': float(parkinson_vol),
                'regime': regime,
                'ratio': float(recent_vol / (historical_vol + 1e-6))
            }
            
        except Exception as e:
            print(f"[WARN] Volatility analysis error: {e}")
            return {'historical': 15.0, 'recent': 15.0, 'percentile': 50.0,
                    'parkinson': 15.0, 'regime': 'medium', 'ratio': 1.0}
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict:
        """Momentum and strength indicators"""
        try:
            close = df['y'].values
            
            # RSI
            rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50.0
            
            # Rate of Change
            roc_10 = ((close[-1] - close[-10]) / close[-10] * 100) if len(close) >= 10 else 0.0
            roc_20 = ((close[-1] - close[-20]) / close[-20] * 100) if len(close) >= 20 else roc_10
            
            # MACD approximation
            ema_12 = pd.Series(close).ewm(span=12).mean().values
            ema_26 = pd.Series(close).ewm(span=26).mean().values
            macd = ema_12[-1] - ema_26[-1]
            signal = pd.Series(ema_12 - ema_26).ewm(span=9).mean().values[-1]
            macd_histogram = macd - signal
            
            # Momentum score (composite)
            momentum_score = (
                (rsi - 50) / 50 * 0.3 +  # RSI contribution
                np.tanh(roc_10 / 10) * 0.3 +  # Short-term ROC
                np.tanh(roc_20 / 20) * 0.2 +  # Long-term ROC
                np.tanh(macd_histogram / (abs(close[-1]) * 0.01)) * 0.2  # MACD
            ) * 100
            
            return {
                'score': float(momentum_score),
                'rsi': float(rsi),
                'roc_short': float(roc_10),
                'roc_long': float(roc_20),
                'macd_histogram': float(macd_histogram)
            }
            
        except Exception as e:
            print(f"[WARN] Momentum analysis error: {e}")
            return {'score': 0.0, 'rsi': 50.0, 'roc_short': 0.0, 'roc_long': 0.0, 'macd_histogram': 0.0}
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict:
        """Volume profile analysis"""
        try:
            if 'Volume' not in df.columns or df['Volume'].sum() == 0:
                return {'profile': 'unknown', 'trend': 'neutral', 'strength': 0.5}
            
            volume = df['Volume'].values
            close = df['y'].values
            
            # Volume moving average
            vol_ma = pd.Series(volume).rolling(20).mean().values
            
            # Current volume vs average
            current_vol_ratio = volume[-1] / (vol_ma[-1] + 1e-6)
            
            # Volume trend
            vol_trend = np.polyfit(range(len(volume[-20:])), volume[-20:], 1)[0]
            vol_trend_normalized = np.tanh(vol_trend / (np.mean(volume[-20:]) + 1e-6))
            
            # On-Balance Volume (OBV)
            price_changes = np.diff(close)
            obv = np.cumsum(np.where(price_changes > 0, volume[1:], -volume[1:]))
            obv_trend = np.polyfit(range(len(obv[-20:])), obv[-20:], 1)[0]
            
            # Volume profile
            if current_vol_ratio > 1.5:
                profile = 'high_volume'
            elif current_vol_ratio < 0.5:
                profile = 'low_volume'
            else:
                profile = 'normal'
            
            # Volume-price relationship
            if obv_trend > 0 and price_changes[-20:].mean() > 0:
                volume_price_trend = 'confirming_bullish'
            elif obv_trend < 0 and price_changes[-20:].mean() < 0:
                volume_price_trend = 'confirming_bearish'
            else:
                volume_price_trend = 'diverging'
            
            return {
                'profile': profile,
                'trend': volume_price_trend,
                'strength': float(current_vol_ratio),
                'obv_trend': float(obv_trend)
            }
            
        except Exception as e:
            print(f"[WARN] Volume analysis error: {e}")
            return {'profile': 'unknown', 'trend': 'neutral', 'strength': 1.0, 'obv_trend': 0.0}
    
    def _analyze_microstructure(self, df: pd.DataFrame) -> Dict:
        """Market microstructure analysis"""
        try:
            close = df['y'].values
            
            # Price efficiency ratio
            net_change = abs(close[-1] - close[0])
            path_length = np.sum(np.abs(np.diff(close)))
            efficiency_ratio = net_change / (path_length + 1e-6)
            
            # Hurst exponent approximation (trend vs mean reversion)
            lags = range(2, min(20, len(close) // 2))
            tau = []
            for lag in lags:
                pp = np.subtract(close[lag:], close[:-lag])
                tau.append(np.sqrt(np.mean(pp**2)))
            
            try:
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0]
            except:
                hurst = 0.5
            
            # Market state
            if hurst > 0.55:
                market_state = 'trending'
            elif hurst < 0.45:
                market_state = 'mean_reverting'
            else:
                market_state = 'random_walk'
            
            return {
                'efficiency_ratio': float(efficiency_ratio),
                'hurst_exponent': float(hurst),
                'market_state': market_state
            }
            
        except Exception as e:
            print(f"[WARN] Microstructure analysis error: {e}")
            return {'efficiency_ratio': 0.5, 'hurst_exponent': 0.5, 'market_state': 'random_walk'}
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Correlation structure analysis"""
        try:
            close = df['y'].values
            
            # Auto-correlation at different lags
            autocorr_1 = np.corrcoef(close[1:], close[:-1])[0, 1] if len(close) > 1 else 0.0
            autocorr_5 = np.corrcoef(close[5:], close[:-5])[0, 1] if len(close) > 5 else 0.0
            autocorr_20 = np.corrcoef(close[20:], close[:-20])[0, 1] if len(close) > 20 else 0.0
            
            # Mean correlation
            mean_autocorr = np.mean([abs(autocorr_1), abs(autocorr_5), abs(autocorr_20)])
            
            return {
                'autocorr_1': float(autocorr_1),
                'autocorr_5': float(autocorr_5),
                'autocorr_20': float(autocorr_20),
                'mean_autocorr': float(mean_autocorr),
                'persistence': 'high' if mean_autocorr > 0.5 else 'low'
            }
            
        except Exception as e:
            print(f"[WARN] Correlation analysis error: {e}")
            return {'autocorr_1': 0.0, 'autocorr_5': 0.0, 'autocorr_20': 0.0, 
                    'mean_autocorr': 0.0, 'persistence': 'low'}
    
    def _calculate_regime_score(self, trend, vol, momentum, volume, microstructure, correlation) -> Dict:
        """Combine all metrics into regime scores"""
        return {
            'trend_score': trend['strength'] * (1 if trend['direction'] == 'bullish' else -1),
            'volatility_score': vol['percentile'],
            'momentum_score': momentum['score'],
            'volume_score': volume['strength'],
            'efficiency_score': microstructure['efficiency_ratio'] * 100,
            'hurst_score': (microstructure['hurst_exponent'] - 0.5) * 200,
            'persistence_score': correlation['mean_autocorr'] * 100
        }
    
    def _classify_regime(self, scores: Dict) -> str:
        """Classify market regime based on scores"""
        trend = scores['trend_score']
        vol = scores['volatility_score']
        momentum = scores['momentum_score']
        
        # Strong trending market
        if abs(trend) > 60 and vol < 70:
            if trend > 0 and momentum > 30:
                return 'strong_bull_trend'
            elif trend < 0 and momentum < -30:
                return 'strong_bear_trend'
        
        # Moderate trend
        if abs(trend) > 30 and vol < 80:
            if trend > 0:
                return 'moderate_bull_trend'
            else:
                return 'moderate_bear_trend'
        
        # High volatility
        if vol > 80:
            if abs(momentum) > 40:
                return 'volatile_trending'
            else:
                return 'volatile_choppy'
        
        # Range-bound
        if abs(trend) < 25 and vol < 50:
            return 'range_bound'
        
        # Default
        return 'transitioning'
    
    def _calculate_confidence(self, scores: Dict) -> float:
        """Calculate confidence in regime classification"""
        # Consistency across metrics
        score_values = list(scores.values())
        score_std = np.std(score_values)
        
        # Lower std = higher confidence
        confidence = 1.0 / (1.0 + score_std / 50.0)
        
        return float(min(1.0, max(0.0, confidence)))
    
    def _detect_market_phase(self, df: pd.DataFrame) -> str:
        """Detect market phase (accumulation, markup, distribution, markdown)"""
        try:
            close = df['y'].values
            
            # Price action over different periods
            short_change = (close[-5:].mean() - close[-10:-5].mean()) / close[-10:-5].mean() if len(close) >= 10 else 0
            long_change = (close[-20:].mean() - close[-40:-20].mean()) / close[-40:-20].mean() if len(close) >= 40 else short_change
            
            # Volume if available
            volume_trend = 0
            if 'Volume' in df.columns:
                volume = df['Volume'].values
                volume_trend = (volume[-10:].mean() - volume[-20:-10].mean()) / volume[-20:-10].mean() if len(volume) >= 20 else 0
            
            # Phase classification
            if short_change > 0.02 and long_change > 0.01 and volume_trend > 0:
                return 'markup'  # Rising prices, increasing volume
            elif short_change < -0.02 and long_change < -0.01 and volume_trend > 0:
                return 'markdown'  # Falling prices, increasing volume
            elif abs(short_change) < 0.01 and volume_trend < 0:
                if long_change > 0:
                    return 'distribution'  # Flat prices after uptrend, decreasing volume
                else:
                    return 'accumulation'  # Flat prices after downtrend, decreasing volume
            else:
                return 'transition'
            
        except Exception as e:
            return 'unknown'
    
    def _classify_volatility_state(self, vol_metrics: Dict) -> str:
        """Classify volatility state"""
        percentile = vol_metrics['percentile']
        
        if percentile < 20:
            return 'extremely_low'
        elif percentile < 40:
            return 'low'
        elif percentile < 60:
            return 'normal'
        elif percentile < 80:
            return 'elevated'
        else:
            return 'extreme'
    
    def _assess_risk_level(self, scores: Dict) -> str:
        """Assess overall risk level"""
        vol = scores['volatility_score']
        efficiency = scores['efficiency_score']
        
        risk_score = vol * 0.6 + (100 - efficiency) * 0.4
        
        if risk_score < 30:
            return 'very_low'
        elif risk_score < 50:
            return 'low'
        elif risk_score < 70:
            return 'moderate'
        elif risk_score < 85:
            return 'high'
        else:
            return 'very_high'
    
    def _calculate_optimal_leverage(self, scores: Dict, vol_metrics: Dict) -> float:
        """Calculate recommended leverage based on conditions"""
        # Base leverage
        base_leverage = 10.0
        
        # Adjust for volatility
        vol_percentile = vol_metrics['percentile']
        vol_adjustment = 1.0 - (vol_percentile / 100.0) * 0.7
        
        # Adjust for trend strength
        trend_strength = abs(scores['trend_score']) / 100.0
        trend_adjustment = 0.7 + trend_strength * 0.3
        
        # Adjust for efficiency
        efficiency_adjustment = scores['efficiency_score'] / 100.0
        
        # Calculate optimal leverage
        optimal_leverage = base_leverage * vol_adjustment * trend_adjustment * efficiency_adjustment
        
        # Clamp between safe limits
        return float(max(1.0, min(optimal_leverage, 20.0)))
    
    def _get_default_regime(self) -> Dict:
        """Return default regime when detection fails"""
        return {
            'regime': 'unknown',
            'confidence': 0.0,
            'market_phase': 'unknown',
            'volatility_state': 'normal',
            'trend_strength': 25.0,
            'volatility_percentile': 50.0,
            'momentum_score': 0.0,
            'volume_profile': 'unknown',
            'risk_level': 'moderate',
            'recommended_leverage': 5.0,
            'metrics': {}
        }
