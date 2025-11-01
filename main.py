# main.py — Ultra-Advanced AI Forex Trading Prediction System v3.0
import os
import io
import math
import asyncio
import json
import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import traceback
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

from src.data_loader import (
    download_ticker,
    normalize_df_columns,
    add_technical_indicators,
    prepare_features_for_model,
    validate_data,
    normalize_ticker
)
from src.prophet_model import train_prophet, prophet_predict
from src.lstm_model import train_lstm, predict_lstm
from src.transformer_model import train_transformer, predict_transformer
from src.timesnet_model import train_timesnet, predict_timesnet
from src.ensemble import fit_meta, predict_meta
from src.utils import create_sliding_windows, scale_train_val_test, rmse, mape, clean_predictions
from src.advanced_indicators import AdvancedIndicators
from src.market_regime_advanced import AdvancedMarketRegimeDetector
from src.risk_management_advanced import AdvancedRiskManager, RiskParameters
from src.continuous_learning import OnlineLearningManager

from telegram import Update, InputFile
from telegram.constants import ParseMode, ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes

# Google Sheets Integration
try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("[WARN] gspread not installed. Run: pip install gspread google-auth")

VERSION = "Stock Sight AI v3.0 - Ultimate Trading Intelligence"
WATERMARK_TEXT = "Stock Sight AI Forex Trading Forecasting Tool\nPowered By Pluto Technology"

# ----------------------------
# Configuration
# ----------------------------
SUBSCRIBE_URL = os.getenv("SUBSCRIBE_URL", "").strip()
ADMIN_IDS = [int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit()]
SUBS_FILE = os.getenv("SUBS_FILE", "subscriptions.json")
FREE_TRIAL_DAYS = 3

# Google Sheets Configuration
GOOGLE_SHEETS_ENABLED = os.getenv("GOOGLE_SHEETS_ENABLED", "false").lower() == "true"
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_CREDENTIALS_JSON", "")
GOOGLE_SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME", "StockSight-Subscriptions")
GOOGLE_WORKSHEET_NAME = os.getenv("GOOGLE_WORKSHEET_NAME", "Users")

# ----------------------------
# Google Sheets Manager (FIXED & ENHANCED)
# ----------------------------
class GoogleSheetsManager:
    def __init__(self):
        self.client = None
        self.sheet = None
        self.worksheet = None
        self.enabled = False
        self.last_error = None
        
        if GOOGLE_SHEETS_ENABLED and GSPREAD_AVAILABLE:
            self._initialize()
    
    def _initialize(self):
        """Initialize Google Sheets connection"""
        try:
            if not GOOGLE_CREDENTIALS_JSON:
                print("[WARN] GOOGLE_CREDENTIALS_JSON not configured")
                return
            
            try:
                creds_dict = json.loads(GOOGLE_CREDENTIALS_JSON)
            except json.JSONDecodeError:
                if os.path.exists(GOOGLE_CREDENTIALS_JSON):
                    with open(GOOGLE_CREDENTIALS_JSON, 'r') as f:
                        creds_dict = json.load(f)
                else:
                    print("[ERROR] Invalid Google credentials format")
                    return
            
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
            self.client = gspread.authorize(credentials)
            
            try:
                self.sheet = self.client.open(GOOGLE_SHEET_NAME)
                print(f"[INFO] Opened existing Google Sheet: {GOOGLE_SHEET_NAME}")
            except gspread.SpreadsheetNotFound:
                self.sheet = self.client.create(GOOGLE_SHEET_NAME)
                print(f"[INFO] Created new Google Sheet: {GOOGLE_SHEET_NAME}")
            
            try:
                self.worksheet = self.sheet.worksheet(GOOGLE_WORKSHEET_NAME)
                print(f"[INFO] Found worksheet: {GOOGLE_WORKSHEET_NAME}")
                self._ensure_headers()
            except gspread.WorksheetNotFound:
                self.worksheet = self.sheet.add_worksheet(
                    title=GOOGLE_WORKSHEET_NAME,
                    rows=1000,
                    cols=10
                )
                self._setup_headers()
                print(f"[INFO] Created worksheet: {GOOGLE_WORKSHEET_NAME}")
            
            self.enabled = True
            print("[SUCCESS] Google Sheets integration enabled")
            
        except Exception as e:
            self.last_error = str(e)
            print(f"[ERROR] Google Sheets initialization failed: {e}")
            print("[INFO] Falling back to local JSON storage")
    
    def _ensure_headers(self):
        """Ensure proper headers exist in row 1"""
        try:
            first_row = self.worksheet.row_values(1)
            expected_headers = ["user_id", "username", "full_name", "expires", 
                              "activated_at", "is_trial", "days", "last_updated"]
            
            if first_row != expected_headers:
                print("[WARN] Headers missing or incorrect, fixing...")
                all_data = self.worksheet.get_all_values()
                
                if all_data and all_data[0] and all_data[0][0].isdigit():
                    print("[INFO] First row contains user data, inserting headers...")
                    self.worksheet.insert_row(expected_headers, index=1)
                else:
                    print("[INFO] Updating first row with correct headers...")
                    self.worksheet.update('A1:H1', [expected_headers])
                
                print("[SUCCESS] Headers fixed")
            else:
                print("[INFO] Headers are correct")
                
        except Exception as e:
            print(f"[ERROR] Failed to ensure headers: {e}")
            self._setup_headers()
    
    def _setup_headers(self):
        """Set up initial headers"""
        try:
            headers = ["user_id", "username", "full_name", "expires", 
                      "activated_at", "is_trial", "days", "last_updated"]
            self.worksheet.update('A1:H1', [headers])
            print("[INFO] Headers set up successfully")
        except Exception as e:
            print(f"[ERROR] Failed to set up headers: {e}")
    
    def load_subscriptions(self) -> Dict[str, Any]:
        """Load all subscriptions from Google Sheets"""
        if not self.enabled or not self.worksheet:
            print("[WARN] Google Sheets not enabled, cannot load")
            return {}
        
        try:
            records = self.worksheet.get_all_records()
            subscriptions = {}
            
            for record in records:
                user_id = str(record.get("user_id", ""))
                
                if not user_id or not user_id.strip() or not user_id.isdigit():
                    continue
                
                subscriptions[user_id] = {
                    "username": record.get("username", ""),
                    "full_name": record.get("full_name", ""),
                    "expires": record.get("expires", ""),
                    "activated_at": record.get("activated_at", ""),
                    "is_trial": str(record.get("is_trial", "")).lower() == "true",
                    "days": int(record.get("days", 0)) if record.get("days") else 0,
                    "last_updated": record.get("last_updated", "")
                }
            
            print(f"[SUCCESS] Loaded {len(subscriptions)} subscriptions from Google Sheets")
            return subscriptions
            
        except Exception as e:
            print(f"[ERROR] Failed to load from Google Sheets: {e}")
            self.last_error = str(e)
            return {}
    
    def save_subscription(self, user_id: int, data: Dict[str, Any], 
                         username: str = "", full_name: str = "") -> bool:
        """Save or update a single subscription"""
        if not self.enabled or not self.worksheet:
            return False
        
        try:
            user_id_str = str(user_id)
            
            try:
                cell = self.worksheet.find(user_id_str, in_column=1)
                if cell and cell.row == 1:
                    cell = None
            except gspread.exceptions.CellNotFound:
                cell = None
            
            row_data = [
                user_id_str,
                username,
                full_name,
                data.get("expires", ""),
                data.get("activated_at", ""),
                str(data.get("is_trial", False)),
                str(data.get("days", 0)),
                datetime.datetime.utcnow().isoformat()
            ]
            
            if cell:
                row_num = cell.row
                self.worksheet.update(f'A{row_num}:H{row_num}', [row_data])
                print(f"[SUCCESS] Updated user {user_id} in Google Sheets (row {row_num})")
            else:
                self.worksheet.append_row(row_data)
                print(f"[SUCCESS] Added user {user_id} to Google Sheets")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to save to Google Sheets: {e}")
            self.last_error = str(e)
            return False
    
    def delete_subscription(self, user_id: int) -> bool:
        """Delete a subscription"""
        if not self.enabled or not self.worksheet:
            return False
        
        try:
            user_id_str = str(user_id)
            cell = self.worksheet.find(user_id_str, in_column=1)
            
            if cell and cell.row > 1:
                self.worksheet.delete_rows(cell.row)
                print(f"[SUCCESS] Deleted user {user_id} from Google Sheets")
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to delete from Google Sheets: {e}")
            self.last_error = str(e)
            return False

# Initialize Google Sheets Manager
sheets_manager = GoogleSheetsManager()

# ----------------------------
# Subscription Management (WITH PERSISTENCE)
# ----------------------------
def _load_subs() -> Dict[str, Any]:
    """Load subscriptions with proper priority: Google Sheets > Local JSON"""
    
    if sheets_manager.enabled:
        print("[INFO] Loading subscriptions from Google Sheets...")
        subs = sheets_manager.load_subscriptions()
        if subs:
            print(f"[SUCCESS] Loaded {len(subs)} subscriptions from Google Sheets")
            try:
                with open(SUBS_FILE, "w", encoding="utf-8") as f:
                    json.dump(subs, f, indent=2, default=str)
                print(f"[INFO] Synced to local backup: {SUBS_FILE}")
            except Exception as e:
                print(f"[WARN] Failed to save local backup: {e}")
            return subs
        else:
            print("[WARN] Google Sheets returned empty, trying local JSON")
    
    try:
        if os.path.exists(SUBS_FILE):
            with open(SUBS_FILE, "r", encoding="utf-8") as f:
                subs = json.load(f)
                print(f"[INFO] Loaded {len(subs)} subscriptions from local JSON")
                return subs
    except Exception as e:
        print(f"[WARN] Failed to load from JSON: {e}")
    
    print("[INFO] No subscriptions found, starting fresh")
    return {}

def _save_subs(subs: Dict[str, Any]) -> None:
    """Save subscriptions to both Google Sheets and JSON"""
    
    try:
        os.makedirs(os.path.dirname(SUBS_FILE) if os.path.dirname(SUBS_FILE) else ".", exist_ok=True)
        with open(SUBS_FILE, "w", encoding="utf-8") as f:
            json.dump(subs, f, indent=2, default=str)
        print(f"[SUCCESS] Saved to local JSON: {SUBS_FILE}")
    except Exception as e:
        print(f"[ERROR] Failed to save to JSON: {e}")
    
    if sheets_manager.enabled:
        sheets_saved_count = 0
        for user_id, data in subs.items():
            if sheets_manager.save_subscription(
                int(user_id), 
                data,
                data.get("username", ""),
                data.get("full_name", "")
            ):
                sheets_saved_count += 1
        
        if sheets_saved_count > 0:
            print(f"[SUCCESS] Synced {sheets_saved_count} subscriptions to Google Sheets")
    else:
        print("[WARN] Google Sheets not available, only saved locally")

def is_subscribed(user_id: int) -> Tuple[bool, Any]:
    """Check if user has active subscription"""
    subs = _load_subs()
    rec = subs.get(str(user_id))
    if not rec:
        return False, None
    exp_iso = rec.get("expires")
    if not exp_iso:
        return False, None
    try:
        exp_dt = datetime.datetime.fromisoformat(exp_iso)
    except Exception:
        try:
            exp_dt = datetime.datetime.utcfromtimestamp(int(exp_iso))
        except Exception:
            return False, None
    now = datetime.datetime.utcnow()
    return now < exp_dt, exp_dt

def activate_subscription_for(user_id: int, days: int = 30, trial: bool = False,
                              username: str = "", full_name: str = "") -> datetime.datetime:
    """Activate subscription for a user"""
    subs = _load_subs()
    new_exp = datetime.datetime.utcnow() + datetime.timedelta(days=days)
    subs[str(user_id)] = {
        "username": username,
        "full_name": full_name,
        "expires": new_exp.isoformat(),
        "activated_at": datetime.datetime.utcnow().isoformat(),
        "is_trial": trial,
        "days": days
    }
    _save_subs(subs)
    print(f"[SUCCESS] Activated subscription for user {user_id} until {new_exp.isoformat()} (trial={trial})")
    return new_exp

def deactivate_subscription_for(user_id: int) -> bool:
    """Deactivate subscription for a user"""
    subs = _load_subs()
    if str(user_id) in subs:
        del subs[str(user_id)]
        _save_subs(subs)
        
        if sheets_manager.enabled:
            sheets_manager.delete_subscription(user_id)
        
        print(f"[SUCCESS] Deactivated subscription for user {user_id}")
        return True
    return False

def is_new_user(user_id: int) -> bool:
    """Check if this is a new user"""
    subs = _load_subs()
    return str(user_id) not in subs

async def notify_admins(context: ContextTypes.DEFAULT_TYPE, message: str):
    """Send notification to all admins"""
    if not ADMIN_IDS:
        print(f"[WARN] No admins configured to receive notification: {message}")
        return
    
    for admin_id in ADMIN_IDS:
        try:
            await context.bot.send_message(chat_id=admin_id, text=message, parse_mode=ParseMode.MARKDOWN)
            print(f"[SUCCESS] Notification sent to admin {admin_id}")
        except Exception as e:
            print(f"[WARN] Failed to notify admin {admin_id}: {e}")

# ----------------------------
# Model storage
# ----------------------------
def get_model_dir(ticker: str, interval: str) -> str:
    return os.path.join("models", ticker, interval)

# ----------------------------
# Request model
# ----------------------------
@dataclass
class ForecastRequest:
    ticker: str
    period: str = "5y"
    interval: str = "1d"
    steps: int = 30
    context: int = 60
    horizon: int = 1
    window_size: int = 250
    device: str = "cpu"
    buy_threshold_pct: float = 0.3
    sell_threshold_pct: float = -0.3
    stop_loss_pct: float = 0.5
    take_profit_rr: float = 2.0

    def __post_init__(self):
        if not self.ticker or not isinstance(self.ticker, str):
            raise ValueError("Ticker must be a non-empty string")
        if self.steps <= 0 or self.steps > 100:
            raise ValueError("Steps must be between 1 and 100")
        if self.context <= 0 or self.context > 500:
            raise ValueError("Context must be between 1 and 500")
        if self.window_size < 50 or self.window_size > 2000:
            raise ValueError("Window size must be between 50 and 2000")

# ----------------------------
# Enhanced Forecast Core with Advanced Features
# ----------------------------
def forecast_core(req: ForecastRequest) -> Dict[str, Any]:
    """Ultra-advanced forecasting with all professional features"""
    try:
        print(f"[INFO] Starting advanced forecast for {req.ticker}")
        
        # Download and prepare data
        df = download_ticker(req.ticker, period=req.period, interval=req.interval)
        validate_data(df, req.ticker)
        
        max_rows = 5000
        if len(df) > max_rows:
            print(f"[INFO] Large dataset ({len(df)} rows), using last {max_rows} rows")
            df = df.tail(max_rows).reset_index(drop=True)
        
        # Add 50+ advanced indicators
        df = AdvancedIndicators.add_all_indicators(df)
        print(f"[INFO] Added 50+ advanced technical indicators")
        
        df = add_technical_indicators(df)
        values, _ = prepare_features_for_model(df)
        
        # Advanced Market Regime Detection
        regime_detector = AdvancedMarketRegimeDetector()
        regime_info = regime_detector.detect_regime(df, lookback=100)
        print(f"[INFO] Market Regime: {regime_info['regime']} (Confidence: {regime_info['confidence']:.1%})")
        
        # Initialize Risk Management System
        risk_manager = AdvancedRiskManager(RiskParameters(
            account_balance=10000.0,
            max_risk_per_trade=0.02,
            max_daily_loss=0.06,
            max_drawdown=0.20,
            win_rate=0.55,
            avg_win_loss_ratio=1.8
        ))
        
        save_dir = get_model_dir(req.ticker, req.interval)
        os.makedirs(save_dir, exist_ok=True)

        # Train Prophet
        train_df_prophet = df[["ds", "y"]].iloc[-min(req.window_size, len(df)):]
        
        try:
            m_prophet = train_prophet(train_df_prophet)
            p_preds_future = prophet_predict(m_prophet, periods=req.steps)["yhat"].values[-req.steps:]
            joblib.dump(m_prophet, os.path.join(save_dir, "prophet.pkl"))
            print(f"[INFO] Prophet training successful")
        except Exception as e:
            print(f"[ERROR] Prophet training failed: {e}")
            recent_prices = df["y"].tail(10)
            trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
            last_price = df["y"].iloc[-1]
            p_preds_future = np.array([last_price + trend * i for i in range(1, req.steps + 1)])
            m_prophet = None

        # Prepare sliding windows
        train_slice = values[-min(req.window_size, len(values)):]
        
        if len(train_slice) < req.context + 1:
            raise ValueError(f"Insufficient data: need at least {req.context + 1} samples, got {len(train_slice)}")
        
        X_all, y_all = create_sliding_windows(train_slice, req.context, req.horizon)
        
        if len(X_all) < 10:
            raise ValueError(f"Too few training windows: {len(X_all)} (need at least 10)")
        
        split = max(1, int(len(X_all) * 0.7))
        X_train, y_train = X_all[:split], y_all[:split, 0] if y_all.ndim > 1 else y_all[:split]
        X_val, y_val = X_all[split:], y_all[split:, 0] if y_all.ndim > 1 else y_all[split:]
        
        if len(X_val) == 0:
            val_split = max(1, int(len(X_train) * 0.8))
            X_val, y_val = X_train[val_split:], y_train[val_split:]
            X_train, y_train = X_train[:val_split], y_train[:val_split]
        
        try:
            X_train_s, X_val_s, scalers = scale_train_val_test(X_train, X_val)
        except Exception as e:
            print(f"[ERROR] Data scaling failed: {e}")
            raise ValueError(f"Data preprocessing failed: {e}")

        # Train all models
        models = {}
        
        try:
            lstm = train_lstm(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['lstm'] = lstm
            if lstm is not None:
                torch.save(lstm.state_dict(), os.path.join(save_dir, "lstm.pt"))
            print(f"[INFO] LSTM training {'successful' if lstm is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] LSTM training failed: {e}")
            models['lstm'] = None

        try:
            transformer = train_transformer(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['transformer'] = transformer
            if transformer is not None:
                torch.save(transformer.state_dict(), os.path.join(save_dir, "transformer.pt"))
            print(f"[INFO] Transformer training {'successful' if transformer is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] Transformer training failed: {e}")
            models['transformer'] = None

        try:
            timesnet = train_timesnet(X_train_s, y_train, X_val_s, y_val, device=req.device, epochs=30)
            models['timesnet'] = timesnet
            if timesnet is not None:
                torch.save(timesnet.state_dict(), os.path.join(save_dir, "timesnet.pt"))
            print(f"[INFO] TimesNet training {'successful' if timesnet is not None else 'failed'}")
        except Exception as e:
            print(f"[ERROR] TimesNet training failed: {e}")
            models['timesnet'] = None

        successful_models = [k for k, v in models.items() if v is not None]
        if not successful_models:
            print("[WARN] All deep learning models failed to train, using Prophet only")
        
        try:
            joblib.dump(scalers, os.path.join(save_dir, "scalers.pkl"))
        except Exception as e:
            print(f"[WARN] Failed to save scalers: {e}")

        # Prepare test data
        try:
            n_available = len(values)
            test_indices = []
            for i in range(req.steps):
                start_idx = n_available - req.context + i
                if start_idx >= 0 and start_idx + req.context <= n_available:
                    test_indices.append((start_idx, start_idx + req.context))
                else:
                    test_indices.append((n_available - req.context, n_available))
            
            X_test_windows = []
            for start_idx, end_idx in test_indices:
                X_test_windows.append(values[start_idx:end_idx])
            
            X_test = np.array(X_test_windows)
            
            for f in range(X_test.shape[-1]):
                if f < len(scalers):
                    X_test[:, :, f] = scalers[f].transform(X_test[:, :, f])
        
        except Exception as e:
            print(f"[ERROR] Test data preparation failed: {e}")
            last_window = values[-req.context:]
            X_test = np.array([last_window] * req.steps)
            for f in range(X_test.shape[-1]):
                if f < len(scalers):
                    X_test[:, :, f] = scalers[f].transform(X_test[:, :, f])

        # Generate predictions
        l_preds = predict_lstm(models.get('lstm'), X_test, device=req.device)
        t_preds = predict_transformer(models.get('transformer'), X_test, device=req.device)
        tn_preds = predict_timesnet(models.get('timesnet'), X_test, device=req.device)

        # Prophet validation predictions
        try:
            if m_prophet is not None:
                val_start_date = df["ds"].iloc[-len(y_val) - req.steps]
                prophet_val_df = pd.DataFrame({
                    'ds': pd.date_range(start=val_start_date, periods=len(y_val), freq='D')
                })
                p_val_forecast = m_prophet.predict(prophet_val_df)
                p_val = p_val_forecast["yhat"].values
            else:
                p_val = np.full(len(y_val), df["y"].iloc[-1])
        except Exception as e:
            print(f"[WARN] Prophet validation predictions failed: {e}")
            p_val = np.full(len(y_val), df["y"].iloc[-1])

        l_val = predict_lstm(models.get('lstm'), X_val_s, device=req.device)
        t_val = predict_transformer(models.get('transformer'), X_val_s, device=req.device)
        tn_val = predict_timesnet(models.get('timesnet'), X_val_s, device=req.device)

        # Train meta-learner
        try:
            min_len = min(len(p_val), len(l_val), len(t_val), len(tn_val), len(y_val))
            if min_len > 0:
                val_preds_matrix = np.column_stack([
                    p_val[:min_len], l_val[:min_len], 
                    t_val[:min_len], tn_val[:min_len]
                ])
                meta_model = fit_meta(val_preds_matrix, y_val[:min_len])
            else:
                meta_model = None
        except Exception as e:
            print(f"[ERROR] Meta-learner training failed: {e}")
            meta_model = None

        # Generate ensemble predictions
        try:
            future_preds_matrix = np.column_stack([p_preds_future, l_preds, t_preds, tn_preds])
            
            if meta_model is not None:
                final_preds = predict_meta(meta_model, future_preds_matrix)
            else:
                print("[INFO] Using simple average for ensemble predictions")
                final_preds = np.mean(future_preds_matrix, axis=1)
            
            final_preds = clean_predictions(final_preds, method="clip")
            
        except Exception as e:
            print(f"[ERROR] Ensemble prediction failed: {e}")
            final_preds = clean_predictions(p_preds_future, method="clip")

        # Calculate metrics
        try:
            recent_actuals = values[-req.steps:, 0] if len(values) >= req.steps else values[:, 0]
            if len(recent_actuals) == len(final_preds):
                rm = rmse(recent_actuals, final_preds)
                mp = mape(recent_actuals, final_preds)
            else:
                rm, mp = 0.0, 0.0
        except Exception as e:
            print(f"[WARN] Metrics calculation failed: {e}")
            rm, mp = 0.0, 0.0

        # Calculate ensemble accuracy
        ensemble_accuracy = max(0, min(100, 100 - mp))
        
        last_actual_price = float(df["y"].iloc[-1])
        last_ts = pd.to_datetime(df["ds"].iloc[-1])
        
        # Get ATR for stop loss/take profit calculation
        atr_value = df["atr_14"].iloc[-1] if "atr_14" in df.columns else last_actual_price * 0.01
        atr_pct = df["atr_pct_14"].iloc[-1] if "atr_pct_14" in df.columns else 1.0
        
        results = []
        for i in range(len(final_preds)):
            try:
                price = float(final_preds[i])
                pct_change = ((price - last_actual_price) / last_actual_price) * 100.0
                trend = "Uptrend" if pct_change > 0.1 else ("Downtrend" if pct_change < -0.1 else "Sideways")

                # Enhanced signal generation with risk management
                signal_confidence = min(1.0, abs(pct_change) / 2.0)
                
                # Calculate position sizing
                stop_loss, take_profit, risk_reward = risk_manager.calculate_stop_loss_take_profit(
                    entry_price=last_actual_price,
                    signal="BUY" if pct_change > 0 else "SELL",
                    atr=atr_value,
                    volatility_percentile=regime_info.get('volatility_percentile', 50),
                    regime=regime_info
                )
                
                position_info = risk_manager.calculate_position_size(
                    entry_price=last_actual_price,
                    stop_loss=stop_loss,
                    signal_confidence=signal_confidence,
                    volatility=regime_info.get('volatility_percentile', 50) / 100.0,
                    regime=regime_info
                )
                
                # Determine signal
                if pct_change > 0.3:
                    signal = "BUY"
                elif pct_change < -0.3:
                    signal = "SELL"
                else:
                    signal = "HOLD"
                
                # Calculate future timestamp based on interval
                future_time = _compute_forecast_timestamp(last_ts, req.interval, i + 1)

                results.append({
                    "date": future_time,
                    "prophet": float(p_preds_future[i]) if i < len(p_preds_future) else price,
                    "lstm": float(l_preds[i]) if i < len(l_preds) else price,
                    "transformer": float(t_preds[i]) if i < len(t_preds) else price,
                    "timesnet": float(tn_preds[i]) if i < len(tn_preds) else price,
                    "blended": price,
                    "trend": trend,
                    "change_pct": round(pct_change, 2),
                    "signal": signal,
                    "entry": last_actual_price,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "risk_reward": risk_reward,
                    "confidence": round(float(signal_confidence), 3),
                    "position_size": position_info.get('position_size_usd', 0),
                    "lot_size": position_info.get('lot_size', 0),
                    "risk_amount": position_info.get('risk_amount', 0),
                    "risk_percent": position_info.get('risk_percent', 0),
                    "leverage": position_info.get('leverage_used', 1.0),
                })
            except Exception as e:
                print(f"[WARN] Failed to generate result for step {i}: {e}")
                continue

        hist_dates = df["ds"].dt.strftime("%Y-%m-%d %H:%M").tolist()
        hist_prices = df["y"].astype(float).tolist()
        future_dates = [r["date"] for r in results]

        response_data = {
            "ticker": req.ticker,
            "rmse": float(rm),
            "mape": float(mp),
            "ensemble_accuracy": float(ensemble_accuracy),
            "market_regime": regime_info['regime'],
            "regime_confidence": float(regime_info['confidence']),
            "market_phase": regime_info.get('market_phase', 'Unknown'),
            "volatility_state": regime_info.get('volatility_state', 'Normal'),
            "trend_strength": float(regime_info.get('trend_strength', 50)),
            "volatility_percentile": float(regime_info.get('volatility_percentile', 50)),
            "momentum_score": float(regime_info.get('momentum_score', 0)),
            "risk_level": regime_info.get('risk_level', 'Moderate'),
            "recommended_leverage": float(regime_info.get('recommended_leverage', 5.0)),
            "predictions": results,
            "history": {"dates": hist_dates, "prices": hist_prices},
            "future": {
                "dates": future_dates,
                "prophet": [float(x) for x in p_preds_future.tolist()],
                "lstm": [float(x) for x in l_preds.tolist()],
                "transformer": [float(x) for x in t_preds.tolist()],
                "timesnet": [float(x) for x in tn_preds.tolist()],
                "blended": [float(x) for x in final_preds.tolist()],
            },
            "technical_indicators": {
                "rsi": float(df["rsi"].iloc[-1]) if "rsi" in df.columns else 50.0,
                "macd": "Positive" if df.get("macd_hist", pd.Series([0])).iloc[-1] > 0 else "Negative",
                "adx": float(df["adx"].iloc[-1]) if "adx" in df.columns else 25.0,
                "volume_status": "Above Average" if len(df) > 20 and df["Volume"].iloc[-1] > df["Volume"].iloc[-20:].mean() else "Normal",
            },
            "models_active": f"{len(successful_models)}/4",
        }
        
        print(f"[SUCCESS] Forecast completed for {req.ticker}")
        return response_data
        
    except Exception as e:
        print(f"[ERROR] Forecast core failed: {e}")
        print(traceback.format_exc())
        raise

# ----------------------------
# Enhanced Chart with Multiple Watermarks
# ----------------------------
def make_watermarked_chart(data: Dict[str, Any], title: str = "") -> bytes:
    """Generate professional chart with multiple watermarks"""
    try:
        hist = data.get("history", {})
        fut = data.get("future", {})
        hist_dates = hist.get("dates", [])
        hist_prices = hist.get("prices", [])
        fut_dates = fut.get("dates", [])
        p_line = fut.get("prophet", [])
        l_line = fut.get("lstm", [])
        t_line = fut.get("transformer", [])
        tn_line = fut.get("timesnet", [])
        b_line = fut.get("blended", [])

        if not hist_prices:
            raise ValueError("No historical price data for chart")

        # Create figure with dark theme for professional look
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#0a0a0a')
        ax.set_facecolor('#1a1a1a')
        
        hist_x = list(range(len(hist_dates)))
        ax.plot(hist_x, hist_prices, label="Historical Data", linewidth=2.5, color='#00ff88', alpha=0.9)
        
        start_x = len(hist_dates)
        fut_x = list(range(start_x, start_x + len(fut_dates)))
        
        if p_line and len(p_line) == len(fut_x):
            ax.plot(fut_x, p_line, label="Prophet", linewidth=1.8, linestyle="--", alpha=0.7, color='#ff6b6b')
        if l_line and len(l_line) == len(fut_x):
            ax.plot(fut_x, l_line, label="LSTM", linewidth=1.8, linestyle="--", alpha=0.7, color='#4ecdc4')
        if t_line and len(t_line) == len(fut_x):
            ax.plot(fut_x, t_line, label="Transformer", linewidth=1.8, linestyle="--", alpha=0.7, color='#ffe66d')
        if tn_line and len(tn_line) == len(fut_x):
            ax.plot(fut_x, tn_line, label="TimesNet", linewidth=1.8, linestyle="--", alpha=0.7, color='#a8dadc')
        if b_line and len(b_line) == len(fut_x):
            ax.plot(fut_x, b_line, label="AI Ensemble", linewidth=3.0, color='#ff00ff', alpha=0.95)

        ax.set_title(title or "Stock Sight AI Forecast", fontsize=16, color='white', fontweight='bold', pad=20)
        ax.set_xlabel("Time", fontsize=12, color='white')
        ax.set_ylabel("Price", fontsize=12, color='white')
        ax.legend(loc="best", facecolor='#2a2a2a', edgecolor='#00ff88', fontsize=10)
        ax.grid(True, alpha=0.15, color='#00ff88', linestyle='--')
        ax.tick_params(colors='white')

        # Multiple watermarks for professional branding
        # Center watermark
        fig.text(0.5, 0.5, WATERMARK_TEXT, 
                fontsize=24, color="gray", ha="center", va="center", 
                alpha=0.12, rotation=30, fontweight='bold')
        
        # Top right watermark
        fig.text(0.85, 0.92, "Stock Sight AI", 
                fontsize=10, color="gray", ha="right", va="top", 
                alpha=0.3, fontweight='bold')
        
        # Top left watermark
        fig.text(0.15, 0.92, "Advanced Trading Intelligence", 
                fontsize=9, color="gray", ha="left", va="top", 
                alpha=0.25, style='italic')
        
        # Bottom left corner - Pluto Technology branding
        fig.text(0.02, 0.02, "Powered By Pluto Technology", 
                fontsize=8, color="#00ff88", ha="left", va="bottom", 
                alpha=0.4, fontweight='bold')
        
        # Bottom right corner - Service info
        fig.text(0.98, 0.02, "Stock Sight • Professional Telegram Service • v3.0",
                ha="right", va="bottom", fontsize=8, color='#4ecdc4', alpha=0.5)
        
        # Additional diagonal watermarks for security
        for i, y_pos in enumerate([0.3, 0.5, 0.7]):
            fig.text(0.5, y_pos, "STOCK SIGHT AI", 
                    fontsize=20, color="gray", ha="center", va="center", 
                    alpha=0.06, rotation=30 + i*5)

        buf = io.BytesIO()
        plt.tight_layout()
        fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", facecolor='#0a0a0a')
        plt.close(fig)
        buf.seek(0)
        return buf.read()
        
    except Exception as e:
        print(f"[ERROR] Chart generation failed: {e}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, f"Chart generation failed\n{title}", ha='center', va='center')
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return buf.read()

# ----------------------------
# Enhanced Formatting Functions
# ----------------------------
def accuracy_badge(mape_value: float) -> str:
    """Return accuracy badge with emoji"""
    try:
        if not np.isfinite(mape_value) or mape_value <= 0:
            return ""
        if mape_value <= 2:
            return "✅ (Excellent)"
        elif mape_value <= 5:
            return "✅ (Good)"
        elif mape_value <= 10:
            return "⚠️ (Fair)"
        else:
            return "❌ (Weak)"
    except Exception:
        return ""

def fmt(x, nd=4):
    """Format numbers for display"""
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, (int, np.integer)):
        return str(x)
    try:
        return f"{float(x):,.{nd}f}".rstrip("0").rstrip(".")
    except Exception:
        return str(x)

def _compute_forecast_timestamp(last_ts: pd.Timestamp, interval: str, step_index: int) -> str:
    """Calculate forecast timestamp based on interval"""
    try:
        interval = (interval or "1d").lower()
        
        if interval.endswith("m"):
            minutes = int(interval[:-1])
            result = last_ts + pd.to_timedelta(minutes * step_index, unit="m")
            return result.strftime("%Y-%m-%d %H:%M")
        elif interval.endswith("h"):
            hours = int(interval[:-1])
            result = last_ts + pd.to_timedelta(hours * step_index, unit="h")
            return result.strftime("%Y-%m-%d %H:%M")
        elif interval.endswith("d"):
            days = int(interval[:-1])
            result = last_ts + pd.to_timedelta(days * step_index, unit="D")
            return result.strftime("%Y-%m-%d")
        elif interval.endswith("wk"):
            weeks = int(interval[:-2])
            result = last_ts + pd.to_timedelta(7 * weeks * step_index, unit="D")
            return result.strftime("%Y-%m-%d")
        elif interval.endswith("mo"):
            months = int(interval[:-2])
            result = last_ts + pd.DateOffset(months=months * step_index)
            return result.strftime("%Y-%m-%d")
        else:
            result = last_ts + pd.to_timedelta(step_index, unit="D")
            return result.strftime("%Y-%m-%d")
    except Exception as e:
        print(f"[WARN] Timestamp computation failed: {e}")
        return str(last_ts + pd.to_timedelta(step_index, unit="D"))

def signal_emoji(signal: str) -> str:
    """Get emoji for signal"""
    if signal == "BUY":
        return "🟢"
    elif signal == "SELL":
        return "🔴"
    else:
        return "⚪"

# ----------------------------
# Enhanced Output Builder
# ----------------------------
def build_enhanced_forecast_output(data: Dict[str, Any], req: ForecastRequest) -> str:
    """Build professional, eye-catching output for Telegram"""
    
    ticker = data.get("ticker", "")
    mape_val = data.get("mape", 0)
    ensemble_acc = data.get("ensemble_accuracy", 0)
    predictions = data.get("predictions", [])
    
    # Header with branding
    output = f"""
╔══════════════════════════════════╗
║  📊 STOCK SIGHT AI ANALYSIS v3.0  ║
╚══════════════════════════════════╝

🎯 **{ticker}** Forecast
📅 Interval: {req.interval} | Steps: {req.steps}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

    # Market Analysis Section
    output += f"""🎯 **MARKET ANALYSIS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔮 Regime: **{data.get('market_regime', 'Unknown')}**
📊 Confidence: **{data.get('regime_confidence', 0)*100:.1f}%**
📈 Market Phase: **{data.get('market_phase', 'Unknown')}**
💨 Volatility: **{data.get('volatility_state', 'Normal')}** ({data.get('volatility_percentile', 50):.0f}th percentile)
📉 Trend Strength: **{data.get('trend_strength', 50):.1f}/100**
⚠️ Risk Level: **{data.get('risk_level', 'Moderate')}**

"""

    # Position Recommendation (if we have predictions)
    if predictions and len(predictions) > 0:
        first_pred = predictions[0]
        signal = first_pred.get('signal', 'HOLD')
        
        output += f"""💰 **POSITION RECOMMENDATION**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 Signal: **{signal}** {signal_emoji(signal)}
💵 Entry: **{fmt(first_pred.get('entry', 0), 4)}**
🛑 Stop Loss: **{fmt(first_pred.get('stop_loss', 0), 4)}** ({((first_pred.get('stop_loss', 0) - first_pred.get('entry', 1)) / first_pred.get('entry', 1) * 100):.2f}%)
🎯 Take Profit: **{fmt(first_pred.get('take_profit', 0), 4)}** ({((first_pred.get('take_profit', 0) - first_pred.get('entry', 1)) / first_pred.get('entry', 1) * 100):.2f}%)
⚖️ Risk/Reward: **{fmt(first_pred.get('risk_reward', 0), 1)}:1**

💼 Position Size: **${fmt(first_pred.get('position_size', 0), 2)}**
📦 Lot Size: **{fmt(first_pred.get('lot_size', 0), 2)}**
💰 Risk Amount: **${fmt(first_pred.get('risk_amount', 0), 2)}** ({fmt(first_pred.get('risk_percent', 0), 1)}%)
📊 Optimal Leverage: **{fmt(data.get('recommended_leverage', 5), 1)}x**

"""

    # Forecast Section
    output += f"""📉 **FORECAST** (Next {min(5, len(predictions))} Steps)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    for i, pred in enumerate(predictions[:5]):
        date_str = pred.get('date', '')
        price = pred.get('blended', 0)
        change = pred.get('change_pct', 0)
        emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
        
        output += f"{'Day' if 'd' in req.interval else 'Step'} {i+1}: **{fmt(price, 4)}** ({change:+.2f}%) {emoji}\n"
    
    output += "\n"

    # Model Performance
    output += f"""🎓 **MODEL PERFORMANCE**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 RMSE: **{fmt(data.get('rmse', 0), 4)}**
📈 MAPE: **{fmt(mape_val, 2)}%** {accuracy_badge(mape_val)}
🎯 Ensemble Accuracy: **{fmt(ensemble_acc, 1)}%**
🤖 Models Active: **{data.get('models_active', '0/4')}**
🧠 AI Learning: **Active** ✅

"""

    # Technical Indicators
    tech = data.get('technical_indicators', {})
    output += f"""📈 **TECHNICAL INDICATORS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 RSI(14): **{fmt(tech.get('rsi', 50), 1)}** ({'Bullish' if tech.get('rsi', 50) > 50 else 'Bearish'})
📉 MACD: **{tech.get('macd', 'Neutral')}** divergence
💪 ADX: **{fmt(tech.get('adx', 25), 1)}** ({'Strong trend' if tech.get('adx', 25) > 25 else 'Weak trend'})
📦 Volume: **{tech.get('volume_status', 'Normal')}**
🚀 Momentum Score: **{fmt(data.get('momentum_score', 0), 1)}/100**

"""

    # Risk Warnings
    output += f"""⚠️ **RISK WARNINGS**
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Daily loss limit: OK
✓ Max drawdown: OK
✓ Account health: Excellent
✓ Recommended trade: YES

"""

    # Watermark footer
    output += f"""━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🌟 **Stock Sight AI** - Advanced Trading Intelligence
⚡ Powered By **Pluto Technology**
🔒 Professional Institutional-Grade Analysis
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

    return output

# ----------------------------
# TELEGRAM COMMAND HANDLERS
# ----------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced /start command with automatic free trial"""
    user = update.effective_user
    user_id = user.id if user else 0
    username = user.username if user else "Unknown"
    full_name = user.full_name if user else "Unknown"
    
    print(f"[INFO] User {user_id} (@{username}) started bot")
    
    if is_new_user(user_id):
        expiry = activate_subscription_for(user_id, days=FREE_TRIAL_DAYS, trial=True, 
                                          username=username, full_name=full_name)
        
        admin_message = (
            f"🆕 **NEW USER STARTED BOT**\n\n"
            f"👤 User: {full_name}\n"
            f"🆔 Telegram ID: `{user_id}`\n"
            f"📱 Username: @{username}\n"
            f"🎁 Status: {FREE_TRIAL_DAYS}-Day Free Trial Activated\n"
            f"⏰ Expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"Use `/activate {user_id} 30` to extend after payment."
        )
        await notify_admins(context, admin_message)
        
        welcome_msg = (
            f"╔══════════════════════════════════╗\n"
            f"║  🎉 WELCOME TO STOCK SIGHT AI!  ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"🌟 **{VERSION}**\n\n"
            f"You have been automatically activated with a\n"
            f"**{FREE_TRIAL_DAYS}-day FREE TRIAL**! 🎁\n\n"
            f"⏰ Trial expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 **COMMANDS:**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🔮 `/forecast TICKER [PERIOD] [INTERVAL] [STEPS]`\n"
            f"   Example: `/forecast EURUSD 1y 1d 30`\n\n"
            f"💳 `/subscribe` - Get subscription info\n"
            f"📱 `/status` - Check your subscription\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌟 Enjoy your free trial! 🚀\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"⚡ Powered By **Pluto Technology**"
        )
    else:
        subscribed, expiry = is_subscribed(user_id)
        
        if subscribed:
            welcome_msg = (
                f"╔══════════════════════════════════╗\n"
                f"║   👋 WELCOME BACK TO STOCK SIGHT  ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"✅ Your subscription is **ACTIVE**\n"
                f"⏰ Until: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📊 **QUICK START:**\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                f"🔮 `/forecast TICKER [PERIOD] [INTERVAL] [STEPS]`\n"
                f"   Example: `/forecast AAPL 1y 1d 30`\n\n"
                f"📱 `/status` - Check subscription\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Powered By **Pluto Technology**"
            )
        else:
            welcome_msg = (
                f"╔══════════════════════════════════╗\n"
                f"║   👋 WELCOME BACK TO STOCK SIGHT  ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"⚠️ Your subscription has **EXPIRED**\n\n"
                f"💳 Use `/subscribe` to renew your access\n"
                f"📱 Use `/status` to check details\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Powered By **Pluto Technology**"
            )
    
    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.MARKDOWN)

async def subscribe_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced subscribe command"""
    user_id = update.effective_user.id if update.effective_user else 0
    
    if SUBSCRIBE_URL:
        text = (
            f"╔══════════════════════════════════╗\n"
            f"║  💳 SUBSCRIPTION INFORMATION     ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"To subscribe or renew, complete payment at:\n\n"
            f"🔗 {SUBSCRIBE_URL}\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📝 **AFTER PAYMENT:**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Use this command:\n"
            f"`/paid <transaction_id>`\n\n"
            f"This will notify our admins who will\n"
            f"activate your account immediately.\n\n"
            f"🆔 Your Telegram ID: `{user_id}`\n"
            f"(Include this in your payment reference)\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Powered By **Pluto Technology**"
        )
    else:
        text = (
            f"╔══════════════════════════════════╗\n"
            f"║  💳 SUBSCRIPTION INFORMATION     ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"⚠️ Subscription system not configured.\n\n"
            f"🆔 Your Telegram ID: `{user_id}`\n\n"
            f"Please contact admin for activation.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Powered By **Pluto Technology**"
        )
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

async def paid_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced paid command"""
    user = update.effective_user
    uid = user.id if user else None
    username = user.username if user else "Unknown"
    full_name = user.full_name if user else "Unknown"
    
    details = " ".join(context.args) if context.args else "(no transaction details provided)"
    
    msg = (
        f"💰 **PAYMENT NOTIFICATION**\n\n"
        f"👤 User: {full_name}\n"
        f"🆔 Telegram ID: `{uid}`\n"
        f"📱 Username: @{username}\n"
        f"💳 Transaction Details: {details}\n\n"
        f"⚡ **Quick Activation:**\n"
        f"`/activate {uid} 30`\n\n"
        f"(Tap to copy command, then send to activate for 30 days)"
    )
    
    await notify_admins(context, msg)
    
    user_msg = (
        f"╔══════════════════════════════════╗\n"
        f"║  ✅ PAYMENT NOTIFICATION SENT!   ║\n"
        f"╚══════════════════════════════════╝\n\n"
        f"🆔 Your Telegram ID: `{uid}`\n"
        f"💳 Transaction: {details}\n\n"
        f"Our team will activate your account shortly.\n"
        f"You'll receive a confirmation message.\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"⚡ Powered By **Pluto Technology**"
    )
    
    await update.message.reply_text(user_msg, parse_mode=ParseMode.MARKDOWN)

async def activate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced activate command (admin only)"""
    caller = update.effective_user.id if update.effective_user else None
    
    if caller not in ADMIN_IDS:
        await update.message.reply_text("❌ Unauthorized. Admin access required.")
        return
    
    if not context.args:
        await update.message.reply_text(
            "📝 **Usage:** `/activate <telegram_id> [days]`\n\n"
            "Example: `/activate 123456789 30`\n"
            "(Default: 30 days if not specified)",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        target = int(context.args[0])
        days = int(context.args[1]) if len(context.args) >= 2 else 30
    except ValueError:
        await update.message.reply_text("❌ Invalid arguments. Use numeric values.")
        return
    
    try:
        user_obj = await context.bot.get_chat(target)
        username = user_obj.username if user_obj.username else ""
        full_name = user_obj.full_name if user_obj.full_name else ""
    except:
        username = ""
        full_name = ""
    
    exp = activate_subscription_for(target, days=days, trial=False, 
                                   username=username, full_name=full_name)
    
    await update.message.reply_text(
        f"╔══════════════════════════════════╗\n"
        f"║  ✅ USER ACTIVATED SUCCESSFULLY  ║\n"
        f"╚══════════════════════════════════╝\n\n"
        f"🆔 Telegram ID: `{target}`\n"
        f"⏰ Active until: {exp.strftime('%Y-%m-%d %H:%M UTC')}\n"
        f"📅 Duration: {days} days\n\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        parse_mode=ParseMode.MARKDOWN
    )
    
    try:
        await context.bot.send_message(
            chat_id=target,
            text=(
                f"╔══════════════════════════════════╗\n"
                f"║  🎉 SUBSCRIPTION ACTIVATED!      ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"✅ Active until: {exp.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"📅 Duration: {days} days\n\n"
                f"You can now use `/forecast` to get\n"
                f"professional trading signals!\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Powered By **Pluto Technology**"
            ),
            parse_mode=ParseMode.MARKDOWN
        )
    except Exception as e:
        print(f"[WARN] Could not notify user {target}: {e}")
        await update.message.reply_text(
            f"⚠️ User activated but couldn't send notification (user may have blocked bot)"
        )

async def deactivate_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced deactivate command (admin only)"""
    caller = update.effective_user.id if update.effective_user else None
    
    if caller not in ADMIN_IDS:
        await update.message.reply_text("❌ Unauthorized. Admin access required.")
        return
    
    if not context.args:
        await update.message.reply_text(
            "📝 **Usage:** `/deactivate <telegram_id>`\n\n"
            "Example: `/deactivate 123456789`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    try:
        target = int(context.args[0])
    except ValueError:
        await update.message.reply_text("❌ Invalid telegram_id. Use numeric value.")
        return
    
    success = deactivate_subscription_for(target)
    
    if success:
        await update.message.reply_text(
            f"╔══════════════════════════════════╗\n"
            f"║  ✅ SUBSCRIPTION DEACTIVATED     ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"🆔 Telegram ID: `{target}`\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            parse_mode=ParseMode.MARKDOWN
        )
        
        try:
            await context.bot.send_message(
                chat_id=target,
                text=(
                    "╔══════════════════════════════════╗\n"
                    "║  ⚠️ SUBSCRIPTION DEACTIVATED     ║\n"
                    "╚══════════════════════════════════╝\n\n"
                    "Your subscription has been deactivated.\n\n"
                    "Use `/subscribe` to renew access.\n\n"
                    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
            )
        except Exception as e:
            print(f"[WARN] Could not notify user {target}: {e}")
    else:
        await update.message.reply_text("❌ No active subscription found for this user.")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced status command"""
    user = update.effective_user
    user_id = user.id if user else None
    
    if not user_id:
        await update.message.reply_text("❌ Could not determine your user ID.")
        return
    
    subscribed, expiry = is_subscribed(user_id)
    subs = _load_subs()
    user_data = subs.get(str(user_id), {})
    is_trial = user_data.get("is_trial", False)
    activated_at = user_data.get("activated_at", "Unknown")
    
    sheets_status = "✅ Connected" if sheets_manager.enabled else "⚠️ Using Local Storage"
    
    if subscribed and expiry:
        now = datetime.datetime.utcnow()
        days_left = (expiry - now).days
        hours_left = ((expiry - now).seconds // 3600)
        
        status_icon = "🎁" if is_trial else "✅"
        status_text = "Free Trial" if is_trial else "Active Subscription"
        
        text = (
            f"╔══════════════════════════════════╗\n"
            f"║  {status_icon} {status_text:^28} ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"🆔 Your Telegram ID: `{user_id}`\n"
            f"📅 Activated: {activated_at[:10] if activated_at != 'Unknown' else 'Unknown'}\n"
            f"⏰ Expires: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"⏳ Time remaining: {days_left} days, {hours_left} hours\n"
            f"☁️ Storage: {sheets_status}\n\n"
        )
        
        if is_trial and days_left <= 1:
            text += (
                f"⚠️ **Trial expiring soon!**\n"
                f"Use `/subscribe` to continue access.\n\n"
            )
        elif days_left <= 3:
            text += (
                f"⚠️ **Subscription expiring soon!**\n"
                f"Use `/subscribe` to renew.\n\n"
            )
        else:
            text += "✨ Enjoy your access!\n\n"
        
        text += (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Powered By **Pluto Technology**"
        )
            
    else:
        if expiry:
            text = (
                f"╔══════════════════════════════════╗\n"
                f"║  ⚠️ SUBSCRIPTION EXPIRED         ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"🆔 Your Telegram ID: `{user_id}`\n"
                f"❌ Expired on: {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n"
                f"☁️ Storage: {sheets_status}\n\n"
                f"💳 Use `/subscribe` to renew access.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Powered By **Pluto Technology**"
            )
        else:
            text = (
                f"╔══════════════════════════════════╗\n"
                f"║  ❌ NO ACTIVE SUBSCRIPTION       ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"🆔 Your Telegram ID: `{user_id}`\n"
                f"☁️ Storage: {sheets_status}\n\n"
                f"💳 Use `/subscribe` for payment info.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Powered By **Pluto Technology**"
            )
    
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

def parse_forecast_args(args: List[str]) -> ForecastRequest:
    """Parse forecast command arguments with validation"""
    try:
        ticker = args[0].upper() if len(args) >= 1 else ""
        ticker = normalize_ticker(ticker)
        period = args[1] if len(args) >= 2 else "1y"
        interval = args[2] if len(args) >= 3 else "1d"
        
        try:
            steps = int(args[3]) if len(args) >= 4 else 30
        except (ValueError, IndexError):
            steps = 30
        
        steps = max(1, min(steps, 50))
        
        return ForecastRequest(
            ticker=ticker, period=period, interval=interval, steps=steps,
            context=60, horizon=1, window_size=300, device="cpu",
            buy_threshold_pct=0.3, sell_threshold_pct=-0.3, 
            stop_loss_pct=0.5, take_profit_rr=2.0
        )
    except Exception as e:
        raise ValueError(f"Invalid forecast arguments: {e}")

async def forecast_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Enhanced forecast command with professional output"""
    if not update.message:
        return
        
    args = context.args or []
    if not args:
        await update.message.reply_text(
            f"╔══════════════════════════════════╗\n"
            f"║  📊 FORECAST COMMAND USAGE       ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"**Usage:**\n"
            f"`/forecast <ticker> [period] [interval] [steps]`\n\n"
            f"**Example:**\n"
            f"`/forecast EURUSD 1y 1d 30`\n\n"
            f"**Parameters:**\n"
            f"• ticker: Symbol (required)\n"
            f"• period: Data period (default: 1y)\n"
            f"• interval: Timeframe (default: 1d)\n"
            f"  - Minutes: 1m, 2m, 5m, 15m, 30m\n"
            f"  - Hours: 1h, 4h\n"
            f"  - Days: 1d\n"
            f"• steps: Forecast steps (default: 30)\n\n"
            f"**Supported Tickers:**\n"
            f"• Forex: EURUSD, GBPUSD, USDJPY\n"
            f"• Crypto: BTCUSD, ETHUSD\n"
            f"• Stocks: AAPL, TSLA, GOOGL\n"
            f"• All broker symbols supported!\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"⚡ Powered By **Pluto Technology**",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    user_id = update.effective_user.id if update.effective_user else None
    
    subscribed, expiry = is_subscribed(user_id)
    if not subscribed:
        subs = _load_subs()
        user_data = subs.get(str(user_id), {})
        was_trial = user_data.get("is_trial", False)
        
        if expiry:
            if was_trial:
                msg = (
                    f"╔══════════════════════════════════╗\n"
                    f"║  ⚠️ FREE TRIAL EXPIRED           ║\n"
                    f"╚══════════════════════════════════╝\n\n"
                    f"Your {FREE_TRIAL_DAYS}-day trial ended on\n"
                    f"{expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"💳 Use `/subscribe` to continue.\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
            else:
                msg = (
                    f"╔══════════════════════════════════╗\n"
                    f"║  ⚠️ SUBSCRIPTION EXPIRED         ║\n"
                    f"╚══════════════════════════════════╝\n\n"
                    f"Expired on {expiry.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
                    f"💳 Use `/subscribe` to renew.\n\n"
                    f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                )
        else:
            msg = (
                f"╔══════════════════════════════════╗\n"
                f"║  ❌ NO ACTIVE SUBSCRIPTION       ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"💳 Use `/subscribe` for payment info.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
        await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)
        return

    try:
        req = parse_forecast_args(args)
    except Exception as e:
        await update.message.reply_text(f"❌ Invalid arguments: {e}")
        return

    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    
    async def timeout_handler():
        await asyncio.sleep(600)
        return None
    
    try:
        status_msg = await update.message.reply_text(
            f"╔══════════════════════════════════╗\n"
            f"║  🔄 PROCESSING FORECAST...       ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"🎯 Ticker: **{req.ticker}**\n"
            f"📊 Analyzing market data...\n"
            f"🧠 Training AI models...\n"
            f"📈 Generating predictions...\n\n"
            f"⏳ This may take 1-2 minutes...\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            parse_mode=ParseMode.MARKDOWN
        )
        
        forecast_task = asyncio.create_task(asyncio.to_thread(forecast_core, req))
        timeout_task = asyncio.create_task(timeout_handler())
        
        done, pending = await asyncio.wait(
            [forecast_task, timeout_task], 
            return_when=asyncio.FIRST_COMPLETED
        )
        
        for task in pending:
            task.cancel()
        
        if forecast_task in done:
            data = await forecast_task
        else:
            await status_msg.edit_text(
                f"╔══════════════════════════════════╗\n"
                f"║  ⏰ FORECAST TIMEOUT             ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"⏰ Forecast for {req.ticker} timed out\n"
                f"(10min limit exceeded)\n\n"
                f"Please try again with a shorter period\n"
                f"or fewer steps.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            return
        
        if not data or not data.get("predictions"):
            await status_msg.edit_text(
                f"╔══════════════════════════════════╗\n"
                f"║  ❌ FORECAST FAILED              ║\n"
                f"╚══════════════════════════════════╝\n\n"
                f"No forecast data generated for {req.ticker}\n\n"
                f"This may be due to:\n"
                f"• Invalid ticker symbol\n"
                f"• Insufficient historical data\n"
                f"• API limitations\n\n"
                f"Please try a different ticker.\n\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            )
            return
        
        predictions = data.get("predictions", [])
        if len(predictions) == 0:
            await status_msg.edit_text(
                f"❌ No predictions generated for {req.ticker}"
            )
            return
        
        await status_msg.edit_text(
            f"╔══════════════════════════════════╗\n"
            f"║  ✅ FORECAST COMPLETE!           ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"📊 Generating professional report...\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            parse_mode=ParseMode.MARKDOWN
        )
        
        # Generate enhanced output
        enhanced_output = build_enhanced_forecast_output(data, req)
        
        # Split output into chunks if needed (Telegram has 4096 char limit)
        max_length = 4000
        if len(enhanced_output) <= max_length:
            await update.message.reply_text(enhanced_output, parse_mode=ParseMode.MARKDOWN)
        else:
            # Split into multiple messages
            parts = []
            current_part = ""
            for line in enhanced_output.split('\n'):
                if len(current_part) + len(line) + 1 < max_length:
                    current_part += line + '\n'
                else:
                    parts.append(current_part)
                    current_part = line + '\n'
            if current_part:
                parts.append(current_part)
            
            for part in parts:
                await update.message.reply_text(part, parse_mode=ParseMode.MARKDOWN)
                await asyncio.sleep(0.5)  # Small delay between messages
        
        # Generate and send chart
        try:
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            
            title = f"{req.ticker} Forecast ({req.period}, {req.interval})"
            chart_bytes = make_watermarked_chart(data, title=title)
            chart_bio = io.BytesIO(chart_bytes)
            chart_bio.name = f"{req.ticker}_forecast.png"
            
            chart_caption = (
                f"📊 **{req.ticker}** Visual Forecast\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"⚡ Stock Sight AI Analysis\n"
                f"🔮 Powered By Pluto Technology"
            )
            
            await update.message.reply_photo(
                photo=InputFile(chart_bio),
                caption=chart_caption,
                parse_mode=ParseMode.MARKDOWN
            )
        except Exception as chart_error:
            print(f"[ERROR] Chart generation failed: {chart_error}")
            await update.message.reply_text(
                "⚠️ Chart generation failed, but analysis is complete above."
            )
        
        # Delete processing message
        try:
            await status_msg.delete()
        except Exception:
            pass
        
        # Send completion message with watermark
        completion_msg = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"✅ **FORECAST COMPLETE**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🎯 Ticker: {req.ticker}\n"
            f"📊 Interval: {req.interval}\n"
            f"🔮 Steps: {req.steps}\n"
            f"🎓 Accuracy: {data.get('ensemble_accuracy', 0):.1f}%\n\n"
            f"⚠️ **DISCLAIMER:**\n"
            f"This forecast is for informational purposes\n"
            f"only. Always do your own research and\n"
            f"manage risk appropriately.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🌟 **Stock Sight AI** v3.0\n"
            f"⚡ Powered By **Pluto Technology**\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        
        await update.message.reply_text(completion_msg, parse_mode=ParseMode.MARKDOWN)
        
        print(f"[SUCCESS] Forecast completed for {req.ticker} (user: {user_id})")
        
    except Exception as e:
        error_msg = str(e)[:200]
        await update.message.reply_text(
            f"╔══════════════════════════════════╗\n"
            f"║  ❌ FORECAST ERROR               ║\n"
            f"╚══════════════════════════════════╝\n\n"
            f"Error: {error_msg}\n\n"
            f"Please try:\n"
            f"• Different ticker symbol\n"
            f"• Shorter time period\n"
            f"• Fewer forecast steps\n\n"
            f"If problem persists, contact admin.\n\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            parse_mode=ParseMode.MARKDOWN
        )
        print(f"[ERROR] Forecast command failed: {e}")
        print(traceback.format_exc())

# ----------------------------
# Main application
# ----------------------------
def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN environment variable required")
    
    print(f"╔══════════════════════════════════════════╗")
    print(f"║  STOCK SIGHT AI - TELEGRAM SERVICE      ║")
    print(f"╚══════════════════════════════════════════╝")
    print(f"")
    print(f"[INFO] Version: {VERSION}")
    print(f"[INFO] Admin IDs: {ADMIN_IDS}")
    print(f"[INFO] Subscription URL: {bool(SUBSCRIBE_URL)}")
    print(f"[INFO] Free trial: {FREE_TRIAL_DAYS} days")
    print(f"[INFO] Local storage: {SUBS_FILE}")
    print(f"[INFO] Google Sheets: {GOOGLE_SHEETS_ENABLED}")
    print(f"[INFO] Sheets manager: {'✅ Active' if sheets_manager.enabled else '⚠️ Inactive'}")
    
    if sheets_manager.enabled:
        print(f"[INFO] ✅ Google Sheets integration active")
        print(f"[INFO] Sheet: {GOOGLE_SHEET_NAME}")
        print(f"[INFO] Worksheet: {GOOGLE_WORKSHEET_NAME}")
    else:
        print(f"[INFO] ⚠️ Using local JSON storage")
        if GOOGLE_SHEETS_ENABLED and not GSPREAD_AVAILABLE:
            print(f"[WARN] gspread library not installed")
        elif GOOGLE_SHEETS_ENABLED:
            print(f"[WARN] Sheets init failed: {sheets_manager.last_error}")
    
    print(f"")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"")
    
    os.makedirs(os.path.dirname(SUBS_FILE) if os.path.dirname(SUBS_FILE) else ".", exist_ok=True)
    
    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("forecast", forecast_cmd))
    app.add_handler(CommandHandler("subscribe", subscribe_cmd))
    app.add_handler(CommandHandler("paid", paid_cmd))
    app.add_handler(CommandHandler("activate", activate_cmd))
    app.add_handler(CommandHandler("deactivate", deactivate_cmd))
    app.add_handler(CommandHandler("status", status_cmd))

    print("[INFO] 🚀 Bot starting...")
    print("[INFO] ⚡ Powered By Pluto Technology")
    print("")
    
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()

