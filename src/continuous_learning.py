# src/continuous_learning.py
"""
Continuous Learning System with Anti-Catastrophic Forgetting
Uses Experience Replay, Elastic Weight Consolidation, and Progressive Neural Networks
"""
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Dict, List, Tuple, Optional
import copy

class ExperienceReplayBuffer:
    """
    Store historical training samples to prevent catastrophic forgetting
    """
    def __init__(self, max_size: int = 10000, priority_sampling: bool = True):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)
        self.max_size = max_size
        self.priority_sampling = priority_sampling
        
    def add(self, X: np.ndarray, y: np.ndarray, loss: float = 1.0):
        """Add experience with priority based on loss"""
        self.buffer.append((X, y))
        self.priorities.append(loss)
        
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Sample from buffer with priority"""
        if len(self.buffer) == 0:
            return np.array([]), np.array([])
        
        batch_size = min(batch_size, len(self.buffer))
        
        if self.priority_sampling and len(self.priorities) > 0:
            # Priority-based sampling (higher loss = higher priority)
            priorities = np.array(self.priorities)
            priorities = priorities / (priorities.sum() + 1e-10)
            
            indices = np.random.choice(
                len(self.buffer),
                size=batch_size,
                replace=False,
                p=priorities
            )
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        
        X_batch = []
        y_batch = []
        
        for idx in indices:
            X, y = self.buffer[idx]
            X_batch.append(X)
            y_batch.append(y)
        
        return np.array(X_batch), np.array(y_batch)
    
    def get_all(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all experiences"""
        if len(self.buffer) == 0:
            return np.array([]), np.array([])
        
        X_all = []
        y_all = []
        
        for X, y in self.buffer:
            X_all.append(X)
            y_all.append(y)
        
        return np.array(X_all), np.array(y_all)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.priorities.clear()

class ElasticWeightConsolidation:
    """
    Elastic Weight Consolidation (EWC) to prevent catastrophic forgetting
    Penalizes changes to important weights
    """
    def __init__(self, model: nn.Module, dataloader, device='cpu', lambda_ewc: float = 0.4):
        self.model = model
        self.device = device
        self.lambda_ewc = lambda_ewc
        
        # Store important parameters
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        # Calculate Fisher Information Matrix
        self.fisher = self._calculate_fisher(dataloader)
    
    def _calculate_fisher(self, dataloader) -> Dict:
        """Calculate Fisher Information Matrix (diagonal approximation)"""
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        
        self.model.eval()
        
        for X, y in dataloader:
            try:
                X, y = X.to(self.device), y.to(self.device)
                
                # Forward pass
                self.model.zero_grad()
                output = self.model(X)
                
                # Calculate loss
                loss = nn.MSELoss()(output, y)
                
                # Backward pass
                loss.backward()
                
                # Accumulate squared gradients (Fisher Information)
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        fisher[n] += p.grad.pow(2)
                
            except Exception as e:
                print(f"[WARN] Fisher calculation batch failed: {e}")
                continue
        
        # Average over batches
        for n in fisher:
            fisher[n] /= len(dataloader)
        
        return fisher
    
    def penalty(self) -> torch.Tensor:
        """Calculate EWC penalty"""
        loss = 0.0
        
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                # EWC loss: sum of fisher * (theta - theta_old)^2
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        
        return self.lambda_ewc * loss

class OnlineLearningManager:
    """
    Manages continuous online learning with multiple anti-forgetting strategies
    """
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        replay_buffer_size: int = 5000,
        replay_ratio: float = 0.3,
        use_ewc: bool = True,
        ewc_lambda: float = 0.4,
        adaptation_rate: float = 0.1
    ):
        self.model = model
        self.device = device
        self.replay_buffer = ExperienceReplayBuffer(max_size=replay_buffer_size)
        self.replay_ratio = replay_ratio
        self.use_ewc = use_ewc
        self.ewc = None
        self.adaptation_rate = adaptation_rate
        
        # Performance tracking
        self.performance_history = []
        self.update_count = 0
        
        # Model snapshots for ensemble
        self.model_snapshots = []
        self.max_snapshots = 5
        
    def update_with_new_data(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        epochs: int = 5,
        batch_size: int = 32,
        lr: float = 0.001
    ) -> Dict:
        """
        Update model with new data while preventing catastrophic forgetting
        """
        try:
            self.model.train()
            
            # Convert to tensors
            X_new_tensor = torch.tensor(X_new.astype(np.float32)).to(self.device)
            y_new_tensor = torch.tensor(y_new.astype(np.float32)).to(self.device)
            
            # Create optimizer with lower learning rate for stability
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr * self.adaptation_rate)
            loss_fn = nn.MSELoss()
            
            total_loss = 0.0
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                batches = 0
                
                # Create mini-batches
                n_samples = len(X_new)
                indices = np.random.permutation(n_samples)
                
                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:i+batch_size]
                    X_batch = X_new_tensor[batch_indices]
                    y_batch = y_new_tensor[batch_indices]
                    
                    # === NEW DATA LOSS ===
                    optimizer.zero_grad()
                    pred = self.model(X_batch)
                    new_loss = loss_fn(pred, y_batch)
                    
                    # === REPLAY BUFFER LOSS ===
                    replay_loss = 0.0
                    if len(self.replay_buffer.buffer) > 0:
                        replay_batch_size = int(batch_size * self.replay_ratio)
                        X_replay, y_replay = self.replay_buffer.sample(replay_batch_size)
                        
                        if len(X_replay) > 0:
                            X_replay_tensor = torch.tensor(X_replay.astype(np.float32)).to(self.device)
                            y_replay_tensor = torch.tensor(y_replay.astype(np.float32)).to(self.device)
                            
                            pred_replay = self.model(X_replay_tensor)
                            replay_loss = loss_fn(pred_replay, y_replay_tensor)
                    
                    # === EWC PENALTY ===
                    ewc_loss = 0.0
                    if self.use_ewc and self.ewc is not None:
                        ewc_loss = self.ewc.penalty()
                    
                    # === COMBINED LOSS ===
                    total_batch_loss = new_loss + replay_loss + ewc_loss
                    
                    # Backward pass
                    total_batch_loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += total_batch_loss.item()
                    batches += 1
                
                if batches > 0:
                    epoch_loss /= batches
                    total_loss += epoch_loss
            
            # Add new experiences to replay buffer
            for i in range(len(X_new)):
                # Calculate individual loss for priority
                with torch.no_grad():
                    pred = self.model(X_new_tensor[i:i+1])
                    loss = loss_fn(pred, y_new_tensor[i:i+1]).item()
                
                self.replay_buffer.add(X_new[i], y_new[i], loss)
            
            self.update_count += 1
            
            # Create snapshot every N updates
            if self.update_count % 10 == 0:
                self._create_model_snapshot()
            
            # Update EWC after significant updates
            if self.use_ewc and self.update_count % 5 == 0:
                self._update_ewc(X_new_tensor, y_new_tensor)
            
            avg_loss = total_loss / epochs
            self.performance_history.append(avg_loss)
            
            return {
                'success': True,
                'avg_loss': float(avg_loss),
                'update_count': self.update_count,
                'replay_buffer_size': len(self.replay_buffer.buffer),
                'snapshots': len(self.model_snapshots)
            }
            
        except Exception as e:
            print(f"[ERROR] Online learning update failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _update_ewc(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor):
        """Update EWC fisher information"""
        try:
            # Create temporary dataloader
            dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
            
            # Recalculate Fisher Information
            self.ewc = ElasticWeightConsolidation(
                self.model, dataloader, self.device, lambda_ewc=0.4
            )
        except Exception as e:
            print(f"[WARN] EWC update failed: {e}")
    
    def _create_model_snapshot(self):
        """Save model snapshot for ensemble predictions"""
        try:
            snapshot = copy.deepcopy(self.model.state_dict())
            self.model_snapshots.append(snapshot)
            
            # Keep only recent snapshots
            if len(self.model_snapshots) > self.max_snapshots:
                self.model_snapshots.pop(0)
                
            print(f"[INFO] Model snapshot created (total: {len(self.model_snapshots)})")
        except Exception as e:
            print(f"[WARN] Snapshot creation failed: {e}")
    
    def ensemble_predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using ensemble of model snapshots"""
        try:
            if len(self.model_snapshots) == 0:
                # Use current model
                self.model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
                    pred = self.model(X_tensor)
                    return pred.cpu().numpy()
            
            # Ensemble predictions
            predictions = []
            
            # Current model
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
                pred = self.model(X_tensor)
                predictions.append(pred.cpu().numpy())
            
            # Snapshot models
            for snapshot in self.model_snapshots:
                temp_model = copy.deepcopy(self.model)
                temp_model.load_state_dict(snapshot)
                temp_model.eval()
                
                with torch.no_grad():
                    pred = temp_model(X_tensor)
                    predictions.append(pred.cpu().numpy())
            
            # Average predictions
            ensemble_pred = np.mean(predictions, axis=0)
            
            return ensemble_pred
            
        except Exception as e:
            print(f"[ERROR] Ensemble prediction failed: {e}")
            # Fallback to current model
            self.model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X.astype(np.float32)).to(self.device)
                pred = self.model(X_tensor)
                return pred.cpu().numpy()
    
    def evaluate_performance(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate current model performance"""
        try:
            self.model.eval()
            
            X_tensor = torch.tensor(X_test.astype(np.float32)).to(self.device)
            y_tensor = torch.tensor(y_test.astype(np.float32)).to(self.device)
            
            with torch.no_grad():
                pred = self.model(X_tensor)
                loss = nn.MSELoss()(pred, y_tensor).item()
            
            pred_np = pred.cpu().numpy()
            y_np = y_test
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((pred_np - y_np) ** 2))
            mae = np.mean(np.abs(pred_np - y_np))
            mape = np.mean(np.abs((pred_np - y_np) / (y_np + 1e-10))) * 100
            
            return {
                'loss': float(loss),
                'rmse': float(rmse),
                'mae': float(mae),
                'mape': float(mape),
                'samples_tested': len(X_test)
            }
            
        except Exception as e:
            print(f"[ERROR] Performance evaluation failed: {e}")
            return {'error': str(e)}
    
    def adaptive_learning_rate(self) -> float:
        """Calculate adaptive learning rate based on recent performance"""
        if len(self.performance_history) < 5:
            return 0.001
        
        recent_losses = self.performance_history[-5:]
        
        # If loss is increasing, reduce learning rate
        if recent_losses[-1] > recent_losses[0]:
            return 0.0005
        # If loss is decreasing, maintain or slightly increase
        else:
            return 0.001
    
    def should_update_model(self, recent_error: float) -> bool:
        """Decide if model should be updated based on recent performance"""
        if len(self.performance_history) < 10:
            return True  # Always update initially
        
        avg_historical_error = np.mean(self.performance_history[-50:])
        
        # Update if recent error is significantly higher
        if recent_error > avg_historical_error * 1.5:
            return True
        
        # Periodic update every N predictions
        if self.update_count % 20 == 0:
            return True
        
        return False
    
    def get_learning_statistics(self) -> Dict:
        """Get statistics about the learning process"""
        return {
            'total_updates': self.update_count,
            'replay_buffer_size': len(self.replay_buffer.buffer),
            'model_snapshots': len(self.model_snapshots),
            'avg_recent_loss': float(np.mean(self.performance_history[-10:])) if len(self.performance_history) >= 10 else 0.0,
            'loss_trend': 'improving' if self._is_improving() else 'stable' if len(self.performance_history) < 5 else 'degrading',
            'ewc_active': self.ewc is not None
        }
    
    def _is_improving(self) -> bool:
        """Check if model performance is improving"""
        if len(self.performance_history) < 10:
            return True
        
        recent = self.performance_history[-5:]
        older = self.performance_history[-10:-5]
        
        return np.mean(recent) < np.mean(older)

class AdaptiveEnsembleManager:
    """
    Manages adaptive weighting of ensemble models based on recent performance
    """
    def __init__(self, num_models: int = 4):
        self.num_models = num_models
        self.model_weights = np.ones(num_models) / num_models
        self.performance_history = [[] for _ in range(num_models)]
        
    def update_weights(self, predictions: List[np.ndarray], actual: np.ndarray):
        """Update model weights based on recent performance"""
        try:
            errors = []
            
            for pred in predictions:
                error = np.mean(np.abs(pred - actual))
                errors.append(error)
            
            # Track performance
            for i, error in enumerate(errors):
                self.performance_history[i].append(error)
                # Keep only recent history
                if len(self.performance_history[i]) > 100:
                    self.performance_history[i].pop(0)
            
            # Calculate weights (inverse of average error)
            if all(len(h) >= 5 for h in self.performance_history):
                avg_errors = [np.mean(h[-10:]) for h in self.performance_history]
                
                # Inverse error weighting
                inv_errors = [1.0 / (e + 1e-6) for e in avg_errors]
                total = sum(inv_errors)
                
                self.model_weights = np.array([w / total for w in inv_errors])
                
                # Ensure minimum weight (prevent complete exclusion)
                self.model_weights = np.maximum(self.model_weights, 0.05)
                self.model_weights = self.model_weights / self.model_weights.sum()
                
        except Exception as e:
            print(f"[WARN] Weight update failed: {e}")
    
    def ensemble_predict(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Weighted ensemble prediction"""
        try:
            if len(predictions) != self.num_models:
                # Fallback to simple average
                return np.mean(predictions, axis=0)
            
            # Weighted average
            weighted_pred = np.zeros_like(predictions[0])
            
            for i, pred in enumerate(predictions):
                weighted_pred += pred * self.model_weights[i]
            
            return weighted_pred
            
        except Exception as e:
            print(f"[ERROR] Ensemble prediction failed: {e}")
            return np.mean(predictions, axis=0)
    
    def get_weights(self) -> Dict:
        """Get current model weights"""
        return {
            f'model_{i}': float(w) for i, w in enumerate(self.model_weights)
        }
