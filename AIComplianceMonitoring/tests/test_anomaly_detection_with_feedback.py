#!/usr/bin/env python
"""
Integration Test Script for Anomaly Detection with RL Feedback Loop

This script tests the entire anomaly detection pipeline including:
1. Data loading and preprocessing 
2. IsolationForest model detection
3. Autoencoder model detection
4. Validation through the RL feedback system
5. Alert generation with feedback integration

Using the HR Records dataset for realistic testing scenarios.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('anomaly_detection_test.log')
    ]
)
logger = logging.getLogger('anomaly_detection_test')

# Add the project root to the Python path so we can import modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# Test Configuration
class TestConfig:
    """Configuration for the test run"""
    
    # Paths
    DATA_PATH = Path("AIComplianceMonitoring/data/5000000 HRA Records.csv")
    MODEL_SAVE_DIR = Path("AIComplianceMonitoring/models")
    RESULTS_DIR = Path("AIComplianceMonitoring/test_results")
    
    # Test parameters
    SAMPLE_SIZE = 5000                  # Number of records to sample for testing
    ANOMALY_INJECTION_RATE = 0.05       # Percentage of anomalies to inject (5%)
    TEST_SPLIT = 0.3                    # Portion of data to use for testing
    RANDOM_SEED = 42                    # Random seed for reproducibility
    
    # IsolationForest parameters
    IF_CONTAMINATION = 0.05             # IsolationForest contamination parameter
    IF_N_ESTIMATORS = 100               # Number of estimators for IsolationForest
    
    # Autoencoder parameters
    AE_EPOCHS = 20                      # Autoencoder training epochs (reduced for faster testing)
    AE_BATCH_SIZE = 64                  # Autoencoder batch size
    AE_THRESHOLD_PERCENTILE = 95        # Percentile for reconstruction error threshold
    
    # Ensemble parameters
    IF_WEIGHT = 0.6                     # Weight for IsolationForest in ensemble
    AE_WEIGHT = 0.4                     # Weight for Autoencoder in ensemble
    ENSEMBLE_THRESHOLD = 0.6            # Threshold for binary predictions
    
    # Feedback loop parameters
    FEEDBACK_STORAGE = Path("feedback_data")
    VALIDATION_THRESHOLD = 0.7          # Confidence threshold for validation
    HUMAN_REVIEW_SAMPLE = 0.2           # Percentage of anomalies for simulated human review

# Create necessary directories
os.makedirs(TestConfig.MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TestConfig.RESULTS_DIR, exist_ok=True)
os.makedirs(TestConfig.FEEDBACK_STORAGE, exist_ok=True)

logger.info("Test configuration initialized")


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_hr_data(filepath, sample_size=None, random_state=42):
    """
    Load the HR dataset and return a sample if requested.
    
    Args:
        filepath: Path to the CSV file
        sample_size: Optional number of records to sample
        random_state: Random state for reproducibility
        
    Returns:
        DataFrame with HR data
    """
    logger.info(f"Loading data from {filepath}")
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} features")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=random_state)
            logger.info(f"Sampled {sample_size} records")
            
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def preprocess_hr_data(df):
    """
    Preprocess the HR data for anomaly detection.
    
    Args:
        df: DataFrame with HR data
        
    Returns:
        Tuple of (processed DataFrame, encoder dict, categorical columns)
    """
    logger.info("Preprocessing HR data")
    
    # Make a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Identify numeric and categorical columns
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df_processed.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    logger.info(f"Found {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns")
    
    # Handle categorical features - one-hot encoding
    encoders = {}
    for col in categorical_cols:
        # Create one-hot encoded columns
        one_hot = pd.get_dummies(df_processed[col], prefix=col, drop_first=False)
        
        # Store the mapping for later use
        encoders[col] = {i: val for i, val in enumerate(df_processed[col].unique())}
        
        # Replace the categorical column with one-hot encoded columns
        df_processed = pd.concat([df_processed, one_hot], axis=1)
        
    # Remove original categorical columns
    df_processed.drop(categorical_cols, axis=1, inplace=True)
    
    # Normalize numeric columns
    for col in numeric_cols:
        if df_processed[col].std() > 0:  # Avoid division by zero
            df_processed[col] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
    
    logger.info(f"Processed data shape: {df_processed.shape}")
    return df_processed, encoders, categorical_cols


def inject_anomalies(df, anomaly_rate=0.05, random_state=42):
    """
    Inject synthetic anomalies into the dataset for testing.
    
    Args:
        df: DataFrame with normal data
        anomaly_rate: Percentage of anomalies to inject
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with injected anomalies and anomaly labels
    """
    logger.info(f"Injecting {anomaly_rate:.1%} synthetic anomalies")
    np.random.seed(random_state)
    
    # Make a copy
    df_with_anomalies = df.copy()
    
    # Add an anomaly label column (0 = normal, 1 = anomaly)
    df_with_anomalies['is_anomaly'] = 0
    
    # Calculate number of anomalies to inject
    num_anomalies = int(len(df) * anomaly_rate)
    
    # Select random rows to inject anomalies
    anomaly_indices = np.random.choice(df.index, size=num_anomalies, replace=False)
    
    # Inject anomalies using different strategies
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for idx in anomaly_indices:
        # Choose anomaly type: extreme value (0), random value (1), or swap columns (2)
        anomaly_type = np.random.randint(0, 3)
        
        if anomaly_type == 0:  # Extreme value
            # Choose a random numeric column
            col = np.random.choice(numeric_cols)
            # Set an extreme value (mean + 5*std)
            mean, std = df[col].mean(), df[col].std()
            extreme = mean + (np.random.choice([-1, 1]) * 5 * std)
            df_with_anomalies.at[idx, col] = extreme
            
        elif anomaly_type == 1:  # Random value
            # Choose a random numeric column
            col = np.random.choice(numeric_cols)
            # Set a random value within the feature range but unusual
            min_val, max_val = df[col].min(), df[col].max()
            range_val = max_val - min_val
            random_val = np.random.uniform(max_val, max_val + range_val * 0.5)
            df_with_anomalies.at[idx, col] = random_val
            
        elif anomaly_type == 2:  # Swap columns
            # Choose two random numeric columns
            cols = np.random.choice(numeric_cols, size=2, replace=False)
            # Swap their values
            val1, val2 = df_with_anomalies.at[idx, cols[0]], df_with_anomalies.at[idx, cols[1]]
            df_with_anomalies.at[idx, cols[0]] = val2
            df_with_anomalies.at[idx, cols[1]] = val1
        
        # Mark as anomaly
        df_with_anomalies.at[idx, 'is_anomaly'] = 1
    
    logger.info(f"Injected {num_anomalies} anomalies into dataset")
    return df_with_anomalies


def split_data(df, test_size=0.3, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Extract 'is_anomaly' label if it exists
    if 'is_anomaly' in df.columns:
        X = df.drop('is_anomaly', axis=1)
        y = df['is_anomaly']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        train_df = X_train.copy()
        train_df['is_anomaly'] = y_train.values
        test_df = X_test.copy()
        test_df['is_anomaly'] = y_test.values
    else:
        # Simple split without stratification
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
    
    logger.info(f"Split data into training set ({len(train_df)} records) and testing set ({len(test_df)} records)")
    return train_df, test_df


# ============================================================================
# ANOMALY DETECTION MODELS
# ============================================================================

class IsolationForestModel:
    """Isolation Forest based anomaly detection model"""
    
    def __init__(self, contamination=0.05, random_state=42):
        """Initialize the model
        
        Args:
            contamination: Expected proportion of anomalies
            random_state: Random seed for reproducibility
        """
        from sklearn.ensemble import IsolationForest
        
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.contamination = contamination
        self.is_fitted = False
        logger.info(f"Initialized IsolationForest model with contamination={contamination}")
    
    def fit(self, X):
        """Fit the model to training data
        
        Args:
            X: Training data (DataFrame or numpy array)
        """
        logger.info(f"Training IsolationForest on {X.shape[0]} samples with {X.shape[1]} features")
        self.model.fit(X)
        self.is_fitted = True
        logger.info("IsolationForest training completed")
        return self
    
    def predict_anomaly_scores(self, X):
        """Predict anomaly scores for data
        
        Args:
            X: Test data (DataFrame or numpy array)
            
        Returns:
            Array of anomaly scores (higher means more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        # Get raw decision scores (negative = anomaly)
        raw_scores = -self.model.decision_function(X)  # Negate so higher = more anomalous
        
        # Scale to [0, 1] for consistency with other models
        # Use min-max scaling on the scores
        min_score, max_score = np.min(raw_scores), np.max(raw_scores)
        if max_score > min_score:  # Avoid division by zero
            scaled_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            scaled_scores = np.zeros_like(raw_scores)
            
        logger.info(f"IsolationForest generated anomaly scores for {len(X)} samples")
        return scaled_scores
    
    def save_model(self, path):
        """Save model to disk
        
        Args:
            path: Path to save the model
        """
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        logger.info(f"Saved IsolationForest model to {path}")
    
    def load_model(self, path):
        """Load model from disk
        
        Args:
            path: Path to load the model from
        """
        import joblib
        self.model = joblib.load(path)
        self.is_fitted = True
        logger.info(f"Loaded IsolationForest model from {path}")
        return self


class AutoencoderModel:
    """Autoencoder-based anomaly detection model using PyTorch"""
    
    def __init__(self, input_dim, threshold_percentile=95):
        """Initialize the PyTorch Autoencoder model
        
        Args:
            input_dim: Input dimension (number of features)
            threshold_percentile: Percentile for anomaly threshold
        """
        import torch
        import torch.nn as nn
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold_percentile = threshold_percentile
        self.threshold = None
        self.input_dim = input_dim
        self.model = None
        self._build_model()
        
        # Set for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
    
    def _build_model(self):
        """Create the PyTorch Autoencoder architecture"""
        import torch.nn as nn
        
        # Define architecture dimensions
        dim1 = int(self.input_dim * 0.75)
        dim2 = int(self.input_dim * 0.5)
        dim3 = int(self.input_dim * 0.33)
        bottleneck = int(self.input_dim * 0.25)
        
        # Create model
        self.model = nn.Sequential(
            # Encoder
            nn.Linear(self.input_dim, dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim3),
            nn.ReLU(),
            nn.Linear(dim3, bottleneck),
            nn.ReLU(),
            
            # Decoder
            nn.Linear(bottleneck, dim3),
            nn.ReLU(),
            nn.Linear(dim3, dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Linear(dim1, self.input_dim),
            nn.Sigmoid()
        ).to(self.device)
    
    def train(self, X, epochs=20, batch_size=64):
        """Train the autoencoder model
        
        Args:
            X: Training data
            epochs: Number of epochs to train
            batch_size: Batch size for training
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        
        # Convert data to PyTorch tensor
        # First ensure we have a numpy array (handles pandas DataFrame input)
        if hasattr(X, 'values'):
            X_numpy = X.values.astype('float32')
        else:
            X_numpy = np.array(X, dtype='float32')
            
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, X_tensor)  # Input = Target for autoencoder
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, targets in dataloader:
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
        
        # Calculate reconstruction error threshold on training data
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
            self.threshold = np.percentile(mse, self.threshold_percentile)
    
    def predict(self, X):
        """Predict anomaly scores for new data
        
        Args:
            X: New data to predict anomaly scores for
            
        Returns:
            np.ndarray: Anomaly scores (higher = more anomalous)
        """
        import torch
        
        # Convert to tensor (handle DataFrame input)
        if hasattr(X, 'values'):
            X_numpy = X.values.astype('float32')
        else:
            X_numpy = np.array(X, dtype='float32')
        X_tensor = torch.tensor(X_numpy, dtype=torch.float32).to(self.device)
        
        # Get reconstructions
        self.model.eval()
        with torch.no_grad():
            reconstructions = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructions) ** 2, dim=1).cpu().numpy()
        
        return mse
    
    def save(self, path):
        """Save the model to disk
        
        Args:
            path: Path to save the model to
        """
        import torch
        from pathlib import Path
        
        # Create the directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save the model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'threshold': self.threshold,
            'threshold_percentile': self.threshold_percentile,
            'input_dim': self.input_dim
        }, path)
    
    def load(self, path):
        """Load the model from disk
        
        Args:
            path: Path to load the model from
        """
        import torch
        
        # Load the checkpoint
        checkpoint = torch.load(path)
        
        # Restore parameters
        self.threshold = checkpoint['threshold']
        self.threshold_percentile = checkpoint['threshold_percentile']
        self.input_dim = checkpoint['input_dim']
        
        # Rebuild model and load weights
        self._build_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
# ============================================================================
# REINFORCEMENT LEARNING FEEDBACK INTEGRATION
# ============================================================================

class MockValidationAgent:
    """Mock validation agent for testing
    
    In a real system, this would be replaced by the actual ValidationAgent
    from the feedback_loop.py module.
    """
    
    def __init__(self, validation_threshold=0.7):
        """Initialize the mock validation agent
        
        Args:
            validation_threshold: Confidence threshold for validation
        """
        self.validation_threshold = validation_threshold
        logger.info(f"Initialized MockValidationAgent with threshold={validation_threshold}")
    
    def validate_anomaly(self, log_data, anomaly_score, features=None):
        """Validate if a detected anomaly is a true positive
        
        Args:
            log_data: The log data record
            anomaly_score: The anomaly score from detection model
            features: Optional feature vector
            
        Returns:
            Tuple of (is_validated, confidence)
        """
        # Simple validation logic based on score and features
        # In a real system, this would use more sophisticated logic
        
        # Get ground truth if available (for testing only)
        ground_truth = log_data.get('is_anomaly', None)
        
        if ground_truth is not None:
            # When we have ground truth, we'll use that but with some noise
            # to simulate imperfect validation
            if ground_truth == 1:  # True anomaly
                confidence = np.random.uniform(0.7, 1.0)  # High confidence for true anomalies
                is_validated = True
            else:  # Normal data point
                confidence = np.random.uniform(0.0, 0.3)  # Low confidence for normals
                is_validated = False
        else:
            # If no ground truth, base validation mostly on anomaly score
            confidence = min(anomaly_score * 1.2, 1.0)  # Boost score slightly
            is_validated = confidence > self.validation_threshold
        
        return is_validated, confidence


class MockHumanInteractionAgent:
    """Mock human interaction agent for testing
    
    In a real system, this would be replaced by the actual HumanInteractionAgent
    from the feedback_loop.py module.
    """
    
    def __init__(self, sample_rate=0.2):
        """Initialize the mock human interaction agent
        
        Args:
            sample_rate: Portion of anomalies to sample for human review
        """
        self.sample_rate = sample_rate
        self.reviewed_items = 0
        logger.info(f"Initialized MockHumanInteractionAgent with sample_rate={sample_rate}")
    
    def request_human_feedback(self, log_data, anomaly_score, validation_result):
        """Simulate requesting feedback from a human
        
        Args:
            log_data: The log data record
            anomaly_score: Anomaly score from detection model
            validation_result: Result from validation agent
            
        Returns:
            Dictionary with feedback
        """
        # Simulate human decision - in a real system this would be a queue
        # that gets processed by a human operator
        
        # Only process a sample of requests based on sample rate
        if np.random.random() > self.sample_rate:
            return None
            
        self.reviewed_items += 1
        
        # Get ground truth if available (for testing only)
        ground_truth = log_data.get('is_anomaly', None)
        
        if ground_truth is not None:
            # When we have ground truth, the "human" provides perfect feedback
            is_anomaly = bool(ground_truth)
            confidence = 1.0
            category = "Simulated" if is_anomaly else "Normal"
        else:
            # Without ground truth, simulate human decision based on score
            is_anomaly = anomaly_score > 0.7
            confidence = np.random.uniform(0.8, 1.0)  # Humans are usually confident
            category = np.random.choice(
                ["Access", "Financial", "Identity", "Resource", "Normal"], 
                p=[0.2, 0.2, 0.2, 0.2, 0.2]
            )
            if category == "Normal":
                is_anomaly = False
        
        feedback = {
            'timestamp': datetime.now().isoformat(),
            'log_id': str(hash(str(log_data))),
            'original_score': anomaly_score,
            'validation_result': validation_result,
            'is_anomaly': is_anomaly,
            'confidence': confidence,
            'category': category if is_anomaly else 'Normal',
            'notes': 'Simulated human feedback for testing'
        }
        
        logger.info(f"Human feedback received for log {feedback['log_id']}: {feedback['is_anomaly']}")
        return feedback


class MockFeedbackIntegration:
    """Mock feedback integration for model updates
    
    In a real system, this would be replaced by the actual FeedbackIntegration
    from the feedback_loop.py module.
    """
    
    def __init__(self, if_model, ae_model, update_threshold=10):
        """Initialize the mock feedback integrator
        
        Args:
            if_model: IsolationForest model
            ae_model: Autoencoder model
            update_threshold: Number of feedback items needed before update
        """
        self.if_model = if_model
        self.ae_model = ae_model
        self.update_threshold = update_threshold
        self.feedback_data = []
        self.performance_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'if_model_updates': 0,
            'ae_model_updates': 0
        }
        
        logger.info(f"Initialized MockFeedbackIntegration with update_threshold={update_threshold}")
    
    def add_feedback(self, feedback, log_data, feature_vector):
        """Add new feedback to the collection
        
        Args:
            feedback: Feedback from human
            log_data: Original log data
            feature_vector: Feature vector used for detection
        """
        if not feedback:
            return
            
        # Store feedback with features for later model updates
        self.feedback_data.append({
            'feedback': feedback,
            'feature_vector': feature_vector,
            'log_data': log_data
        })
        
        # Update performance metrics
        if 'is_anomaly' in log_data:
            ground_truth = bool(log_data['is_anomaly'])
            predicted = feedback['is_anomaly']
            
            if ground_truth and predicted:      # True positive
                self.performance_metrics['true_positives'] += 1
            elif ground_truth and not predicted:  # False negative
                self.performance_metrics['false_negatives'] += 1
            elif not ground_truth and predicted:  # False positive
                self.performance_metrics['false_positives'] += 1
            else:                                # True negative
                self.performance_metrics['true_negatives'] += 1
        
        # Calculate derived metrics
        tp = self.performance_metrics['true_positives']
        fp = self.performance_metrics['false_positives']
        tn = self.performance_metrics['true_negatives']
        fn = self.performance_metrics['false_negatives']
        
        # Avoid division by zero
        if tp + fp > 0:
            self.performance_metrics['precision'] = tp / (tp + fp)
        if tp + fn > 0:
            self.performance_metrics['recall'] = tp / (tp + fn)
        if self.performance_metrics['precision'] + self.performance_metrics['recall'] > 0:
            p = self.performance_metrics['precision']
            r = self.performance_metrics['recall']
            self.performance_metrics['f1_score'] = 2 * p * r / (p + r)
        
        logger.debug(f"Added feedback to collection. Total items: {len(self.feedback_data)}")
        
        # Check if we should update models
        if len(self.feedback_data) >= self.update_threshold:
            self.update_models()
    
    def update_models(self):
        """Update models based on collected feedback
        
        In a real system, this would fine-tune the models with reinforcement learning
        """
        if not self.feedback_data:
            return
            
        logger.info(f"Updating models with {len(self.feedback_data)} feedback items")
        
        # In a real system, we would retrain or fine-tune models here
        # For this test, we'll just log that it would happen
        
        # Extract current performance metrics for logging
        precision = self.performance_metrics['precision']
        recall = self.performance_metrics['recall']
        f1 = self.performance_metrics['f1_score']
        
        logger.info(f"Model update triggered. Current metrics: ")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        # Clear feedback data after update
        self.feedback_data = []
        
        return self.performance_metrics


# ============================================================================
# MAIN TEST FUNCTION
# ============================================================================

def run_anomaly_detection_test():
    """Run the full anomaly detection test with RL feedback loop"""
    test_start_time = datetime.now()
    logger.info(f"Starting anomaly detection test at {test_start_time}")
    
    # Step 1: Load and prepare data
    logger.info("===== STEP 1: LOAD AND PREPARE DATA =====")
    df = load_hr_data(TestConfig.DATA_PATH, sample_size=TestConfig.SAMPLE_SIZE)
    
    # Preprocess data
    df_processed, encoders, categorical_cols = preprocess_hr_data(df)
    
    # Inject synthetic anomalies
    df_with_anomalies = inject_anomalies(df_processed, 
                                        anomaly_rate=TestConfig.ANOMALY_INJECTION_RATE)
    
    # Split into train and test sets
    train_df, test_df = split_data(df_with_anomalies, test_size=TestConfig.TEST_SPLIT)
    
    # Extract features and labels
    X_train = train_df.drop('is_anomaly', axis=1)
    y_train = train_df['is_anomaly']
    X_test = test_df.drop('is_anomaly', axis=1)
    y_test = test_df['is_anomaly']
    
    logger.info(f"Data preparation complete. Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 2: Train models
    logger.info("\n===== STEP 2: TRAIN ANOMALY DETECTION MODELS =====")
    
    # IsolationForest model
    if_model = IsolationForestModel(
        contamination=TestConfig.IF_CONTAMINATION,
        random_state=TestConfig.RANDOM_SEED
    )
    logger.info(f"Initialized IsolationForest model with contamination={TestConfig.IF_CONTAMINATION}")
    
    logger.info(f"Training IsolationForest on {X_train.shape[0]} samples with {X_train.shape[1]} features")
    if_model.fit(X_train)
    logger.info("IsolationForest training completed")
    
    # Save trained IsolationForest model
    if_model.save_model(TestConfig.MODEL_SAVE_DIR / "isolation_forest_model.pkl")
    logger.info(f"Saved IsolationForest model to {TestConfig.MODEL_SAVE_DIR / 'isolation_forest_model.pkl'}")
    
    # Autoencoder model with PyTorch
    ae_model = AutoencoderModel(
        input_dim=X_train.shape[1],
        threshold_percentile=TestConfig.AE_THRESHOLD_PERCENTILE
    )
    logger.info(f"Initialized Autoencoder model with {X_train.shape[1]} input dimensions")
    
    logger.info(f"Training Autoencoder on {X_train.shape[0]} samples")
    ae_model.train(X_train, epochs=TestConfig.AE_EPOCHS, batch_size=TestConfig.AE_BATCH_SIZE)
    logger.info("Autoencoder training completed")
    
    # Save trained Autoencoder model
    ae_model.save(TestConfig.MODEL_SAVE_DIR / "autoencoder_model.pth")
    logger.info(f"Saved Autoencoder model to {TestConfig.MODEL_SAVE_DIR / 'autoencoder_model.pth'}")
    
    # Step 3: Set up feedback loop components
    logger.info("\n===== STEP 3: SETUP FEEDBACK LOOP COMPONENTS =====")
    
    validation_agent = MockValidationAgent(
        validation_threshold=TestConfig.VALIDATION_THRESHOLD
    )
    
    human_agent = MockHumanInteractionAgent(
        sample_rate=TestConfig.HUMAN_REVIEW_SAMPLE
    )
    
    feedback_integration = MockFeedbackIntegration(
        if_model=if_model,
        ae_model=ae_model,
        update_threshold=10  # Update after 10 feedback items
    )
    
    # Step 4: Run anomaly detection on test data and process through feedback loop
    logger.info("\n===== STEP 4: DETECT ANOMALIES AND PROCESS FEEDBACK =====")
    
    # Get anomaly predictions from both models
    logger.info("Generating anomaly predictions with IsolationForest")
    if_scores = if_model.predict_anomaly_scores(X_test)
    logger.info(f"IsolationForest anomaly scores generated for {len(if_scores)} test samples")
    
    logger.info("Generating anomaly predictions with Autoencoder")
    ae_scores = ae_model.predict(X_test)
    logger.info(f"Autoencoder anomaly scores generated for {len(ae_scores)} test samples")
    
    # Scale both scores to 0-1 range for ensemble
    if_scores_scaled = minmax_scale(if_scores)
    ae_scores_scaled = minmax_scale(ae_scores)
    
    # Calculate metrics for individual models and ensemble
    metrics = {
        'isolation_forest': {
            'auroc': roc_auc_score(y_test, if_scores_scaled),
            'precision': precision_score(y_test, (if_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'recall': recall_score(y_test, (if_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'f1_score': f1_score(y_test, (if_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
        },
        'autoencoder': {
            'auroc': roc_auc_score(y_test, ae_scores_scaled),
            'precision': precision_score(y_test, (ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'recall': recall_score(y_test, (ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'f1_score': f1_score(y_test, (ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
        },
        'ensemble': {
            'auroc': roc_auc_score(y_test, TestConfig.IF_WEIGHT * if_scores_scaled + TestConfig.AE_WEIGHT * ae_scores_scaled),
            'precision': precision_score(y_test, (TestConfig.IF_WEIGHT * if_scores_scaled + TestConfig.AE_WEIGHT * ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'recall': recall_score(y_test, (TestConfig.IF_WEIGHT * if_scores_scaled + TestConfig.AE_WEIGHT * ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'f1_score': f1_score(y_test, (TestConfig.IF_WEIGHT * if_scores_scaled + TestConfig.AE_WEIGHT * ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)),
            'confusion_matrix': confusion_matrix(y_test, (TestConfig.IF_WEIGHT * if_scores_scaled + TestConfig.AE_WEIGHT * ae_scores_scaled > TestConfig.ENSEMBLE_THRESHOLD).astype(int)).tolist()
        },
    }
    
    # Ensemble the predictions (weighted average)
    logger.info("Generating ensemble predictions")
    ensemble_scores = TestConfig.IF_WEIGHT * if_scores_scaled + TestConfig.AE_WEIGHT * ae_scores_scaled
    
    # Apply threshold to get binary predictions
    ensemble_preds = (ensemble_scores > TestConfig.ENSEMBLE_THRESHOLD).astype(int)
    logger.info(f"Generated ensemble predictions with {sum(ensemble_preds)} potential anomalies")
    
    # Process each test sample through the feedback loop
    validation_results = []
    human_feedback_count = 0
    
    for i in range(len(X_test)):
        # Extract log data and score
        log_data = test_df.iloc[i].to_dict()
        anomaly_score = ensemble_scores[i]
        feature_vector = X_test.iloc[i].values
        
        # Validate through the validation agent
        is_validated, confidence = validation_agent.validate_anomaly(
            log_data, anomaly_score, feature_vector
        )
        
        # Record validation results
        validation_results.append({
            'index': i,
            'true_label': log_data['is_anomaly'],
            'anomaly_score': anomaly_score,
            'is_validated': is_validated,
            'confidence': confidence
        })
        
        # For validated anomalies, request human feedback
        if is_validated:
            feedback = human_agent.request_human_feedback(
                log_data, anomaly_score, (is_validated, confidence)
            )
            
            if feedback:
                human_feedback_count += 1
                # Integrate feedback
                feedback_integration.add_feedback(feedback, log_data, feature_vector)
    
    # Step 5: Evaluate results
    logger.info("\n===== STEP 5: EVALUATE RESULTS =====")
    
    # Calculate detection metrics
    y_pred = np.array([1 if result['is_validated'] else 0 for result in validation_results])
    
    from sklearn.metrics import classification_report, precision_recall_fscore_support
    
    # Generate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary'
    )
    
    # Print classification report
    logger.info("Classification Report:")
    report = classification_report(y_test, y_pred)
    logger.info("\n" + report)
    
    # Print confusion matrix
    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\n{cm}")
    
    # Create results summary
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'human_feedback_count': human_feedback_count,
        'test_samples': len(X_test),
        'anomalies_injected': int(sum(y_test)),
    }
    
    # Add feedback integration metrics
    metrics.update(feedback_integration.performance_metrics)
    
    # Log summary metrics
    logger.info("\nResults Summary:")
    logger.info(f"Total test samples: {metrics['test_samples']}")
    logger.info(f"Anomalies injected: {metrics['anomalies_injected']}")
    logger.info(f"Human feedback received: {metrics['human_feedback_count']}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    
    # Save results to file
    results_file = TestConfig.RESULTS_DIR / f"results_{test_start_time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        import json
        json.dump(metrics, f, indent=2)
    
    # Visualize results if matplotlib is available
    try:
        # Plot ROC curves
        logger.info("Generating ROC curves for individual models and ensemble")
        plt.figure(figsize=(10, 8))
        
        # IsolationForest ROC
        if_fpr, if_tpr, _ = roc_curve(y_test, if_scores_scaled)
        if_roc_auc = auc(if_fpr, if_tpr)
        plt.plot(if_fpr, if_tpr, label=f'IsolationForest (AUC = {if_roc_auc:.3f})')
        
        # Autoencoder ROC
        ae_fpr, ae_tpr, _ = roc_curve(y_test, ae_scores_scaled)
        ae_roc_auc = auc(ae_fpr, ae_tpr)
        plt.plot(ae_fpr, ae_tpr, label=f'Autoencoder (AUC = {ae_roc_auc:.3f})')
        
        # Ensemble ROC
        ensemble_fpr, ensemble_tpr, _ = roc_curve(y_test, ensemble_scores)
        ensemble_roc_auc = auc(ensemble_fpr, ensemble_tpr)
        plt.plot(ensemble_fpr, ensemble_tpr, label=f'Ensemble (AUC = {ensemble_roc_auc:.3f})', linewidth=2)
        
        # Add diagonal reference line
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Save plot
        plt.savefig(TestConfig.RESULTS_DIR / f"roc_curve_{test_start_time.strftime('%Y%m%d_%H%M%S')}.png")
        logger.info(f"ROC curve saved to {TestConfig.RESULTS_DIR}")
        
        # Create confusion matrix visualization
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = [0, 1]
        plt.xticks(tick_marks, ['Normal', 'Anomaly'])
        plt.yticks(tick_marks, ['Normal', 'Anomaly'])
        
        # Add text annotations to confusion matrix
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Save confusion matrix plot
        plt.savefig(TestConfig.RESULTS_DIR / f"confusion_matrix_{test_start_time.strftime('%Y%m%d_%H%M%S')}.png")
        logger.info(f"Confusion matrix visualization saved to {TestConfig.RESULTS_DIR}")
        
    except Exception as e:
        logger.warning(f"Could not generate visualizations: {str(e)}")
    
    test_end_time = datetime.now()
    test_duration = (test_end_time - test_start_time).total_seconds() / 60.0  # in minutes
    
    logger.info(f"\nTest completed in {test_duration:.2f} minutes")
    logger.info(f"Results saved to {results_file}")
    
    return metrics


if __name__ == "__main__":
    try:
        metrics = run_anomaly_detection_test()
        print("\nTest completed successfully!")
        print(f"F1 Score: {metrics['f1']:.4f}")
    except Exception as e:
        logger.exception("Error running anomaly detection test")
        print(f"\nTest failed with error: {str(e)}")
        sys.exit(1)