"""
ML Classifier for Data Discovery Agent.

This module implements ML classifiers (LightGBM, SVM) for structured data
classification to identify sensitive information in databases.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Union, Tuple, Optional
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

logger = logging.getLogger(__name__)

class MLClassifier:
    """
    ML Classifier for structured data classification.
    Uses LightGBM and SVM for identifying sensitive data in structured formats.
    """
    
    def __init__(self, classifier_type: str = "lightgbm"):
        """
        Initialize ML classifier for structured data.
        
        Args:
            classifier_type: Type of classifier to use ('lightgbm' or 'svm')
        """
        self.classifier_type = classifier_type
        self.model = None
        self.feature_names = []
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
        # Initialize the classifier
        self._initialize_classifier()
    
    def _initialize_classifier(self):
        """Initialize the ML classifier based on the specified type."""
        if self.classifier_type == "lightgbm":
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,  # Use all available cores
                    verbose=-1  # Suppress LightGBM output
                ))
            ])
        elif self.classifier_type == "svm":
            self.model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', SVC(
                    kernel='rbf',
                    C=1.0,
                    probability=True,
                    random_state=42
                ))
            ])
        else:
            raise ValueError(f"Unsupported classifier type: {self.classifier_type}")
    
    def train(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        grid_search: bool = False,
        cv: int = 3,
        scoring: str = 'f1_weighted',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """
        Train the classifier on structured data with optional grid search.
        
        Args:
            X: Feature dataframe
            y: Target labels
            grid_search: Whether to perform grid search for hyperparameters
            cv: Number of cross-validation folds for grid search
            scoring: Scoring metric for grid search
            n_jobs: Number of jobs to run in parallel (-1 means using all processors)
            
        Returns:
            Dictionary with training metrics and best parameters if grid_search=True
        """
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if grid_search:
            # Define parameter grid based on classifier type
            if self.classifier_type == "lightgbm":
                param_grid = {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.01, 0.05, 0.1],
                    'classifier__num_leaves': [15, 31, 63],
                    'classifier__min_child_samples': [20, 50, 100]
                }
            elif self.classifier_type == "svm":
                param_grid = {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__gamma': ['scale', 'auto'],
                    'classifier__kernel': ['rbf', 'poly']
                }
            
            # Create grid search with cross-validation
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
            grid_search = GridSearchCV(
                estimator=self.model,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                n_jobs=n_jobs,
                verbose=1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Update model with best estimator
            self.model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            # Train the model with early stopping
            if self.classifier_type == "lightgbm":
                self.model.fit(
                    X_train, y_train,
                    classifier__eval_set=[(X_val, y_val)],
                    classifier__eval_metric='logloss',
                    classifier__early_stopping_rounds=10,
                    classifier__verbose=0
                )
            else:
                self.model.fit(X_train, y_train)
            best_params = None
        
        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        y_pred_proba = self.model.predict_proba(X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_val, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_val, y_pred, average='weighted', zero_division=0)
        }
        
        # Add feature importance and best parameters if available
        if hasattr(self.model.named_steps['classifier'], 'feature_importances_'):
            importances = self.model.named_steps['classifier'].feature_importances_
            metrics["feature_importances"] = dict(zip(self.feature_names, importances.tolist()))
        
        # Add best parameters if grid search was performed
        if best_params is not None:
            metrics["best_params"] = best_params
            metrics["best_score"] = grid_search.best_score_
        
        return metrics
    
    def predict(self, data: Union[pd.DataFrame, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Predict sensitivity classification for structured data.
        
        Args:
            data: Structured data as DataFrame or dictionary
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Convert dictionary to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        
        # Ensure all required features are present
        for feature in self.feature_names:
            if feature not in data.columns:
                data[feature] = 0  # Default value for missing features
        
        # Make prediction
        prediction_proba = self.model.predict_proba(data[self.feature_names])
        prediction = self.model.predict(data[self.feature_names])
        
        # Get confidence score (probability of the predicted class)
        confidence = np.max(prediction_proba, axis=1)[0]
        
        return {
            "classification": "sensitive" if prediction[0] == 1 else "non-sensitive",
            "confidence": float(confidence),
            "contains_sensitive_data": bool(prediction[0] == 1)
        }
    
    def analyze_column(self, column_data: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Analyze a database column to determine if it contains sensitive information.
        
        Args:
            column_data: Series containing column data
            column_name: Name of the column
            
        Returns:
            Dictionary with analysis results
        """
        # Extract features from the column
        features = self._extract_column_features(column_data, column_name)
        
        # Make prediction using extracted features
        result = self.predict(features)
        
        # Add additional information
        result["column_name"] = column_name
        result["sample_values"] = column_data.head(3).tolist()
        result["data_type"] = str(column_data.dtype)
        
        return result
    
    def _extract_column_features(self, column_data: pd.Series, column_name: str) -> Dict[str, Any]:
        """
        Extract features from a database column for sensitivity classification.
        
        Args:
            column_data: Series containing column data
            column_name: Name of the column
            
        Returns:
            Dictionary of extracted features
        """
        # Extract basic statistical features
        features = {
            "name_contains_sensitive_keyword": any(keyword in column_name.lower() for keyword in [
                "ssn", "social", "security", "password", "credit", "card", "cvv", "secret",
                "address", "phone", "email", "birth", "dob", "gender", "race", "ethnic",
                "income", "salary", "health", "medical", "account", "license"
            ]),
            "unique_ratio": len(column_data.unique()) / len(column_data) if len(column_data) > 0 else 0,
            "null_ratio": column_data.isnull().mean(),
            "is_numeric": pd.api.types.is_numeric_dtype(column_data),
            "is_string": pd.api.types.is_string_dtype(column_data),
            "is_datetime": pd.api.types.is_datetime64_dtype(column_data),
            "avg_string_length": column_data.astype(str).str.len().mean() if pd.api.types.is_string_dtype(column_data) else 0,
        }
        
        # Add pattern-based features for string columns
        if pd.api.types.is_string_dtype(column_data):
            sample = column_data.dropna().astype(str).sample(min(100, len(column_data))).tolist()
            
            # Check for common patterns in the sample
            features.update({
                "contains_email_pattern": any('@' in str(x) and '.' in str(x).split('@')[-1] for x in sample),
                "contains_phone_pattern": any(len(str(x).replace('-', '').replace('(', '').replace(')', '').replace(' ', '')) >= 10 and 
                                            str(x).replace('-', '').replace('(', '').replace(')', '').replace(' ', '').isdigit() for x in sample),
                "contains_ssn_pattern": any(len(str(x).replace('-', '')) == 9 and str(x).replace('-', '').isdigit() for x in sample),
                "contains_credit_card_pattern": any(len(str(x).replace('-', '').replace(' ', '')) >= 15 and 
                                                str(x).replace('-', '').replace(' ', '').isdigit() for x in sample),
                "contains_address_pattern": any(('street' in str(x).lower() or 'ave' in str(x).lower() or 
                                              'road' in str(x).lower() or 'dr' in str(x).lower()) for x in sample)
            })
        
        return features
