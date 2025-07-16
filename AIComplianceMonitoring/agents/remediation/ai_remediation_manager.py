"""
AI-based Remediation Manager that uses machine learning to predict breach severity and recommend actions.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss


class AIResponseEngine:
    """Engine that handles AI-based response decisions."""
    
    def __init__(self):
        """Initialize the AI response engine."""
        self.breach_severity_model = None
        self.action_recommendation_model = None
        self.context_aware_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train_breach_severity_model(self, historical_data: pd.DataFrame):
        """Train model to predict breach severity."""
        # Prepare data
        X = historical_data.drop(['severity', 'description'], axis=1)
        y = historical_data['severity']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.breach_severity_model = RandomForestClassifier(n_estimators=100)
        self.breach_severity_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.breach_severity_model.predict(X_train)
        test_pred = self.breach_severity_model.predict(X_test)
        
        print("Breach Severity Model Performance:")
        print("Train:", classification_report(y_train, train_pred))
        print("Test:", classification_report(y_test, test_pred))
        
    def train_action_recommendation_model(self, historical_data: pd.DataFrame):
        """Train model to recommend remediation actions."""
        # Prepare data
        X = historical_data.drop(['recommended_actions', 'description'], axis=1)
        y = historical_data['recommended_actions']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train model
        self.action_recommendation_model = RandomForestClassifier(n_estimators=100)
        self.action_recommendation_model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = self.action_recommendation_model.predict(X_train)
        test_pred = self.action_recommendation_model.predict(X_test)
        
        print("Action Recommendation Model Performance:")
        print("Train:", classification_report(y_train, train_pred))
        print("Test:", classification_report(y_test, test_pred))
    
    def train_context_aware_model(self):
        """Train transformer-based context-aware model."""
        # Initialize transformer model
        self.context_aware_model = AutoModelForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=3
        )
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
        # TODO: Add training code when data is available
        
    def predict_breach_severity(self, breach_data: Dict[str, Any]) -> float:
        """Predict severity of a breach."""
        if not self.breach_severity_model:
            raise ValueError("Breach severity model not trained")
            
        # Prepare input
        input_data = pd.DataFrame([breach_data])
        input_data = input_data.drop(['description'], axis=1)
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        severity_score = self.breach_severity_model.predict_proba(input_scaled)[0]
        return float(np.max(severity_score))
    
    def recommend_actions(self, breach_data: Dict[str, Any]) -> List[str]:
        """Recommend remediation actions for a breach."""
        if not self.action_recommendation_model:
            raise ValueError("Action recommendation model not trained")
            
        # Prepare input
        input_data = pd.DataFrame([breach_data])
        input_data = input_data.drop(['description'], axis=1)
        input_scaled = self.scaler.transform(input_data)
        
        # Predict
        actions = self.action_recommendation_model.predict(input_scaled)[0]
        return actions
    
    def analyze_context(self, breach_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze context of the breach."""
        if not self.context_aware_model:
            raise ValueError("Context-aware model not trained")
            
        # Prepare input
        text = f"Breach type: {breach_data.get('breach_type', '')}"
        inputs = self.tokenizer(text, return_tensors='pt')
        
        # Get model output
        with torch.no_grad():
            outputs = self.context_aware_model(**inputs)
            
        # Process output
        scores = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]
        return {
            'business_impact': float(scores[0]),
            'security_risk': float(scores[1]),
            'regulatory_risk': float(scores[2])
        }


class AIEnhancedRemediationManager:
    """AI-enhanced Remediation Manager that uses ML models to make better decisions."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the AI-enhanced remediation manager."""
        self.config = config
        self.ai_engine = AIResponseEngine()
        self.is_trained = False
        
    def train_models(self, historical_data: pd.DataFrame):
        """Train all AI models."""
        self.ai_engine.train_breach_severity_model(historical_data)
        self.ai_engine.train_action_recommendation_model(historical_data)
        self.ai_engine.train_context_aware_model()
        self.is_trained = True
    
    def handle_breach(self, breach_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a breach using AI recommendations."""
        if not self.is_trained:
            raise ValueError("Models must be trained before handling breaches")
            
        # Get AI recommendations
        severity = self.ai_engine.predict_breach_severity(breach_data)
        actions = self.ai_engine.recommend_actions(breach_data)
        context_analysis = self.ai_engine.analyze_context(breach_data)
        
        # Combine with rule-based decisions
        rule_based_actions = self._get_rule_based_actions(breach_data)
        
        # Calculate final actions
        final_actions = self._combine_recommendations(
            ai_actions=actions,
            rule_actions=rule_based_actions,
            severity=severity,
            context=context_analysis
        )
        
        return {
            'severity': severity,
            'context': context_analysis,
            'recommended_actions': final_actions,
            'confidence': self._calculate_confidence(
                ai_actions=actions,
                rule_actions=rule_based_actions
            )
        }
    
    def _get_rule_based_actions(self, breach_data: Dict[str, Any]) -> List[str]:
        """Get actions based on traditional rules."""
        # TODO: Implement rule-based action selection
        return []
    
    def _combine_recommendations(self, 
                               ai_actions: List[str],
                               rule_actions: List[str],
                               severity: float,
                               context: Dict[str, float]) -> List[str]:
        """Combine AI and rule-based recommendations."""
        # Weight AI recommendations based on severity and context
        weight = severity * context['business_impact']
        
        # Combine actions with weighted scoring
        all_actions = list(set(ai_actions + rule_actions))
        action_scores = {
            action: (action in ai_actions) * weight + (action in rule_actions) * (1 - weight)
            for action in all_actions
        }
        
        # Select top actions
        sorted_actions = sorted(
            action_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [action for action, _ in sorted_actions[:3]]  # Top 3 actions
    
    def _calculate_confidence(self, 
                             ai_actions: List[str],
                             rule_actions: List[str]) -> float:
        """Calculate confidence in recommendations."""
        # Calculate agreement score
        agreement = len(set(ai_actions) & set(rule_actions)) / max(len(ai_actions), len(rule_actions))
        
        return float(agreement)
