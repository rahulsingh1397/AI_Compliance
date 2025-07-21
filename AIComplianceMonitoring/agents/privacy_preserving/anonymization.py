"""
Advanced Anonymization Techniques

This module implements sophisticated anonymization methods including K-anonymity,
L-diversity, T-closeness, differential privacy, and data suppression/generalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass
from enum import Enum, auto
import random
from collections import Counter, defaultdict
import math
from scipy import stats
import warnings

class AnonymizationMethod(Enum):
    """Types of anonymization methods."""
    K_ANONYMITY = auto()
    L_DIVERSITY = auto()
    T_CLOSENESS = auto()
    DIFFERENTIAL_PRIVACY = auto()
    SUPPRESSION = auto()
    GENERALIZATION = auto()

@dataclass
class AnonymizationConfig:
    """Configuration for anonymization parameters."""
    k: int = 3  # K-anonymity parameter
    l: int = 2  # L-diversity parameter
    t: float = 0.2  # T-closeness parameter
    epsilon: float = 1.0  # Differential privacy budget
    delta: float = 1e-5  # Differential privacy delta
    quasi_identifiers: List[str] = None  # Columns that are quasi-identifiers
    sensitive_attributes: List[str] = None  # Sensitive columns
    generalization_hierarchies: Dict[str, List[List[str]]] = None  # Generalization rules

class AdvancedAnonymizer:
    """
    Implements advanced anonymization techniques for privacy-preserving data processing.
    """
    
    def __init__(self, config: Optional[AnonymizationConfig] = None):
        """
        Initialize the advanced anonymizer.
        
        Args:
            config: Configuration for anonymization parameters
        """
        self.config = config or AnonymizationConfig()
        self.generalization_cache = {}
        
    def k_anonymize(self, df: pd.DataFrame, k: Optional[int] = None) -> pd.DataFrame:
        """
        Apply K-anonymity to ensure each record is indistinguishable from at least k-1 others.
        
        Args:
            df: Input DataFrame
            k: K-anonymity parameter (default from config)
            
        Returns:
            K-anonymized DataFrame
        """
        k = k or self.config.k
        quasi_identifiers = self.config.quasi_identifiers or []
        
        if not quasi_identifiers:
            # If no quasi-identifiers specified, use all non-sensitive columns
            sensitive = set(self.config.sensitive_attributes or [])
            quasi_identifiers = [col for col in df.columns if col not in sensitive]
        
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        
        # Keep only groups with at least k records
        valid_groups = []
        for name, group in grouped:
            if len(group) >= k:
                valid_groups.append(group)
            else:
                # For small groups, we need to generalize or suppress
                generalized_group = self._generalize_small_group(group, quasi_identifiers, k)
                if generalized_group is not None:
                    valid_groups.append(generalized_group)
        
        if not valid_groups:
            warnings.warn("No valid groups found for K-anonymity. Consider lowering k or improving generalization.")
            return pd.DataFrame()
        
        return pd.concat(valid_groups, ignore_index=True)
    
    def l_diversify(self, df: pd.DataFrame, l: Optional[int] = None) -> pd.DataFrame:
        """
        Apply L-diversity to ensure sensitive attributes have diverse values within each group.
        
        Args:
            df: Input DataFrame (should already be K-anonymous)
            l: L-diversity parameter (default from config)
            
        Returns:
            L-diverse DataFrame
        """
        l = l or self.config.l
        quasi_identifiers = self.config.quasi_identifiers or []
        sensitive_attributes = self.config.sensitive_attributes or []
        
        if not sensitive_attributes:
            warnings.warn("No sensitive attributes specified for L-diversity")
            return df
        
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        valid_groups = []
        
        for name, group in grouped:
            is_l_diverse = True
            
            # Check L-diversity for each sensitive attribute
            for sensitive_attr in sensitive_attributes:
                if sensitive_attr not in group.columns:
                    continue
                    
                # Count distinct values in sensitive attribute
                value_counts = group[sensitive_attr].value_counts()
                
                # Check if we have at least l distinct values
                if len(value_counts) < l:
                    is_l_diverse = False
                    break
                
                # Check if the most frequent value doesn't dominate too much
                max_freq = value_counts.iloc[0]
                if max_freq > len(group) / l:
                    is_l_diverse = False
                    break
            
            if is_l_diverse:
                valid_groups.append(group)
            else:
                # Try to make the group L-diverse through sampling or generalization
                diverse_group = self._make_l_diverse(group, sensitive_attributes, l)
                if diverse_group is not None:
                    valid_groups.append(diverse_group)
        
        if not valid_groups:
            warnings.warn("No L-diverse groups found. Consider lowering l or improving data diversity.")
            return pd.DataFrame()
        
        return pd.concat(valid_groups, ignore_index=True)
    
    def t_closeness(self, df: pd.DataFrame, t: Optional[float] = None) -> pd.DataFrame:
        """
        Apply T-closeness to maintain statistical properties of sensitive attributes.
        
        Args:
            df: Input DataFrame (should already be K-anonymous and L-diverse)
            t: T-closeness parameter (default from config)
            
        Returns:
            T-close DataFrame
        """
        t = t or self.config.t
        quasi_identifiers = self.config.quasi_identifiers or []
        sensitive_attributes = self.config.sensitive_attributes or []
        
        if not sensitive_attributes:
            warnings.warn("No sensitive attributes specified for T-closeness")
            return df
        
        # Calculate global distribution for each sensitive attribute
        global_distributions = {}
        for sensitive_attr in sensitive_attributes:
            if sensitive_attr in df.columns:
                if df[sensitive_attr].dtype in ['object', 'category']:
                    # Categorical attribute
                    global_distributions[sensitive_attr] = df[sensitive_attr].value_counts(normalize=True)
                else:
                    # Numerical attribute - use histogram
                    global_distributions[sensitive_attr] = df[sensitive_attr]
        
        # Group by quasi-identifiers and check T-closeness
        grouped = df.groupby(quasi_identifiers)
        valid_groups = []
        
        for name, group in grouped:
            is_t_close = True
            
            for sensitive_attr in sensitive_attributes:
                if sensitive_attr not in group.columns:
                    continue
                
                global_dist = global_distributions[sensitive_attr]
                
                if isinstance(global_dist, pd.Series):  # Categorical
                    # Calculate Earth Mover's Distance for categorical data
                    group_dist = group[sensitive_attr].value_counts(normalize=True)
                    distance = self._earth_movers_distance_categorical(group_dist, global_dist)
                else:  # Numerical
                    # Use Kolmogorov-Smirnov test for numerical data
                    _, p_value = stats.ks_2samp(group[sensitive_attr], global_dist)
                    distance = 1 - p_value  # Convert p-value to distance-like measure
                
                if distance > t:
                    is_t_close = False
                    break
            
            if is_t_close:
                valid_groups.append(group)
        
        if not valid_groups:
            warnings.warn("No T-close groups found. Consider increasing t or improving data distribution.")
            return pd.DataFrame()
        
        return pd.concat(valid_groups, ignore_index=True)
    
    def add_differential_privacy(
        self, 
        df: pd.DataFrame, 
        epsilon: Optional[float] = None,
        delta: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Add differential privacy by introducing calibrated noise.
        
        Args:
            df: Input DataFrame
            epsilon: Privacy budget (smaller = more private)
            delta: Probability of privacy breach
            
        Returns:
            DataFrame with differential privacy noise added
        """
        epsilon = epsilon or self.config.epsilon
        delta = delta or self.config.delta
        
        df_noisy = df.copy()
        
        # Add noise to numerical columns
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                # Calculate sensitivity (max change in output for unit change in input)
                sensitivity = self._calculate_sensitivity(df[col])
                
                # Add Laplace noise for epsilon-differential privacy
                noise_scale = sensitivity / epsilon
                noise = np.random.laplace(0, noise_scale, size=len(df))
                
                df_noisy[col] = df[col] + noise
                
                # For integer columns, round the results
                if df[col].dtype == 'int64':
                    df_noisy[col] = df_noisy[col].round().astype('int64')
        
        return df_noisy
    
    def suppress_data(
        self, 
        df: pd.DataFrame, 
        suppression_rate: float = 0.1,
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Apply data suppression by removing or masking certain values.
        
        Args:
            df: Input DataFrame
            suppression_rate: Fraction of data to suppress (0.0 to 1.0)
            columns: Specific columns to suppress (default: quasi-identifiers)
            
        Returns:
            DataFrame with suppressed data
        """
        df_suppressed = df.copy()
        target_columns = columns or self.config.quasi_identifiers or df.columns.tolist()
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            # Randomly select rows to suppress
            n_suppress = int(len(df) * suppression_rate)
            suppress_indices = random.sample(range(len(df)), n_suppress)
            
            # Replace with NaN or a suppression marker
            df_suppressed.loc[suppress_indices, col] = '*'  # Suppression marker
        
        return df_suppressed
    
    def generalize_data(
        self, 
        df: pd.DataFrame, 
        generalization_level: int = 1
    ) -> pd.DataFrame:
        """
        Apply data generalization using predefined hierarchies.
        
        Args:
            df: Input DataFrame
            generalization_level: Level of generalization (higher = more general)
            
        Returns:
            DataFrame with generalized data
        """
        df_generalized = df.copy()
        hierarchies = self.config.generalization_hierarchies or {}
        
        for col, hierarchy in hierarchies.items():
            if col not in df.columns:
                continue
                
            if generalization_level >= len(hierarchy):
                generalization_level = len(hierarchy) - 1
            
            # Apply generalization mapping
            generalization_map = {}
            for i, level_values in enumerate(hierarchy[generalization_level]):
                if i < len(hierarchy[0]):  # Original values
                    generalization_map[hierarchy[0][i]] = level_values
            
            # Apply the mapping
            df_generalized[col] = df_generalized[col].map(generalization_map).fillna(df_generalized[col])
        
        return df_generalized
    
    def full_anonymization_pipeline(
        self, 
        df: pd.DataFrame,
        methods: List[AnonymizationMethod] = None
    ) -> pd.DataFrame:
        """
        Apply a complete anonymization pipeline with multiple techniques.
        
        Args:
            df: Input DataFrame
            methods: List of anonymization methods to apply
            
        Returns:
            Fully anonymized DataFrame
        """
        if methods is None:
            methods = [
                AnonymizationMethod.SUPPRESSION,
                AnonymizationMethod.GENERALIZATION,
                AnonymizationMethod.K_ANONYMITY,
                AnonymizationMethod.L_DIVERSITY,
                AnonymizationMethod.T_CLOSENESS,
                AnonymizationMethod.DIFFERENTIAL_PRIVACY
            ]
        
        result = df.copy()
        
        for method in methods:
            if method == AnonymizationMethod.SUPPRESSION:
                result = self.suppress_data(result)
            elif method == AnonymizationMethod.GENERALIZATION:
                result = self.generalize_data(result)
            elif method == AnonymizationMethod.K_ANONYMITY:
                result = self.k_anonymize(result)
            elif method == AnonymizationMethod.L_DIVERSITY:
                result = self.l_diversify(result)
            elif method == AnonymizationMethod.T_CLOSENESS:
                result = self.t_closeness(result)
            elif method == AnonymizationMethod.DIFFERENTIAL_PRIVACY:
                result = self.add_differential_privacy(result)
        
        return result
    
    def _generalize_small_group(
        self, 
        group: pd.DataFrame, 
        quasi_identifiers: List[str], 
        k: int
    ) -> Optional[pd.DataFrame]:
        """Generalize a small group to meet K-anonymity requirements."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated generalization strategies
        return None  # For now, we suppress small groups
    
    def _make_l_diverse(
        self, 
        group: pd.DataFrame, 
        sensitive_attributes: List[str], 
        l: int
    ) -> Optional[pd.DataFrame]:
        """Attempt to make a group L-diverse through sampling or other techniques."""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated diversity enhancement strategies
        return None  # For now, we remove non-diverse groups
    
    def _earth_movers_distance_categorical(
        self, 
        dist1: pd.Series, 
        dist2: pd.Series
    ) -> float:
        """Calculate Earth Mover's Distance for categorical distributions."""
        # Simplified implementation - in practice, use proper EMD calculation
        all_categories = set(dist1.index) | set(dist2.index)
        
        total_distance = 0.0
        for category in all_categories:
            p1 = dist1.get(category, 0)
            p2 = dist2.get(category, 0)
            total_distance += abs(p1 - p2)
        
        return total_distance / 2  # Normalize
    
    def _calculate_sensitivity(self, series: pd.Series) -> float:
        """Calculate the sensitivity of a numerical series."""
        # For most practical purposes, we can use the range as sensitivity
        return series.max() - series.min()
    
    def evaluate_privacy_metrics(self, original_df: pd.DataFrame, anonymized_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate privacy metrics for the anonymized dataset.
        
        Args:
            original_df: Original dataset
            anonymized_df: Anonymized dataset
            
        Returns:
            Dictionary containing privacy evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['data_retention_rate'] = len(anonymized_df) / len(original_df)
        metrics['attribute_retention_rate'] = len(anonymized_df.columns) / len(original_df.columns)
        
        # K-anonymity check
        if self.config.quasi_identifiers:
            grouped = anonymized_df.groupby(self.config.quasi_identifiers)
            group_sizes = [len(group) for _, group in grouped]
            metrics['min_group_size'] = min(group_sizes) if group_sizes else 0
            metrics['k_anonymity_satisfied'] = metrics['min_group_size'] >= self.config.k
        
        # Information loss metrics (simplified)
        numerical_cols = original_df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            original_var = original_df[numerical_cols].var().mean()
            anonymized_var = anonymized_df[numerical_cols].var().mean()
            metrics['variance_preservation'] = anonymized_var / original_var if original_var > 0 else 0
        
        return metrics
