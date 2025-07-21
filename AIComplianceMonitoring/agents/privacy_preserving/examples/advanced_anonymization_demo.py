"""
Advanced Anonymization Techniques Demo

This script demonstrates the advanced anonymization capabilities including:
- K-anonymity
- L-diversity  
- T-closeness
- Differential privacy
- Data suppression/generalization
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our privacy-preserving components
from .. import DataProtectionManager, AnonymizationConfig, AnonymizationMethod

def create_sample_dataset():
    """Create a sample dataset for demonstration."""
    np.random.seed(42)
    
    # Generate synthetic healthcare data
    n_records = 1000
    
    data = {
        'age': np.random.randint(18, 90, n_records),
        'zipcode': np.random.choice(['12345', '12346', '12347', '54321', '54322'], n_records),
        'gender': np.random.choice(['M', 'F'], n_records),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_records),
        'salary': np.random.normal(50000, 20000, n_records).astype(int),
        'disease': np.random.choice(['Diabetes', 'Hypertension', 'Heart Disease', 'Cancer', 'Healthy'], n_records),
        'treatment_cost': np.random.normal(5000, 2000, n_records).astype(int)
    }
    
    return pd.DataFrame(data)

def demo_k_anonymity():
    """Demonstrate K-anonymity implementation."""
    print("\n=== K-Anonymity Demo ===")
    
    # Create sample data
    df = create_sample_dataset()
    print(f"Original dataset size: {len(df)} records")
    
    # Configure anonymization
    config = AnonymizationConfig(
        k=5,  # Each group must have at least 5 records
        quasi_identifiers=['age', 'zipcode', 'gender', 'education']
    )
    
    # Initialize data protection manager
    dp = DataProtectionManager(anonymization_config=config)
    
    # Apply K-anonymity
    k_anonymous_df = dp.k_anonymize_dataframe(df)
    print(f"K-anonymous dataset size: {len(k_anonymous_df)} records")
    
    # Verify K-anonymity
    if len(k_anonymous_df) > 0:
        grouped = k_anonymous_df.groupby(config.quasi_identifiers)
        min_group_size = min([len(group) for _, group in grouped])
        print(f"Minimum group size: {min_group_size} (should be >= {config.k})")
        print(f"K-anonymity satisfied: {min_group_size >= config.k}")
    
    return k_anonymous_df

def demo_l_diversity():
    """Demonstrate L-diversity implementation."""
    print("\n=== L-Diversity Demo ===")
    
    # Create sample data
    df = create_sample_dataset()
    
    # Configure anonymization
    config = AnonymizationConfig(
        k=5,
        l=3,  # Each group must have at least 3 distinct sensitive values
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease']
    )
    
    dp = DataProtectionManager(anonymization_config=config)
    
    # First apply K-anonymity
    k_anonymous_df = dp.k_anonymize_dataframe(df)
    
    if len(k_anonymous_df) > 0:
        # Then apply L-diversity
        l_diverse_df = dp.l_diversify_dataframe(k_anonymous_df)
        print(f"L-diverse dataset size: {len(l_diverse_df)} records")
        
        # Verify L-diversity
        if len(l_diverse_df) > 0:
            grouped = l_diverse_df.groupby(config.quasi_identifiers)
            l_diverse_groups = 0
            for _, group in grouped:
                distinct_diseases = group['disease'].nunique()
                if distinct_diseases >= config.l:
                    l_diverse_groups += 1
            
            print(f"L-diverse groups: {l_diverse_groups}/{len(grouped)} groups")
            print(f"L-diversity satisfied: {l_diverse_groups == len(grouped)}")
        
        return l_diverse_df
    else:
        print("No data available after K-anonymity")
        return pd.DataFrame()

def demo_differential_privacy():
    """Demonstrate Differential Privacy implementation."""
    print("\n=== Differential Privacy Demo ===")
    
    # Create sample data
    df = create_sample_dataset()
    
    # Configure differential privacy
    config = AnonymizationConfig(
        epsilon=1.0,  # Privacy budget
        delta=1e-5    # Privacy parameter
    )
    
    dp = DataProtectionManager(anonymization_config=config)
    
    # Apply differential privacy
    dp_df = dp.add_differential_privacy_to_dataframe(df)
    
    print(f"Original salary mean: {df['salary'].mean():.2f}")
    print(f"DP salary mean: {dp_df['salary'].mean():.2f}")
    print(f"Original age mean: {df['age'].mean():.2f}")
    print(f"DP age mean: {dp_df['age'].mean():.2f}")
    
    # Calculate noise added
    salary_noise = abs(df['salary'].mean() - dp_df['salary'].mean())
    age_noise = abs(df['age'].mean() - dp_df['age'].mean())
    
    print(f"Noise added to salary: {salary_noise:.2f}")
    print(f"Noise added to age: {age_noise:.2f}")
    
    return dp_df

def demo_full_pipeline():
    """Demonstrate the complete anonymization pipeline."""
    print("\n=== Full Anonymization Pipeline Demo ===")
    
    # Create sample data
    df = create_sample_dataset()
    print(f"Original dataset: {len(df)} records, {len(df.columns)} columns")
    
    # Configure comprehensive anonymization
    config = AnonymizationConfig(
        k=3,
        l=2,
        t=0.3,
        epsilon=2.0,
        delta=1e-5,
        quasi_identifiers=['age', 'zipcode', 'gender'],
        sensitive_attributes=['disease', 'salary'],
        generalization_hierarchies={
            'age': [
                ['18-25', '26-35', '36-45', '46-55', '56-65', '65+'],  # Level 0 (most specific)
                ['Young', 'Middle-aged', 'Senior']  # Level 1 (more general)
            ]
        }
    )
    
    dp = DataProtectionManager(anonymization_config=config)
    
    # Apply full anonymization pipeline
    methods = [
        AnonymizationMethod.SUPPRESSION,
        AnonymizationMethod.K_ANONYMITY,
        AnonymizationMethod.L_DIVERSITY,
        AnonymizationMethod.DIFFERENTIAL_PRIVACY
    ]
    
    anonymized_df = dp.full_anonymization_pipeline(df, methods)
    
    print(f"Anonymized dataset: {len(anonymized_df)} records, {len(anonymized_df.columns)} columns")
    
    # Evaluate privacy metrics
    if len(anonymized_df) > 0:
        metrics = dp.anonymizer.evaluate_privacy_metrics(df, anonymized_df)
        
        print("\n=== Privacy Metrics ===")
        print(f"Data retention rate: {metrics.get('data_retention_rate', 0):.2%}")
        print(f"Attribute retention rate: {metrics.get('attribute_retention_rate', 0):.2%}")
        print(f"K-anonymity satisfied: {metrics.get('k_anonymity_satisfied', False)}")
        print(f"Minimum group size: {metrics.get('min_group_size', 0)}")
        
        if 'variance_preservation' in metrics:
            print(f"Variance preservation: {metrics['variance_preservation']:.2%}")
    
    return anonymized_df

def demo_privacy_utility_tradeoff():
    """Demonstrate the privacy-utility tradeoff with different parameters."""
    print("\n=== Privacy-Utility Tradeoff Demo ===")
    
    df = create_sample_dataset()
    
    # Test different privacy levels
    privacy_levels = [
        {'k': 2, 'epsilon': 5.0, 'name': 'Low Privacy'},
        {'k': 5, 'epsilon': 2.0, 'name': 'Medium Privacy'},
        {'k': 10, 'epsilon': 0.5, 'name': 'High Privacy'}
    ]
    
    print(f"Original dataset size: {len(df)} records")
    
    for level in privacy_levels:
        config = AnonymizationConfig(
            k=level['k'],
            epsilon=level['epsilon'],
            quasi_identifiers=['age', 'zipcode', 'gender'],
            sensitive_attributes=['disease']
        )
        
        dp = DataProtectionManager(anonymization_config=config)
        
        # Apply anonymization
        anonymized = dp.full_anonymization_pipeline(df)
        
        # Calculate metrics
        retention_rate = len(anonymized) / len(df) if len(df) > 0 else 0
        
        print(f"{level['name']} (k={level['k']}, ε={level['epsilon']}): "
              f"{len(anonymized)} records retained ({retention_rate:.1%})")

def main():
    """Run all anonymization demos."""
    print("=== Advanced Anonymization Techniques Demo ===")
    print("Demonstrating K-anonymity, L-diversity, T-closeness, and Differential Privacy")
    
    try:
        # Run individual demos
        demo_k_anonymity()
        demo_l_diversity()
        demo_differential_privacy()
        demo_full_pipeline()
        demo_privacy_utility_tradeoff()
        
        print("\n=== Demo Complete ===")
        print("Successfully demonstrated all advanced anonymization techniques!")
        
    except Exception as e:
        print(f"Error during demo: {str(e)}")
        print("Make sure all required dependencies are installed:")
        print("pip install pandas scipy scikit-learn")

if __name__ == "__main__":
    main()
