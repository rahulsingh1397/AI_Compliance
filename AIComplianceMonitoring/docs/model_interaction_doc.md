# Anomaly Detection Model Integration with RL Feedback System

## Overview

This document describes how the two primary machine learning models (IsolationForest and Autoencoder) interact with the Reinforcement Learning (RL) feedback system. The integration creates a continuous learning loop that improves anomaly detection accuracy over time while reducing false positives.

## Model Architecture

### Primary Detection Models

1. **IsolationForest**
   - Unsupervised algorithm based on decision trees
   - Works by isolating observations through random feature selection and partitioning
   - Effective at detecting anomalies in high-dimensional spaces
   - Produces anomaly scores in range [0,1] where higher values indicate stronger anomalies

2. **Autoencoder**
   - Neural network architecture with encoder-decoder structure
   - Learns efficient data representations (encoding) and reconstruction (decoding)
   - Reconstruction error serves as anomaly indicator
   - More effective at capturing complex patterns in structured and unstructured data

### RL Feedback Components

1. **ValidationAgent**
   - Uses RandomForest classifier
   - Predicts whether detected anomalies are true or false positives
   - Produces confidence scores for classifications
   
2. **HumanInteractionAgent**
   - Manages prioritization queue for human review
   - Collects structured feedback from experts
   - Maintains feedback history

3. **FeedbackIntegration**
   - Updates model parameters based on accumulated feedback
   - Optimizes detection thresholds using precision/recall metrics
   - Maintains model versioning and deployment

## Integration Flow

### 1. Anomaly Detection Phase

```
Raw Logs → LogIngestion → [IsolationForest, Autoencoder] → Anomaly Candidates
```

- Both models process normalized log data independently
- Each model produces anomaly scores for each log entry
- Ensemble scoring combines both models' outputs with configurable weights
- Logs exceeding threshold scores become anomaly candidates

### 2. Validation Phase

```
Anomaly Candidates → ValidationAgent → Validated Anomalies
```

- ValidationAgent applies RandomForest classifier to anomaly candidates
- Features include original anomaly scores plus extracted context features
- Produces validation score indicating true/false positive probability
- High-confidence validations are automatically processed
- Medium-confidence cases are flagged for human review
- Low-confidence cases may be excluded or downgraded in severity

### 3. Alert Generation Phase

```
Validated Anomalies → AlertModule → Prioritized Alerts
```

- AlertModule receives only anomalies that passed validation
- Severity is adjusted based on validation confidence
- Alerts are structured with context information for human review
- Alerts are prioritized into high/medium/low queues

### 4. Feedback Collection Phase

```
Prioritized Alerts → HumanInteractionAgent → Expert Feedback
```

- Security analysts review prioritized alerts
- Feedback is collected on each alert (true positive, false positive, nature of anomaly)
- Feedback is stored in structured format with original features and scores

### 5. Model Improvement Phase

```
Expert Feedback → FeedbackIntegration → Model Updates
```

- Accumulated feedback used to retrain ValidationAgent periodically
- Feedback influences weighting between IsolationForest and Autoencoder in ensemble
- Detection thresholds are optimized for precision/recall balance
- Parameter tuning occurs on scheduled basis or triggered by performance metrics

## Feedback Loop Mechanisms

### Model-Specific Feedback

1. **IsolationForest Improvements:**
   - Feedback adjusts the anomaly score threshold dynamically
   - Feature importance is refined based on which features contribute to validated anomalies
   - The contamination parameter can be auto-tuned based on true positive rate

2. **Autoencoder Improvements:**
   - Reconstruction error thresholds are optimized per data category
   - Network architecture can be fine-tuned for high-value anomaly types
   - Layer weights are adjusted to emphasize features with higher correlation to true positives

### Integration Benefits

1. **Reduced False Positives:**
   - Initial models may generate many false positives
   - RL feedback loop progressively filters these out
   - System learns patterns that humans consistently identify as normal

2. **Adaptive Detection:**
   - Models become more sensitive to anomalies in areas with confirmed issues
   - System can adjust to evolving threat patterns over time
   - Different weights may be applied to different log sources or systems

3. **Performance Metrics:**
   - Precision: Percentage of identified anomalies that are true positives
   - Recall: Percentage of actual anomalies that are successfully identified
   - F1 Score: Harmonic mean of precision and recall
   - False Positive Rate: Percentage of normal events incorrectly flagged

## Implementation Details

### Data Flow Between Components

1. **From Detection to Validation:**
   ```python
   # In anomaly_detection.py
   anomalies = self.detect_anomalies(logs_df)
   
   # In feedback_loop.py via rl_feedback_manager.py
   validation_results = validation_agent.validate_anomalies(anomalies)
   ```

2. **From Validation to Alerts:**
   ```python
   # In alert_module.py
   if self.feedback_manager is not None:
       validation_result = self.feedback_manager.process_anomaly(anomaly_data)
       # Use validation results to filter or adjust severity
   ```

3. **From Human Feedback to Model Updates:**
   ```python
   # In human_interaction_agent.py
   feedback = self.collect_feedback(alert_id, is_valid, feedback_details)
   
   # In feedback_integration.py
   self.update_models(feedback_batch)
   ```

### Configuration Parameters

Key parameters that control the interaction between models and the RL system:

1. **Ensemble Weights:**
   - `isolation_forest_weight`: Default 0.6
   - `autoencoder_weight`: Default 0.4
   - Adjusted based on relative performance in feedback

2. **Validation Thresholds:**
   - `high_confidence_threshold`: Default 0.8
   - `medium_confidence_threshold`: Default 0.5
   - Determines when human review is required

3. **Feedback Batch Sizes:**
   - `min_feedback_for_update`: Default 50
   - Minimum feedback entries before model update triggered

4. **Update Schedules:**
   - `feedback_check_interval`: Default 1 day
   - How often to evaluate if models need retraining

## Future Enhancements

1. **Advanced Ensemble Methods:**
   - Add support for more dynamic weighting between models
   - Implement stacked ensembles with meta-learners

2. **Transfer Learning:**
   - Allow knowledge transfer between similar systems
   - Apply feedback from one environment to improve others

3. **Active Learning:**
   - Proactively query humans about uncertain predictions
   - Focus human attention on boundary cases for maximum learning impact

4. **Explainability:**
   - Enhance model interpretability with SHAP values or LIME
   - Provide context and reasoning for anomaly classifications
