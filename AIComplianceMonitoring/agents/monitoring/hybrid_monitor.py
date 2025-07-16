"""
Hybrid Monitoring System that integrates traditional rule-based monitoring with AI-based approaches.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from AIComplianceMonitoring.agents.monitoring.agent import MonitoringAgent, MonitoringAgentConfig
from AIComplianceMonitoring.agents.monitoring.hybrid_anomaly_detector import HybridAnomalyDetector
from AIComplianceMonitoring.agents.remediation.manager import RemediationManager
from AIComplianceMonitoring.agents.remediation.ai_remediation_manager import AIEnhancedRemediationManager

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class HybridMonitoringConfig(MonitoringAgentConfig):
    """Configuration for hybrid monitoring agent."""
    use_hybrid_anomaly_detection: bool = True
    use_ai_remediation: bool = True
    hybrid_confidence_threshold: float = 0.7
    ai_training_data_path: Optional[str] = None
    fallback_to_traditional: bool = True


class HybridMonitoringAgent(MonitoringAgent):
    """
    Hybrid Monitoring Agent that combines traditional rule-based monitoring with
    AI-powered anomaly detection and remediation.
    """
    
    def __init__(self, config: HybridMonitoringConfig):
        """Initialize the hybrid monitoring agent."""
        super().__init__(config)
        self.hybrid_config = config
        self.hybrid_detector = None
        self.ai_remediation_manager = None
        
        # Initialize AI components if enabled
        self._initialize_ai_components()
        
        # Stats for AI components
        self.ai_stats = {
            "anomaly_detection": {
                "total_detections": 0,
                "lstm_detections": 0,
                "autoencoder_detections": 0,
                "combined_detections": 0,
                "false_positives": 0
            },
            "remediation": {
                "total_actions": 0,
                "ai_recommended_actions": 0,
                "rule_based_actions": 0,
                "hybrid_actions": 0
            }
        }
    
    def _initialize_ai_components(self):
        """Initialize AI components if enabled."""
        if self.hybrid_config.use_hybrid_anomaly_detection:
            logger.info("Initializing Hybrid Anomaly Detection System")
            self.hybrid_detector = HybridAnomalyDetector()
            
            # Train the model if training data is available
            if self.hybrid_config.ai_training_data_path:
                try:
                    training_data = pd.read_csv(self.hybrid_config.ai_training_data_path)
                    self.hybrid_detector.train(training_data)
                    logger.info("Hybrid anomaly detector trained successfully")
                except Exception as e:
                    logger.error(f"Failed to train hybrid anomaly detector: {e}")
                    if self.hybrid_config.fallback_to_traditional:
                        logger.warning("Falling back to traditional anomaly detection")
                        self.hybrid_config.use_hybrid_anomaly_detection = False
        
        if self.hybrid_config.use_ai_remediation:
            logger.info("Initializing AI-Enhanced Remediation Manager")
            
            # Keep the traditional remediation manager as fallback
            self.traditional_remediation_module = self.remediation_module
            
            # Create AI remediation manager
            ai_remediation_config = {
                # Inherit configuration from traditional remediation
                **self.config.remediation_config.config,
                "ai_enhanced": True
            }
            
            try:
                self.ai_remediation_manager = AIEnhancedRemediationManager(ai_remediation_config)
                
                # Train if training data is available
                if self.hybrid_config.ai_training_data_path:
                    training_data = pd.read_csv(self.hybrid_config.ai_training_data_path)
                    self.ai_remediation_manager.train_models(training_data)
                    logger.info("AI remediation manager trained successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AI remediation manager: {e}")
                if self.hybrid_config.fallback_to_traditional:
                    logger.warning("Falling back to traditional remediation")
                    self.hybrid_config.use_ai_remediation = False
    
    def _detect_anomalies(self, logs: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """Detect anomalies using hybrid approach or fall back to traditional."""
        if self.hybrid_config.use_hybrid_anomaly_detection and self.hybrid_detector and self.hybrid_detector.is_trained:
            try:
                # Use hybrid anomaly detection
                detection_results = self.hybrid_detector.detect_anomalies(logs)
                
                # Extract anomalous entries
                anomaly_mask = detection_results['anomalies']['combined']
                anomalous_logs = logs.iloc[anomaly_mask]
                
                # Create alert data
                alerts = []
                for idx, row in anomalous_logs.iterrows():
                    confidence = detection_results['scores']['combined'][idx]
                    alerts.append({
                        'source': row.get('source', 'unknown'),
                        'user': row.get('user', 'unknown'),
                        'resource': row.get('resource', 'unknown'),
                        'timestamp': row.get('timestamp', ''),
                        'anomaly_score': float(confidence),
                        'detection_method': 'hybrid_ai'
                    })
                
                # Update stats
                self.ai_stats['anomaly_detection']['total_detections'] += len(alerts)
                self.ai_stats['anomaly_detection']['lstm_detections'] += sum(detection_results['anomalies']['lstm'])
                self.ai_stats['anomaly_detection']['autoencoder_detections'] += sum(detection_results['anomalies']['ae'])
                self.ai_stats['anomaly_detection']['combined_detections'] += sum(detection_results['anomalies']['combined'])
                
                return anomalous_logs, alerts
            
            except Exception as e:
                logger.error(f"Hybrid anomaly detection failed: {e}")
                if self.hybrid_config.fallback_to_traditional:
                    logger.warning("Falling back to traditional anomaly detection")
                    return super()._detect_anomalies(logs)
                return pd.DataFrame(), []
        else:
            # Fall back to traditional anomaly detection
            return super()._detect_anomalies(logs)
    
    def _handle_compliance_breach(self, breach_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle compliance breach using AI-enhanced or traditional approach."""
        if self.hybrid_config.use_ai_remediation and self.ai_remediation_manager and self.ai_remediation_manager.is_trained:
            try:
                # Use AI-enhanced remediation
                ai_recommendation = self.ai_remediation_manager.handle_breach(breach_data)
                
                # Get traditional remediation for comparison
                traditional_actions = self.traditional_remediation_module.handle_compliance_breach(breach_data)
                
                # Decide based on confidence
                if ai_recommendation.get('confidence', 0) >= self.hybrid_config.hybrid_confidence_threshold:
                    # Use AI recommendation
                    actions_to_take = ai_recommendation['recommended_actions']
                    
                    # Convert to format expected by monitoring agent
                    results = []
                    for action in actions_to_take:
                        results.append({
                            'action': action,
                            'status': 'success',
                            'breach_type': breach_data.get('breach_type', 'unknown'),
                            'severity': ai_recommendation.get('severity', 5),
                            'method': 'ai_enhanced'
                        })
                    
                    # Update stats
                    self.ai_stats['remediation']['total_actions'] += len(results)
                    self.ai_stats['remediation']['ai_recommended_actions'] += len(results)
                    
                    return results
                else:
                    # Fall back to traditional remediation
                    self.ai_stats['remediation']['total_actions'] += len(traditional_actions)
                    self.ai_stats['remediation']['rule_based_actions'] += len(traditional_actions)
                    
                    # Mark as traditional
                    for action in traditional_actions:
                        action['method'] = 'traditional'
                    
                    return traditional_actions
            
            except Exception as e:
                logger.error(f"AI-enhanced remediation failed: {e}")
                if self.hybrid_config.fallback_to_traditional:
                    logger.warning("Falling back to traditional remediation")
                    return super()._handle_compliance_breach(breach_data)
                return []
        else:
            # Fall back to traditional remediation
            traditional_actions = super()._handle_compliance_breach(breach_data)
            
            # Update stats
            self.ai_stats['remediation']['total_actions'] += len(traditional_actions)
            self.ai_stats['remediation']['rule_based_actions'] += len(traditional_actions)
            
            # Mark as traditional
            for action in traditional_actions:
                action['method'] = 'traditional'
            
            return traditional_actions
    
    def _monitor_source(self, source_type: str, source_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor a specific log source using hybrid approach.
        
        Args:
            source_type: Type of the source (aws_s3, azure_blob, on_prem)
            source_config: Configuration for the source
        """
        try:
            logger.info(f"Monitoring source: {source_type} - {source_config.get('name', 'unnamed')}")
            
            # Get logs from the source
            logs = self.log_ingestion_module.ingest_logs(source_type, source_config)
            
            if not logs or logs.empty:
                logger.info(f"No logs found for source {source_type}")
                return {
                    "status": "success", 
                    "message": "No logs found", 
                    "source_type": source_type,
                    "source_name": source_config.get("name", "unnamed")
                }
            
            logger.info(f"Found {len(logs)} log entries to process")
            
            # Detect anomalies using hybrid approach
            anomalous_logs, alerts = self._detect_anomalies(logs)
            
            if not anomalous_logs.empty:
                logger.info(f"Detected {len(anomalous_logs)} anomalous log entries")
                
                # Generate alerts
                if alerts:
                    for alert in alerts:
                        self.alert_module.add_alert(alert)
                
                # Check for compliance breaches
                for _, log_entry in anomalous_logs.iterrows():
                    # Convert to dict for compliance checker
                    log_dict = log_entry.to_dict()
                    
                    # Check compliance
                    compliance_results = self.compliance_module.check_compliance(
                        log_dict.get("user", ""), 
                        log_dict.get("resource", ""),
                        log_dict.get("action", ""),
                        log_dict.get("timestamp", "")
                    )
                    
                    # Handle any breaches detected
                    if compliance_results.get("breaches", []):
                        for breach in compliance_results["breaches"]:
                            if self.config.enable_remediation:
                                # Use hybrid approach to handle breach
                                remediation_results = self._handle_compliance_breach(breach)
                                
                                # Log results
                                for result in remediation_results:
                                    logger.info(f"Remediation action '{result['action']}' " +
                                               f"for {breach.get('breach_type', 'unknown')} breach: {result['status']}")
            
            return {
                "status": "success",
                "message": f"Processed {len(logs)} logs, found {len(anomalous_logs)} anomalies",
                "source_type": source_type,
                "source_name": source_config.get("name", "unnamed"),
                "anomaly_count": len(anomalous_logs),
                "alert_count": len(alerts),
                "using_hybrid_detection": self.hybrid_config.use_hybrid_anomaly_detection,
                "using_ai_remediation": self.hybrid_config.use_ai_remediation
            }
            
        except Exception as e:
            logger.error(f"Error monitoring source {source_type}: {e}")
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "source_type": source_type,
                "source_name": source_config.get("name", "unnamed")
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring stats including AI components."""
        # Get base stats
        stats = super().get_stats()
        
        # Add AI stats
        stats["ai_components"] = {
            "hybrid_anomaly_detection": {
                "enabled": self.hybrid_config.use_hybrid_anomaly_detection,
                "trained": self.hybrid_detector.is_trained if self.hybrid_detector else False,
                **self.ai_stats["anomaly_detection"]
            },
            "ai_remediation": {
                "enabled": self.hybrid_config.use_ai_remediation,
                "trained": self.ai_remediation_manager.is_trained if self.ai_remediation_manager else False,
                **self.ai_stats["remediation"]
            }
        }
        
        return stats
