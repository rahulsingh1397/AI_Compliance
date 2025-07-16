import pandas as pd
import logging
from typing import Dict, Any, List

from AIComplianceMonitoring.integrations.csl_service import CslService

logger = logging.getLogger(__name__)

class ComplianceChecker:
    """
    Handles compliance checks against OFAC and BIS lists.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the compliance checker.

        Args:
            config: Configuration object with necessary parameters.
        """
        self.config = config
        csl_url = self.config.get("csl_json_url", "https://data.trade.gov/downloadable_consolidated_screening_list/v1/consolidated.json")
        self.csl_service = CslService(csl_url=csl_url)
        logger.info("ComplianceChecker initialized with live CSL service.")



    def check_compliance(self, logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks logs for entities on OFAC and BIS lists.

        Args:
            logs_df: DataFrame containing the logs to check.

        Returns:
            DataFrame with added compliance check columns.
        """
        if logs_df.empty:
            return logs_df

        logger.info(f"Performing compliance check on {len(logs_df)} log entries.")

        # Ensure 'user' and 'resource' columns exist
        if 'user' not in logs_df.columns:
            logs_df['user'] = ''
        if 'resource' not in logs_df.columns:
            logs_df['resource'] = ''
            
        # Convert to string and fill NaNs
        logs_df['user'] = logs_df['user'].astype(str).fillna('')
        logs_df['resource'] = logs_df['resource'].astype(str).fillna('')

        # Check against Consolidated Screening List
        logs_df['csl_match'] = logs_df.apply(
            lambda row: self.csl_service.search_name(row['user']) or self.csl_service.search_name(row['resource']),
            axis=1
        )

        # For simplicity, the compliance breach is determined by a CSL match.
        logs_df['compliance_breach'] = logs_df['csl_match']

        num_breaches = logs_df['compliance_breach'].sum()
        if num_breaches > 0:
            logger.warning(f"Detected {num_breaches} potential compliance breaches.")
        else:
            logger.info("No compliance breaches detected.")

        return logs_df

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the compliance checker.
        """
        return {
            "csl_list_size": self.csl_service.get_list_size()
        }
