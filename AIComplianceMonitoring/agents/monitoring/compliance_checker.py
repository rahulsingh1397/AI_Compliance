import pandas as pd
import logging
from typing import Dict, Any, List

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
        self.ofac_list = self._load_ofac_list()
        self.bis_list = self._load_bis_list()
        logger.info("ComplianceChecker initialized.")

    def _load_ofac_list(self) -> List[str]:
        """
        Loads the OFAC Specially Designated Nationals (SDN) list.
        In a real implementation, this would fetch the list from the U.S. Treasury.
        """
        # For demonstration, using a mock list.
        mock_ofac_list = [
            "Evil Corp",
            "Bad Actor Inc.",
            "Sanctioned Entity Ltd."
        ]
        logger.info(f"Loaded {len(mock_ofac_list)} entities into OFAC mock list.")
        return mock_ofac_list

    def _load_bis_list(self) -> List[str]:
        """
        Loads the BIS Entity List.
        In a real implementation, this would fetch the list from the Bureau of Industry and Security.
        """
        # For demonstration, using a mock list.
        mock_bis_list = [
            "Questionable Tech",
            "Risky Business Associates"
        ]
        logger.info(f"Loaded {len(mock_bis_list)} entities into BIS mock list.")
        return mock_bis_list

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

        # Check against OFAC list
        logs_df['ofac_match'] = logs_df.apply(
            lambda row: any(entity in row['user'] or entity in row['resource'] for entity in self.ofac_list),
            axis=1
        )

        # Check against BIS list
        logs_df['bis_match'] = logs_df.apply(
            lambda row: any(entity in row['user'] or entity in row['resource'] for entity in self.bis_list),
            axis=1
        )
        
        # Combine results into a single compliance flag
        logs_df['compliance_breach'] = logs_df['ofac_match'] | logs_df['bis_match']

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
            "ofac_list_size": len(self.ofac_list),
            "bis_list_size": len(self.bis_list)
        }
