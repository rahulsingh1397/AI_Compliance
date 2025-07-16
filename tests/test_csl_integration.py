import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from AIComplianceMonitoring.integrations.csl_service import CslService
from AIComplianceMonitoring.agents.monitoring.compliance_checker import ComplianceChecker

# Sample CSL data for mocking the API response
SAMPLE_CSL_DATA = {
    "results": [
        {
            "name": "Sanctioned Individual",
            "source": "OFAC - Specially Designated Nationals List"
        },
        {
            "name": "Restricted Entity",
            "source": "BIS - Entity List"
        }
    ]
}

class TestCslIntegration(unittest.TestCase):

    @patch('requests.get')
    def test_csl_service_and_checker_integration(self, mock_get):
        """Test the full integration of CslService with ComplianceChecker."""
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = SAMPLE_CSL_DATA
        mock_get.return_value = mock_response

        # 1. Test CslService directly
        csl_url = "http://fake-csl-url.com/consolidated.json"
        service = CslService(csl_url=csl_url, cache_ttl_seconds=600)

        # Verify search works for existing and non-existing names
        self.assertTrue(service.search_name("Sanctioned Individual"))
        self.assertTrue(service.search_name("restricted entity")) # Should be case-insensitive
        self.assertFalse(service.search_name("Clear Name Inc."))
        self.assertEqual(service.get_list_size(), 2)

        # Verify that requests.get was called
        mock_get.assert_called_once_with(csl_url, timeout=60)

        # 2. Test ComplianceChecker integration
        config = {"csl_json_url": csl_url}
        checker = ComplianceChecker(config)

        # Create a sample DataFrame
        import pandas as pd
        logs_data = {
            'user': ['Sanctioned Individual', 'Clear Name Inc.', 'Another User'],
            'resource': ['some_file.txt', 'Restricted Entity', 'safe_resource']
        }
        logs_df = pd.DataFrame(logs_data)

        # Run compliance check
        result_df = checker.check_compliance(logs_df)

        # Verify the results
        self.assertIn('compliance_breach', result_df.columns)
        # Entry 0: user is sanctioned
        self.assertTrue(result_df.loc[0, 'compliance_breach'])
        # Entry 1: resource is restricted
        self.assertTrue(result_df.loc[1, 'compliance_breach'])
        # Entry 2: no matches
        self.assertFalse(result_df.loc[2, 'compliance_breach'])

if __name__ == '__main__':
    unittest.main()
