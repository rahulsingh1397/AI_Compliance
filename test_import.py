print("Attempting to import PrivacyPreservingAgent...")
try:
    from AIComplianceMonitoring.agents.privacy_preserving_agent.agent import PrivacyPreservingAgent
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
