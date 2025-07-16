import requests
import logging
import time
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CslService:
    """
    A service to download, cache, and search the Consolidated Screening List (CSL).
    """
    def __init__(self, csl_url: str, cache_ttl_seconds: int = 86400): # Default TTL is 24 hours
        """
        Initializes the CSL service.

        Args:
            csl_url: The URL to the consolidated.json file.
            cache_ttl_seconds: The time-to-live for the cache in seconds.
        """
        self.csl_url = csl_url
        self.cache_ttl = cache_ttl_seconds
        self._cache: Optional[Dict[str, Any]] = None
        self._last_updated: float = 0
        self._name_set = set()

    def _is_cache_valid(self) -> bool:
        """
        Checks if the current cache is still valid.
        """
        if self._cache is None:
            return False
        return (time.time() - self._last_updated) < self.cache_ttl

    def _update_cache(self):
        """
        Downloads the CSL data and updates the cache.
        """
        logger.info(f"Downloading CSL data from {self.csl_url}")
        try:
            response = requests.get(self.csl_url, timeout=60)
            response.raise_for_status() # Raise an exception for bad status codes
            data = response.json()
            self._cache = data
            self._last_updated = time.time()
            self._build_name_set()
            logger.info(f"Successfully updated CSL cache with {len(self._cache.get('results', []))} entries.")
        except requests.RequestException as e:
            logger.error(f"Failed to download CSL data: {e}")
            # Keep stale cache if update fails

    def _build_name_set(self):
        """
        Builds a set of names from the CSL data for efficient searching.
        """
        if not self._cache:
            return
        
        self._name_set.clear()
        for entry in self._cache.get('results', []):
            if 'name' in entry:
                self._name_set.add(entry['name'].lower())

    def search_name(self, name: str) -> bool:
        """
        Searches for a name in the CSL.

        Args:
            name: The name to search for.

        Returns:
            True if the name is found, False otherwise.
        """
        if not self._is_cache_valid():
            self._update_cache()
        
        if not self._cache:
            logger.warning("Cannot perform search; CSL cache is empty.")
            return False

        return name.lower() in self._name_set

    def get_list_size(self) -> int:
        """
        Returns the number of unique names in the CSL list.
        """
        if not self._is_cache_valid():
            self._update_cache()
        return len(self._name_set)
