"""Abstract repository interface for triage records persistence."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseTriageRepository(ABC):
    """Abstract base repository defining the contract for triage records data operations."""

    @abstractmethod
    async def save_triage_record(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persist a triage evaluation record.

        :param record_data: Dictionary containing assessment, pet details, and image reference.
        :return: Saved record data including generated identifiers.
        """
        pass

    @abstractmethod
    async def get_recent_records(
        self,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve recently registered triage records, optionally filtered by user_id.

        :param limit: Maximum number of records to return (defaults to 10).
        :param user_id: Optional authenticated user UUID for scoped history.
        :return: List of triage records ordered chronologically descending.
        """
        pass

    @abstractmethod
    async def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single triage record by its unique database identifier.

        :param record_id: UUID or identifier string.
        :return: Record dictionary if found, None otherwise.
        """
        pass
