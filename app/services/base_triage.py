"""Abstract base interface for veterinary triage evaluation services."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from app.schemas.triage import ChatMessage, TriageAssessmentResponse


class BaseTriageService(ABC):
    """Abstract base class defining the contract for pet medical triage services."""

    @abstractmethod
    async def evaluate_condition(
        self,
        image_bytes: bytes,
        mime_type: str,
        pet_type: str,
        symptoms_description: str,
    ) -> TriageAssessmentResponse:
        """
        Evaluate pet condition using visual and textual symptoms.

        :param image_bytes: Raw binary content of the lesion or condition photograph.
        :param mime_type: Image MIME type (e.g., 'image/jpeg', 'image/png').
        :param pet_type: Species or pet classification (e.g., 'Dog', 'Cat').
        :param symptoms_description: Narrative description of symptoms and behavior.
        :return: TriageAssessmentResponse containing structured medical triage advice.
        """
        pass

    @abstractmethod
    async def follow_up_chat(
        self,
        triage_context: Dict[str, Any],
        user_message: str,
        history: List[ChatMessage],
    ) -> str:
        """
        Conduct contextual follow-up chat based on an initial triage case.

        :param triage_context: Dictionary with previous triage case findings.
        :param user_message: Latest inquiry submitted by user.
        :param history: Prior conversation items.
        :return: Clinical AI response string.
        """
        pass
