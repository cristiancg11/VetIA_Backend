"""Data transfer models and schemas for pet triage, clinic localization, follow-up chat, and database persistence."""

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field, model_validator


class TriageUrgencyLevel(str, Enum):
    """Clinical urgency severity enumeration."""
    CRITICAL = "CRITICAL"
    MODERATE = "MODERATE"
    MILD = "MILD"


class FacilityType(str, Enum):
    """Recommended facility classification."""
    EMERGENCY_24H = "Clínica 24 horas con quirófano"
    GENERAL_CLINIC = "Consultorio general"


URGENCY_COLOR_MAPPING = {
    TriageUrgencyLevel.CRITICAL.value: "#ef4444",
    TriageUrgencyLevel.MODERATE.value: "#f59e0b",
    TriageUrgencyLevel.MILD.value: "#10b981",
}


class TriageAssessmentResponse(BaseModel):
    """Structured response model for multimodal veterinary triage assessment."""

    urgency_level: Literal["CRITICAL", "MODERATE", "MILD"] = Field(
        ...,
        description="Clinical triage urgency level estimated by visual and symptomatic assessment.",
    )
    urgency_color: Literal["#ef4444", "#f59e0b", "#10b981"] = Field(
        ...,
        description="Hexadecimal color associated with urgency level (#ef4444: CRITICAL, #f59e0b: MODERATE, #10b981: MILD).",
    )
    preliminary_assessment: str = Field(
        ...,
        description="Objective and empathetic clinical summary of visible findings and reported symptoms.",
    )
    immediate_care_tips: List[str] = Field(
        ...,
        description="Step-by-step immediate safe first-aid and stabilization guidelines.",
    )
    recommended_facility_type: Literal[
        "Clínica 24 horas con quirófano",
        "Consultorio general",
    ] = Field(
        ...,
        description="Recommended veterinary facility type based on estimated severity.",
    )
    warning_disclaimer: str = Field(
        ...,
        description="Mandatory medical and legal disclaimer clarifying that AI triage does not replace a physical veterinary exam.",
    )
    image_url: Optional[str] = Field(
        None,
        description="Public URL of the stored pet image in Supabase Storage.",
    )
    record_id: Optional[str] = Field(
        None,
        description="Unique database identifier for the persisted triage record.",
    )

    @model_validator(mode="after")
    def synchronize_urgency_color(self) -> "TriageAssessmentResponse":
        """Ensure strict synchronization between urgency level and its hex color."""
        expected_color = URGENCY_COLOR_MAPPING.get(self.urgency_level)
        if expected_color and self.urgency_color != expected_color:
            self.urgency_color = expected_color
        return self


class TriageRecordDBResponse(BaseModel):
    """Database model representation for persisted triage records."""

    id: Optional[str] = Field(None, description="Unique record identifier / UUID")
    created_at: Optional[str] = Field(None, description="Timestamp of record creation")
    user_id: Optional[str] = Field(None, description="Associated authenticated user identifier")
    pet_type: str = Field(..., description="Species or animal type")
    symptoms_description: str = Field(..., description="Narrative description of symptoms")
    image_url: Optional[str] = Field(None, description="Public URL of the uploaded pet photograph")
    urgency_level: str = Field(..., description="Clinical triage urgency level")
    urgency_color: str = Field(..., description="Hexadecimal color associated with urgency")
    preliminary_assessment: str = Field(..., description="Clinical evaluation summary")
    immediate_care_tips: List[str] = Field(
        default_factory=list,
        description="Immediate safe care guidelines",
    )
    recommended_facility_type: str = Field(..., description="Recommended facility classification")
    warning_disclaimer: Optional[str] = Field(
        None,
        description="Medical warning disclaimer",
    )
    user_lat: Optional[float] = Field(None, description="User geographical latitude")
    user_lng: Optional[float] = Field(None, description="User geographical longitude")

    model_config = {"extra": "ignore"}


class ClinicLocationResponse(BaseModel):
    """Geolocated veterinary clinic data model."""

    id: str = Field(..., description="Unique clinic identifier")
    name: str = Field(..., description="Commercial clinic or hospital name")
    phone: str = Field(..., description="Contact telephone for inquiries or emergencies")
    hours: str = Field(..., description="Operating business hours")
    distance_km: float = Field(..., description="Calculated distance in kilometers from user location")
    coordinates: List[float] = Field(..., description="Geographical coordinates [latitude, longitude]")
    address: str = Field(..., description="Street physical address")
    is_emergency_facility: bool = Field(
        ...,
        description="Flag indicating if the facility has 24/7 emergency and surgical capabilities",
    )


class ChatMessage(BaseModel):
    """Message item in follow-up chat history."""

    role: Literal["user", "model", "system"] = Field(..., description="Sender role")
    content: str = Field(..., description="Text content of message")


class ChatRequest(BaseModel):
    """Payload for contextual follow-up questions."""

    message: str = Field(..., description="User question regarding care or triage")
    conversation_history: List[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversational context",
    )


class ChatResponse(BaseModel):
    """Response model for AI follow-up chat."""

    reply: str = Field(..., description="Veterinary AI guidance reply")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    triage_id: str = Field(..., description="Associated triage case ID")
