"""Data transfer models and schemas for pet triage, clinic localization, follow-up chat, and database persistence."""

import re
import uuid
from enum import Enum
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class DiagnosticQuestion(BaseModel):
    """Pregunta de descarte clínico para confirmación diagnóstica."""
    id: str = Field(..., description="Identificador único de la pregunta (ej. q1, q2)")
    question: str = Field(..., description="Texto de la pregunta orientada al tutor")
    type: Optional[str] = Field(
        default="yes_no",
        description="Tipo de respuesta requerida (yes_no)",
    )


class TriageUrgencyLevel(str, Enum):
    """Clinical urgency severity enumeration."""
    CRITICAL = "CRITICAL"
    MODERATE = "MODERATE"
    MILD = "MILD"


class TriageAnalysisUrgency(str, Enum):
    """Clinical urgency severity for the multimodal AI triage response."""
    CRITICO = "CRITICO"
    URGENTE = "URGENTE"
    MODERADO = "MODERADO"
    LEVE = "LEVE"


class TriageAnalysisColor(str, Enum):
    """Color codes for the multimodal AI triage response."""
    RED = "RED"
    ORANGE = "ORANGE"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class TriageAnalysisResponse(BaseModel):
    """
    Structured multimodal veterinary triage response schema.
    Conforms to the Manchester Triage System adaptation with Chain-of-Thought clinical reasoning.
    """
    urgency_level: str = Field(
        ...,
        description="Nivel de urgencia clínica: CRITICO | URGENTE | MODERADO | LEVE",
    )
    color_code: str = Field(
        ...,
        description="Código de color según Manchester Triage System: RED | ORANGE | YELLOW | GREEN",
    )
    visual_findings: str = Field(
        ...,
        description="Detalles observados con precisión en la imagen (colorimetría, textura, consistencia y nitidez fotográfica)",
    )
    presumptive_diagnosis: List[str] = Field(
        ...,
        description="Lista de diagnósticos presuntivos clínicos correlacionados con la imagen y síntomas",
    )
    vital_risks: List[str] = Field(
        ...,
        description="Lista de riesgos vitales o complicaciones potenciales detectadas",
    )
    first_aid_actions: List[str] = Field(
        ...,
        description="Pautas secuenciales de primeros auxilios y estabilización inmediata y segura para el tutor",
    )
    warnings_contraindications: List[str] = Field(
        ...,
        description="Medicamentos o acciones estrictamente prohibidas y contraindicadas (ej. NO inducir vómito, NO dar ibuprofeno)",
    )
    estimated_fluid_deficit_percent: float = Field(
        default=0.0,
        description="Porcentaje numérico estimado de déficit hídrico o deshidratación (0.0 a 15.0%)",
    )
    triage_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Identificador único del caso de triaje (UUID)",
    )
    diagnostic_questions: List[DiagnosticQuestion] = Field(
        default_factory=list,
        description="Lista de 3 a 4 preguntas de descarte clínico para confirmación diagnóstica",
    )

    @field_validator("urgency_level", mode="before")
    @classmethod
    def normalize_urgency_level(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if "CRITIC" in s:
            return "CRITICO"
        if "URGENT" in s:
            return "URGENTE"
        if "MODERAD" in s:
            return "MODERADO"
        if "LEV" in s:
            return "LEVE"
        return s

    @field_validator("color_code", mode="before")
    @classmethod
    def normalize_color_code(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if "ROJO" in s or "RED" in s:
            return "RED"
        if "NARANJA" in s or "ORANGE" in s:
            return "ORANGE"
        if "AMARILLO" in s or "YELLOW" in s:
            return "YELLOW"
        if "VERDE" in s or "GREEN" in s:
            return "GREEN"
        return s

    @field_validator("estimated_fluid_deficit_percent", mode="before")
    @classmethod
    def normalize_fluid_deficit(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = re.sub(r"[^\d.]", "", v)
            return float(clean) if clean else 0.0
        return 0.0




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


class ConfirmDiagnosisRequest(BaseModel):
    """Payload para reevaluación y confirmación del triaje con respuestas del test de descarte."""

    triage_id: str = Field(..., description="Identificador único del caso de triaje previo")
    answers: Dict[str, Any] = Field(
        ...,
        description="Respuestas del test de descarte clínico (ej. {'q1': True, 'q2': False, 'q3': True})",
    )
    initial_symptoms: Optional[str] = Field(
        None,
        description="Síntomas iniciales del paciente (opcional si se rescatan del contexto)",
    )
    species: Optional[str] = Field(
        None,
        description="Especie del paciente (ej. Canino, Felino)",
    )
    initial_urgency_level: Optional[str] = Field(
        None,
        description="Nivel preliminar de urgencia antes de responder las preguntas",
    )
    questions: Optional[List[DiagnosticQuestion]] = Field(
        default=None,
        description="Listado opcional de preguntas para enriquecer el contexto",
    )


class ConfirmDiagnosisResponse(BaseModel):
    """Respuesta tras reevaluación de urgencia con las preguntas de descarte clínico."""

    triage_id: str = Field(..., description="Identificador del triaje")
    urgency_level: str = Field(
        ...,
        description="Nivel de urgencia definitivo: CRITICO | URGENTE | MODERADO | LEVE",
    )
    color_code: str = Field(
        ...,
        description="Código de color según Manchester Triage System: RED | ORANGE | YELLOW | GREEN",
    )
    is_escalated: bool = Field(
        ...,
        description="Indica si la urgencia escaló tras las respuestas de descarte",
    )
    clinical_reassessment: str = Field(
        ...,
        description="Razonamiento clínico de la reevaluación considerando el test confirmatorio",
    )
    presumptive_diagnosis: List[str] = Field(
        default_factory=list,
        description="Diagnósticos presuntivos actualizados",
    )
    vital_risks: List[str] = Field(
        default_factory=list,
        description="Riesgos vitales actualizados",
    )
    first_aid_actions: List[str] = Field(
        default_factory=list,
        description="Pautas secuenciales de primeros auxilios y estabilización inmediata",
    )
    warnings_contraindications: List[str] = Field(
        default_factory=list,
        description="Contraindicaciones médicas absolutas (NO medicamentos humanos, etc.)",
    )
    estimated_fluid_deficit_percent: float = Field(
        default=0.0,
        description="Déficit hídrico o porcentaje de deshidratación estimado",
    )
    recommended_facility_type: str = Field(
        default="Clínica 24 horas con quirófano",
        description="Tipo de centro de atención veterinaria recomendado",
    )

    @field_validator("urgency_level", mode="before")
    @classmethod
    def normalize_urgency_level(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if "CRITIC" in s:
            return "CRITICO"
        if "URGENT" in s:
            return "URGENTE"
        if "MODERAD" in s:
            return "MODERADO"
        if "LEV" in s:
            return "LEVE"
        return s

    @field_validator("color_code", mode="before")
    @classmethod
    def normalize_color_code(cls, v: Any) -> str:
        s = str(v).strip().upper()
        if "ROJO" in s or "RED" in s:
            return "RED"
        if "NARANJA" in s or "ORANGE" in s:
            return "ORANGE"
        if "AMARILLO" in s or "YELLOW" in s:
            return "YELLOW"
        if "VERDE" in s or "GREEN" in s:
            return "GREEN"
        return s

    @field_validator("estimated_fluid_deficit_percent", mode="before")
    @classmethod
    def normalize_fluid_deficit(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            clean = re.sub(r"[^\d.]", "", v)
            return float(clean) if clean else 0.0
        return 0.0


class EmergencyChatRequest(BaseModel):
    """Payload para el asistente conversacional de primeros auxilios y triaje."""

    triage_id: str = Field(..., description="Identificador del caso de triaje")
    message: Optional[str] = Field(None, description="Pregunta o consulta urgente del tutor")
    user_message: Optional[str] = Field(None, description="Alias para la consulta del tutor")
    context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Contexto clínico opcional del paciente (especie, urgencia, hallazgos)",
    )
    conversation_history: Optional[List[ChatMessage]] = Field(
        default_factory=list,
        description="Historial previo de mensajes de la conversación",
    )

    def get_query(self) -> str:
        """Returns the non-empty user query from either user_message or message."""
        return (self.user_message or self.message or "").strip()


class EmergencyChatResponse(BaseModel):
    """Respuesta del asistente de primeros auxilios en emergencias veterinarias."""

    triage_id: str = Field(..., description="Identificador del triaje")
    reply: str = Field(..., description="Instrucciones precisas y empáticas de primeros auxilios")
    timestamp: str = Field(..., description="Timestamp ISO 8601")
    disclaimer: str = Field(
        default="VetIA no reemplaza la atención veterinaria presencial ni prescribe fármacos bajo receta.",
        description="Aviso legal y médico obligatorio",
    )
