"""Pydantic schemas for voice-based clinical triage and field audio parsing."""

from typing import Optional
from pydantic import BaseModel, Field


class VoiceTriageResponse(BaseModel):
    """Structured clinical response extracted from audio voice reports recorded in the field."""

    transcription: str = Field(
        ...,
        description="Transcripción textual literal de lo narrado por el usuario o productor",
        examples=["La vaca primeriza lleva dos días sin comer y con la panza hinchada del lado izquierdo."],
    )
    detected_species: Optional[str] = Field(
        default=None,
        description="Especie animal identificada (bovino, equino, porcino, ovino, canino, felino, aves, cuy)",
        examples=["bovino"],
    )
    extracted_symptoms: str = Field(
        ...,
        description="Signos y síntomas clínicos aislados e identificados en el relato",
        examples=["Anorexia, distensión ruminal marcada en fosa paralumbar izquierda, debilidad."],
    )
    estimated_duration: Optional[str] = Field(
        default=None,
        description="Tiempo aproximado de evolución reportado en el relato",
        examples=["2 días"],
    )
    preliminary_urgency: str = Field(
        ...,
        description="Nivel de urgencia estimado (LEVE, MODERADO, URGENTE, CRITICO)",
        examples=["URGENTE"],
    )
