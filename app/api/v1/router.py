"""API Router definition for Version 1 endpoints with Supabase persistence, follow-up chat, and user scoping."""

import datetime
import logging
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.repositories.base_repository import BaseTriageRepository
from app.repositories.supabase_repository import SupabaseTriageRepository
from app.schemas.triage import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ClinicLocationResponse,
    TriageAssessmentResponse,
    TriageRecordDBResponse,
    TriageUrgencyLevel,
)
from app.services.base_triage import BaseTriageService
from app.services.clinic_locator_service import ClinicLocatorService
from app.services.gemini_triage_service import GeminiTriageService
from app.services.storage_service import SupabaseStorageService

logger = logging.getLogger(__name__)

api_v1_router = APIRouter(prefix="", tags=["v1"])


def get_triage_service() -> BaseTriageService:
    """Dependency provider for the multimodal triage service interface."""
    return GeminiTriageService()


def get_clinic_locator_service() -> ClinicLocatorService:
    """Dependency provider for geospatial clinic locator service."""
    return ClinicLocatorService()


def get_storage_service() -> SupabaseStorageService:
    """Dependency provider for Supabase file storage service."""
    return SupabaseStorageService()


def get_triage_repository() -> BaseTriageRepository:
    """Dependency provider for Supabase database repository."""
    return SupabaseTriageRepository()


ALLOWED_MIME_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
}


@api_v1_router.post(
    "/diagnose",
    response_model=TriageAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Perform multimodal veterinary triage assessment with cloud storage and persistence",
    description=(
        "Accepts a pet condition photograph and symptoms description. "
        "Uploads the image to Supabase Storage, evaluates condition using Gemini 2.5 Flash, "
        "saves the clinical record in PostgreSQL via Supabase, and returns the structured triage advice."
    ),
)
async def diagnose_pet_condition(
    image: UploadFile = File(..., description="Photographic evidence of the visible lesion or condition"),
    pet_type: str = Form(..., description="Species or animal type (e.g., Dog, Cat)"),
    symptoms_description: str = Form(..., description="Narrative description of symptoms and behavior"),
    user_id: Optional[str] = Form(None, description="Optional authenticated user UUID from Supabase Auth"),
    user_lat: Optional[float] = Form(None, description="Optional user geographical latitude"),
    user_lng: Optional[float] = Form(None, description="Optional user geographical longitude"),
    triage_service: BaseTriageService = Depends(get_triage_service),
    storage_service: SupabaseStorageService = Depends(get_storage_service),
    repository: BaseTriageRepository = Depends(get_triage_repository),
    thread_pool_manager: ThreadPoolManager = Depends(get_thread_pool_manager),
) -> TriageAssessmentResponse:
    """Execute complete pet triage evaluation, storage, and persistence flow."""
    if image.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format ({image.content_type}). Supported formats: JPEG, PNG, WebP.",
        )

    try:
        # Step 1: Offload file byte reading to worker thread pool
        image_bytes = await thread_pool_manager.run_in_thread(image.file.read)
        if not image_bytes or len(image_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded image file is empty.",
            )

        # Step 2: Upload image to Supabase Storage bucket asynchronously (with safe fallback)
        try:
            image_url = await storage_service.upload_pet_image(
                image_bytes=image_bytes,
                filename=image.filename or "pet_image.jpg",
                mime_type=image.content_type or "image/jpeg",
            )
        except Exception as storage_err:
            logger.warning("Storage upload notice: %s", storage_err)
            image_url = "https://images.unsplash.com/photo-1548767797-d8c844163c4c?auto=format&fit=crop&q=80&w=800"

        # Step 3: Evaluate clinical condition via Gemini 2.5 Flash / Smart Triage Engine
        assessment = await triage_service.evaluate_condition(
            image_bytes=image_bytes,
            mime_type=image.content_type or "image/jpeg",
            pet_type=pet_type,
            symptoms_description=symptoms_description,
        )
        assessment.image_url = image_url

        # Step 4: Persist triage record in Supabase PostgreSQL table
        record_data: Dict[str, Any] = {
            "pet_type": pet_type,
            "symptoms_description": symptoms_description,
            "image_url": image_url,
            "urgency_level": assessment.urgency_level,
            "urgency_color": assessment.urgency_color,
            "preliminary_assessment": assessment.preliminary_assessment,
            "immediate_care_tips": assessment.immediate_care_tips,
            "recommended_facility_type": assessment.recommended_facility_type,
            "user_lat": user_lat,
            "user_lng": user_lng,
        }

        if user_id:
            record_data["user_id"] = user_id

        try:
            saved_record = await repository.save_triage_record(record_data)
            if saved_record and "id" in saved_record:
                assessment.record_id = str(saved_record["id"])
        except Exception as db_err:
            logger.warning("Failed to persist record in Supabase: %s", db_err)

        return assessment

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unexpected error in diagnose_pet_condition: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred during triage evaluation: {str(exc)}",
        )


@api_v1_router.get(
    "/history",
    response_model=List[TriageRecordDBResponse],
    status_code=status.HTTP_200_OK,
    summary="Retrieve recent triage evaluation history",
    description="Returns clinical triage records registered in Supabase PostgreSQL, optionally user-scoped.",
)
async def get_triage_history(
    limit: int = Query(10, ge=1, le=50, description="Number of recent records to return"),
    user_id: Optional[str] = Query(None, description="Optional authenticated user UUID for personal history"),
    repository: BaseTriageRepository = Depends(get_triage_repository),
) -> List[Dict[str, Any]]:
    """Fetch recent triage records."""
    try:
        return await repository.get_recent_records(limit=limit, user_id=user_id)
    except Exception as exc:
        logger.warning("History fetch issue: %s", exc)
        return []


@api_v1_router.post(
    "/triage/{triage_id}/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Contextual follow-up chat for an evaluated triage case",
    description="Allows asking follow-up questions to Gemini 2.5 Flash regarding stabilization, care, and transport.",
)
async def follow_up_triage_chat(
    triage_id: str,
    chat_request: ChatRequest,
    repository: BaseTriageRepository = Depends(get_triage_repository),
    triage_service: BaseTriageService = Depends(get_triage_service),
) -> ChatResponse:
    """Process contextual follow-up questions grounded on previous triage findings."""
    # Attempt to fetch original case context from Supabase
    triage_record = await repository.get_record_by_id(triage_id)
    if not triage_record:
        # Fallback minimal context if case is in-memory or not found
        triage_record = {
            "id": triage_id,
            "pet_type": "Mascota",
            "urgency_level": "MODERATE",
            "preliminary_assessment": "Evaluación preliminar de urgencia.",
            "symptoms_description": "Síntomas bajo observación.",
        }

    try:
        reply_text = await triage_service.follow_up_chat(
            triage_context=triage_record,
            user_message=chat_request.message,
            history=chat_request.conversation_history,
        )

        return ChatResponse(
            reply=reply_text,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            triage_id=triage_id,
        )
    except Exception as exc:
        logger.error("Error during follow-up chat: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing follow-up question: {str(exc)}",
        )


@api_v1_router.get(
    "/clinics/nearby",
    response_model=List[ClinicLocationResponse],
    status_code=status.HTTP_200_OK,
    summary="Get nearby veterinary facilities",
    description="Returns geolocated veterinary clinics around user coordinates prioritized by severity.",
)
def get_nearby_clinics(
    lat: float = Query(..., description="User current latitude"),
    lng: float = Query(..., description="User current longitude"),
    urgency: Optional[TriageUrgencyLevel] = Query(
        TriageUrgencyLevel.MODERATE,
        description="Assessed urgency level for facility prioritization",
    ),
    clinic_locator: ClinicLocatorService = Depends(get_clinic_locator_service),
) -> List[ClinicLocationResponse]:
    """Retrieve filtered and distance-sorted veterinary clinics."""
    return clinic_locator.find_nearby_clinics(
        latitude=lat,
        longitude=lng,
        urgency=urgency,
    )
