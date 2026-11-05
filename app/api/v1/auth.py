"""Authentication and Supabase health endpoints for VetIA API."""

import datetime
import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client
from supabase_auth.errors import (
    AuthApiError,
    AuthInvalidCredentialsError,
    AuthWeakPasswordError,
)

from app.core.config import Settings, get_settings
from app.core.supabase import get_supabase_client
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    RegisterResponse,
    SupabaseHealthResponse,
    UserResponse,
)

logger = logging.getLogger(__name__)

auth_router = APIRouter(prefix="", tags=["Authentication & Health"])


def _extract_user_response(user: Any) -> Optional[UserResponse]:
    """Helper to convert Supabase User object or dict into a standard UserResponse schema."""
    if not user:
        return None

    user_id = str(getattr(user, "id", None) or (user.get("id") if isinstance(user, dict) else ""))
    email = str(getattr(user, "email", None) or (user.get("email") if isinstance(user, dict) else ""))
    user_metadata = getattr(user, "user_metadata", None) or (
        user.get("user_metadata", {}) if isinstance(user, dict) else {}
    )
    created_at = str(
        getattr(user, "created_at", None)
        or (user.get("created_at") if isinstance(user, dict) else "")
    )
    full_name = user_metadata.get("full_name") if isinstance(user_metadata, dict) else None

    return UserResponse(
        id=user_id,
        email=email,
        full_name=full_name,
        created_at=created_at or None,
        user_metadata=user_metadata or {},
    )


@auth_router.post(
    "/auth/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user in Supabase Auth",
    description="Registers user with email and password via Supabase Auth, storing optional metadata.",
)
async def register_user(
    request: RegisterRequest,
    supabase: Client = Depends(get_supabase_client),
) -> RegisterResponse:
    """Execute user registration in Supabase Auth with structured exception handling."""
    logger.info("[AUTH] Intentando registrar nuevo usuario: %s", request.email)

    if len(request.password) < 6:
        logger.warning(
            "[AUTH REGISTRATION REJECTED] Contraseña demasiado corta para el correo %s",
            request.email,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="La contraseña debe tener al menos 6 caracteres.",
        )

    signup_credentials: Dict[str, Any] = {
        "email": str(request.email),
        "password": request.password,
    }

    if request.full_name:
        signup_credentials["options"] = {"data": {"full_name": request.full_name.strip()}}

    try:
        auth_response = supabase.auth.sign_up(signup_credentials)

        # Detect silent duplicate user: Supabase returns user with empty identities if already registered
        user_identities = getattr(auth_response.user, "identities", None)
        if auth_response.user and (user_identities is not None and len(user_identities) == 0):
            logger.warning(
                "[AUTH REGISTRATION DUPLICATE] El correo ya está registrado en Supabase: %s",
                request.email,
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El correo electrónico ya se encuentra registrado. Inicia sesión o recupera tu contraseña.",
            )

        logger.info(
            "[AUTH REGISTRATION SUCCESS] Usuario registrado con éxito en Supabase Auth: %s (ID: %s)",
            request.email,
            getattr(auth_response.user, "id", "desconocido"),
        )

        user_data = _extract_user_response(auth_response.user)
        session_data = None
        message = "Usuario registrado exitosamente."

        if auth_response.session:
            session_data = {
                "access_token": auth_response.session.access_token,
                "token_type": auth_response.session.token_type or "bearer",
                "expires_in": auth_response.session.expires_in,
            }
        else:
            message = (
                "Usuario registrado exitosamente. Si la confirmación de correo está activa, "
                "por favor verifica tu bandeja de entrada antes de iniciar sesión."
            )

        return RegisterResponse(
            message=message,
            user=user_data,
            session=session_data,
        )

    except HTTPException:
        raise
    except AuthWeakPasswordError as exc:
        logger.error(
            "[SUPABASE AUTH ERROR] Contraseña débil al registrar %s: %s",
            request.email,
            exc,
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"La contraseña es demasiado débil: {str(exc)}",
        )
    except AuthApiError as exc:
        error_msg = str(exc.message if hasattr(exc, "message") else exc)
        status_code_val = getattr(exc, "status", None)
        logger.error(
            "[SUPABASE AUTH API ERROR] Fallo al registrar usuario '%s' (Status: %s): %s",
            request.email,
            status_code_val,
            error_msg,
            exc_info=True,
        )

        lowered = error_msg.lower()
        if "already registered" in lowered or "user already exists" in lowered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El correo electrónico ya se encuentra registrado.",
            )
        elif "security purposes" in lowered or "rate limit" in lowered:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Demasiadas solicitudes en poco tiempo. Por seguridad, espera unos segundos antes de reintentar.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error en Supabase Auth: {error_msg}",
            )
    except Exception as exc:
        logger.error(
            "[SUPABASE AUTH UNEXPECTED ERROR] Excepción inesperada registrando %s: %s",
            request.email,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante el registro: {str(exc)}",
        )


@auth_router.post(
    "/auth/login",
    response_model=LoginResponse,
    status_code=status.HTTP_200_OK,
    summary="Authenticate user and return Supabase session token",
    description="Authenticates credentials with Supabase Auth (sign_in_with_password) and issues session tokens.",
)
async def login_user(
    request: LoginRequest,
    supabase: Client = Depends(get_supabase_client),
) -> LoginResponse:
    """Authenticate existing user credentials via Supabase Auth."""
    logger.info("[AUTH] Intento de inicio de sesión para: %s", request.email)

    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": str(request.email),
            "password": request.password,
        })

        if not auth_response.session:
            logger.error(
                "[SUPABASE AUTH LOGIN] No se recibió objeto session para el usuario: %s",
                request.email,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No se pudo iniciar sesión. Verifica si tu correo requiere confirmación previa.",
            )

        logger.info(
            "[AUTH LOGIN SUCCESS] Inicio de sesión exitoso en Supabase Auth: %s",
            request.email,
        )

        return LoginResponse(
            access_token=auth_response.session.access_token,
            token_type=auth_response.session.token_type or "bearer",
            expires_in=auth_response.session.expires_in,
            refresh_token=auth_response.session.refresh_token,
            user=_extract_user_response(auth_response.user),
        )

    except HTTPException:
        raise
    except (AuthInvalidCredentialsError, AuthApiError) as exc:
        error_msg = str(getattr(exc, "message", str(exc)))
        logger.error(
            "[SUPABASE AUTH LOGIN ERROR] Fallo al autenticar usuario '%s': %s",
            request.email,
            error_msg,
        )
        lowered = error_msg.lower()
        if "invalid login credentials" in lowered:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Credenciales inválidas. Verifica tu correo y contraseña.",
            )
        elif "email not confirmed" in lowered:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="El correo electrónico no ha sido confirmado aún. Revisa tu bandeja de entrada.",
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error al autenticar con Supabase: {error_msg}",
            )
    except Exception as exc:
        logger.error(
            "[SUPABASE AUTH UNEXPECTED ERROR] Excepción inesperada autenticando %s: %s",
            request.email,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno durante la autenticación: {str(exc)}",
        )


@auth_router.get(
    "/health/supabase",
    response_model=SupabaseHealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check for Supabase database and Auth connectivity",
    description="Validates that Supabase credentials are valid and tests communication with Supabase database.",
)
async def check_supabase_health(
    settings: Settings = Depends(get_settings),
    supabase: Client = Depends(get_supabase_client),
) -> SupabaseHealthResponse:
    """Probe Supabase database and Auth connectivity."""
    db_status = "healthy"
    auth_status = "active"
    details: Dict[str, Any] = {
        "url": settings.SUPABASE_URL,
        "table": settings.SUPABASE_TABLE_NAME,
        "bucket": settings.SUPABASE_STORAGE_BUCKET,
    }

    try:
        # Perform lightweight count or select to verify DB connectivity
        response = (
            supabase.table(settings.SUPABASE_TABLE_NAME)
            .select("id")
            .limit(1)
            .execute()
        )
        details["rows_accessible"] = len(response.data) if response.data is not None else 0
        details["db_query"] = "success"
    except Exception as exc:
        db_status = f"unhealthy: {str(exc)}"
        details["db_error"] = str(exc)
        logger.error(
            "[SUPABASE HEALTH CHECK ERROR] No se pudo consultar la tabla '%s' en Supabase: %s",
            settings.SUPABASE_TABLE_NAME,
            exc,
            exc_info=True,
        )

    overall_status = "healthy" if "unhealthy" not in db_status else "degraded"

    return SupabaseHealthResponse(
        status=overall_status,
        supabase_url=settings.SUPABASE_URL,
        database=db_status,
        auth=auth_status,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        details=details,
    )
