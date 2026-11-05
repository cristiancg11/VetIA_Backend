"""Supabase client initialization and singleton management."""

import logging
from typing import Optional
from supabase import Client, create_client

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_supabase_client: Optional[Client] = None
_supabase_admin_client: Optional[Client] = None


def get_supabase_client(settings: Optional[Settings] = None) -> Client:
    """
    Provide or instantiate a singleton Supabase Client using standard (publishable/anon) key.
    Used for Auth flows (sign_up, sign_in_with_password) and authenticated user requests.
    """
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    cfg = settings or get_settings()

    if not cfg.SUPABASE_URL or not cfg.SUPABASE_KEY:
        logger.error(
            "[SUPABASE INIT ERROR] SUPABASE_URL o SUPABASE_KEY no estan configuradas en el entorno."
        )
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured in environment.")

    try:
        _supabase_client = create_client(
            supabase_url=cfg.SUPABASE_URL,
            supabase_key=cfg.SUPABASE_KEY,
        )
        logger.info(
            "[SUPABASE INIT] Cliente de Supabase inicializado exitosamente (URL: %s)",
            cfg.SUPABASE_URL,
        )
        return _supabase_client
    except Exception as exc:
        logger.error(
            "[SUPABASE INIT ERROR] Fallo al crear cliente de Supabase: %s",
            exc,
            exc_info=True,
        )
        raise


def get_supabase_admin_client(settings: Optional[Settings] = None) -> Client:
    """
    Provide a Supabase client with administrative privileges.
    If SUPABASE_SERVICE_ROLE_KEY is present, it will be used to bypass RLS policies
    for backend database inserts and storage management. Otherwise falls back to
    standard client.
    """
    global _supabase_admin_client
    if _supabase_admin_client is not None:
        return _supabase_admin_client

    cfg = settings or get_settings()
    if cfg.SUPABASE_SERVICE_ROLE_KEY and cfg.SUPABASE_SERVICE_ROLE_KEY.strip():
        try:
            _supabase_admin_client = create_client(
                supabase_url=cfg.SUPABASE_URL,
                supabase_key=cfg.SUPABASE_SERVICE_ROLE_KEY,
            )
            logger.info(
                "[SUPABASE ADMIN] Cliente administrativo inicializado con Service Role Key."
            )
            return _supabase_admin_client
        except Exception as exc:
            logger.error(
                "[SUPABASE ADMIN ERROR] Fallo al inicializar cliente con Service Role: %s. Usando cliente estandar.",
                exc,
                exc_info=True,
            )

    # Fallback to standard client
    return get_supabase_client(settings=cfg)
