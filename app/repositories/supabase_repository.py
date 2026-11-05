"""Concrete Supabase repository implementation for triage records."""

import uuid
import logging
from typing import Any, Dict, List, Optional
from supabase import Client, create_client

from app.core.config import Settings, get_settings
from app.core.supabase import get_supabase_admin_client
from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.repositories.base_repository import BaseTriageRepository

logger = logging.getLogger(__name__)


class SupabaseTriageRepository(BaseTriageRepository):
    """
    Supabase implementation of the triage records repository.
    Executes synchronous database operations in dedicated worker threads
    to maintain non-blocking asynchronous endpoints with resilient fallback.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        thread_pool_manager: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._thread_pool_manager = thread_pool_manager or get_thread_pool_manager()
        self._table_name = self._settings.SUPABASE_TABLE_NAME

        if not self._settings.SUPABASE_URL or not self._settings.SUPABASE_KEY:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY must be configured in environment."
            )

        self._client: Client = get_supabase_admin_client(settings=self._settings)

    def _sync_save_record(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronous insert into Supabase triage_records table with graceful degradation."""
        try:
            response = self._client.table(self._table_name).insert(record_data).execute()
            if response.data and len(response.data) > 0:
                logger.info(
                    "[SUPABASE DB SUCCESS] Registro guardado exitosamente en '%s' (ID: %s)",
                    self._table_name,
                    response.data[0].get("id"),
                )
                return response.data[0]
            return record_data
        except Exception as exc:
            # If insert fails because user_id column is not yet migrated, retry without user_id
            if "user_id" in record_data and "user_id" in str(exc):
                try:
                    clean_data = {k: v for k, v in record_data.items() if k != "user_id"}
                    retry_res = self._client.table(self._table_name).insert(clean_data).execute()
                    if retry_res.data and len(retry_res.data) > 0:
                        logger.info(
                            "[SUPABASE DB RETRY SUCCESS] Registro guardado sin user_id en '%s' (ID: %s)",
                            self._table_name,
                            retry_res.data[0].get("id"),
                        )
                        return retry_res.data[0]
                    return clean_data
                except Exception as retry_err:
                    logger.error(
                        "[SUPABASE DB ERROR] Reintento de inserción sin user_id falló en '%s': %s",
                        self._table_name,
                        retry_err,
                    )

            logger.error(
                "[SUPABASE DB INSERT ERROR] Fallo al insertar registro en la tabla '%s'. Detalle: %s. "
                "Si el código es 42501 (RLS policy violation), verifica las políticas RLS en Supabase "
                "o configura SUPABASE_SERVICE_ROLE_KEY.",
                self._table_name,
                exc,
                exc_info=True,
            )
            fallback = dict(record_data)
            if "id" not in fallback:
                fallback["id"] = str(uuid.uuid4())
            return fallback

    def _sync_get_recent(
        self,
        limit: int,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Synchronous select query for recent records, optionally filtered by user_id."""
        try:
            if user_id:
                try:
                    user_filtered = (
                        self._client.table(self._table_name)
                        .select("*")
                        .eq("user_id", user_id)
                        .order("created_at", desc=True)
                        .limit(limit)
                        .execute()
                    )
                    return user_filtered.data or []
                except Exception as filter_err:
                    logger.warning(
                        "[SUPABASE DB NOTICE] Filtro por user_id notice: %s. Usando consulta general.",
                        filter_err,
                    )

            response = (
                self._client.table(self._table_name)
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            return response.data or []
        except Exception as exc:
            logger.error(
                "[SUPABASE DB SELECT ERROR] Fallo al obtener registros de '%s': %s",
                self._table_name,
                exc,
                exc_info=True,
            )
            return []

    def _sync_get_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Synchronous fetch of a single triage case by ID."""
        try:
            response = (
                self._client.table(self._table_name)
                .select("*")
                .eq("id", record_id)
                .limit(1)
                .execute()
            )
            if response.data and len(response.data) > 0:
                return response.data[0]
            return None
        except Exception as exc:
            logger.error(
                "[SUPABASE DB SELECT ERROR] Fallo al obtener registro con id '%s' de '%s': %s",
                record_id,
                self._table_name,
                exc,
                exc_info=True,
            )
            return None

    async def save_triage_record(self, record_data: Dict[str, Any]) -> Dict[str, Any]:
        """Asynchronously save triage record by delegating to worker thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_save_record,
            record_data,
        )

    async def get_recent_records(
        self,
        limit: int = 10,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Asynchronously fetch recent records by delegating to worker thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_get_recent,
            limit,
            user_id,
        )

    async def get_record_by_id(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Asynchronously fetch record by ID via worker thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_get_by_id,
            record_id,
        )
