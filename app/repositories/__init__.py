"""Repositories package for data persistence abstraction."""

from .base_repository import BaseTriageRepository
from .supabase_repository import SupabaseTriageRepository

__all__ = ["BaseTriageRepository", "SupabaseTriageRepository"]
