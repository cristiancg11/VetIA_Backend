"""Data transfer schemas for Supabase authentication and health probes."""

import re
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field, field_validator

EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class RegisterRequest(BaseModel):
    """Payload for user registration in Supabase Auth."""

    email: str = Field(..., description="User email address")
    password: str = Field(
        ...,
        min_length=6,
        description="User password (minimum 6 characters required by Supabase Auth)",
    )
    full_name: Optional[str] = Field(
        None,
        description="Optional full name or pet owner name to store in user metadata",
    )

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        """Validate standard email structure without external email-validator dependency."""
        cleaned = v.strip().lower()
        if not EMAIL_REGEX.match(cleaned):
            raise ValueError("Formato de correo electrónico inválido.")
        return cleaned


class LoginRequest(BaseModel):
    """Payload for user login via Supabase Auth."""

    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=1, description="User password")

    @field_validator("email")
    @classmethod
    def validate_email_format(cls, v: str) -> str:
        """Validate standard email structure."""
        cleaned = v.strip().lower()
        if not EMAIL_REGEX.match(cleaned):
            raise ValueError("Formato de correo electrónico inválido.")
        return cleaned



class UserResponse(BaseModel):
    """Standardized user profile representation from Supabase."""

    id: str = Field(..., description="Unique UUID of the user in Supabase Auth")
    email: str = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, description="User's full name from metadata")
    created_at: Optional[str] = Field(None, description="ISO timestamp of user creation")
    user_metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata dictionary attached to Supabase Auth user",
    )


class RegisterResponse(BaseModel):
    """Structured response returned upon successful registration."""

    message: str = Field(..., description="Status message or confirmation instructions")
    user: Optional[UserResponse] = Field(None, description="Registered user information")
    session: Optional[Dict[str, Any]] = Field(
        None,
        description="Session object if instant login was enabled without email confirmation",
    )


class LoginResponse(BaseModel):
    """Authentication session and bearer token returned upon successful login."""

    access_token: str = Field(..., description="Supabase JWT access token")
    token_type: str = Field("bearer", description="Token scheme (bearer)")
    expires_in: Optional[int] = Field(None, description="Token validity lifetime in seconds")
    refresh_token: Optional[str] = Field(None, description="Refresh token for renewing session")
    user: Optional[UserResponse] = Field(None, description="Authenticated user info")


class SupabaseHealthResponse(BaseModel):
    """Structured health probe response confirming Supabase connectivity."""

    status: str = Field(..., description="Overall health state (healthy, degraded, error)")
    supabase_url: str = Field(..., description="Configured Supabase instance URL")
    database: str = Field(..., description="Database connection and table check result")
    auth: str = Field(..., description="Supabase Auth service status")
    timestamp: str = Field(..., description="ISO 8601 evaluation timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Diagnostic extra details")
