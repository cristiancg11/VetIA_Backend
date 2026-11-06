"""Unit tests for field voice triage audio validation and structured clinical responses."""

import unittest
from fastapi import HTTPException

from app.api.v1.voice import (
    ALLOWED_AUDIO_EXTENSIONS,
    ALLOWED_AUDIO_MIME_TYPES,
    validate_and_resolve_audio_mime,
)
from app.schemas.voice_triage import VoiceTriageResponse
from app.services.voice_service import (
    clean_and_parse_voice_json,
    generate_voice_fallback,
    is_transient_voice_error,
)


class TestVoiceServiceValidation(unittest.TestCase):
    """Test suite for audio format validation, MIME resolution, and structured response parsing."""

    def test_valid_audio_extensions_and_mimes(self):
        """Test that accepted audio extensions resolve to appropriate MIME types."""
        cases = [
            ("audio_campo.wav", "audio/wav", "audio/wav"),
            ("registro.mp3", "audio/mpeg", "audio/mpeg"),
            ("relato.m4a", "audio/x-m4a", "audio/x-m4a"),
            ("nota_voz.webm", "audio/webm", "audio/webm"),
            ("grabacion.wav", "application/octet-stream", "audio/wav"),
            ("campo.mp3", "", "audio/mp3"),
            ("bovino.m4a", "", "audio/m4a"),
            ("equino.webm", "", "audio/webm"),
            ("ovino.ogg", "audio/ogg", "audio/ogg"),
            ("aves.aac", "audio/aac", "audio/aac"),
        ]
        for filename, ct, expected_mime in cases:
            with self.subTest(filename=filename, content_type=ct):
                resolved = validate_and_resolve_audio_mime(filename, ct)
                self.assertEqual(resolved, expected_mime)

    def test_invalid_audio_formats_raise_http_400(self):
        """Test that unsupported formats like txt, exe, pdf, or video raise HTTP 400."""
        invalid_cases = [
            ("documento.pdf", "application/pdf"),
            ("virus.exe", "application/x-msdownload"),
            ("foto.png", "image/png"),
            ("video.avi", "video/x-msvideo"),
            ("datos.txt", "text/plain"),
            ("archivo_sin_extension", "unknown/binary"),
        ]
        for filename, ct in invalid_cases:
            with self.subTest(filename=filename, content_type=ct):
                with self.assertRaises(HTTPException) as ctx:
                    validate_and_resolve_audio_mime(filename, ct)
                self.assertEqual(ctx.exception.status_code, 400)
                self.assertIn("Formato de audio no compatible", ctx.exception.detail)

    def test_allowed_constants_completeness(self):
        """Verify that required extensions (.wav, .mp3, .m4a, .webm) are present in allowed sets."""
        required_exts = {".wav", ".mp3", ".m4a", ".webm"}
        for ext in required_exts:
            self.assertIn(ext, ALLOWED_AUDIO_EXTENSIONS)

        required_mimes = {"audio/wav", "audio/mpeg", "audio/m4a", "audio/webm"}
        for mime in required_mimes:
            self.assertIn(mime, ALLOWED_AUDIO_MIME_TYPES)

    def test_clean_and_parse_voice_json_markdown_blocks(self):
        """Test parsing of raw Gemini markdown JSON blocks."""
        raw_markdown = """```json
        {
            "transcription": "Tengo una ternera de 3 meses con diarrea amarilla",
            "detected_species": "bovino",
            "extracted_symptoms": "Diarrea acuosa amarillenta, decúbito",
            "estimated_duration": "1 día",
            "preliminary_urgency": "URGENTE"
        }
        ```"""
        parsed = clean_and_parse_voice_json(raw_markdown)
        self.assertIsInstance(parsed, dict)
        self.assertEqual(parsed["detected_species"], "bovino")
        self.assertEqual(parsed["preliminary_urgency"], "URGENTE")
        self.assertIn("ternera", parsed["transcription"])

    def test_clean_and_parse_voice_json_raw_json(self):
        """Test parsing of clean raw JSON response without codeblocks."""
        raw_json = '{"transcription": "Caballo con cólico", "detected_species": "equino", "extracted_symptoms": "Dolor", "estimated_duration": "2h", "preliminary_urgency": "CRITICO"}'
        parsed = clean_and_parse_voice_json(raw_json)
        self.assertEqual(parsed["detected_species"], "equino")
        self.assertEqual(parsed["preliminary_urgency"], "CRITICO")

    def test_generate_voice_fallback_schema_compatibility(self):
        """Test that the deterministic fallback produces data conforming to VoiceTriageResponse."""
        fallback = generate_voice_fallback("Google Gemini API 503 test simulation")
        self.assertIsInstance(fallback, dict)
        response_model = VoiceTriageResponse(**fallback)
        self.assertIsNotNone(response_model.transcription)
        self.assertIsNotNone(response_model.extracted_symptoms)
        self.assertIn(response_model.preliminary_urgency, ["LEVE", "MODERADO", "URGENTE", "CRITICO"])

    def test_voice_triage_response_valid_schema(self):
        """Test that VoiceTriageResponse enforces valid schema data."""
        payload = {
            "transcription": "El lechón no mama y tiembla mucho",
            "detected_species": "porcino",
            "extracted_symptoms": "Hipotermia, temblores musculares, inanición",
            "estimated_duration": "6 horas",
            "preliminary_urgency": "URGENTE",
        }
        model = VoiceTriageResponse(**payload)
        self.assertEqual(model.detected_species, "porcino")
        self.assertEqual(model.preliminary_urgency, "URGENTE")
        self.assertEqual(model.estimated_duration, "6 horas")

    def test_transient_voice_error_detection(self):
        """Test transient error detection for 503, 429, and quota exceeded."""
        class Mock503Exception(Exception):
            code = 503

        class MockQuotaException(Exception):
            code = 429

            def __str__(self):
                return "429 RESOURCE_EXHAUSTED: Quota limit exceeded for audio model."

        self.assertTrue(is_transient_voice_error(Mock503Exception("Unavailable")))
        self.assertTrue(is_transient_voice_error(MockQuotaException()))
        self.assertFalse(is_transient_voice_error(ValueError("Invalid argument")))


if __name__ == "__main__":
    unittest.main()
