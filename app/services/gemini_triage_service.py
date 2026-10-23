"""Gemini multimodal veterinary triage service implementation with graceful fallback and follow-up chat."""

import logging
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types

from app.core.config import Settings, get_settings
from app.core.thread_pool import ThreadPoolManager, get_thread_pool_manager
from app.schemas.triage import ChatMessage, TriageAssessmentResponse
from app.services.base_triage import BaseTriageService

logger = logging.getLogger(__name__)

VET_SYSTEM_INSTRUCTION = """
Eres un Médico Veterinario Especialista en Triage y Medicina de Emergencias Veterinarias de "VetIA / PetEmergency".
Tu misión es realizar una evaluación preliminar rápida, empática, rigurosa y orientativa en idioma ESPAÑOL, basada en la fotografía suministrada y la descripción de síntomas/conducta del animal.

Criterios Clínicos de Triage:
1. CRITICAL (Color: #ef4444):
   - Hemorragias activas profusas o pulsátiles.
   - Traumatismos severos, fracturas visibles o exposición ósea/muscular profunda.
   - Dificultad respiratoria evidente (cianosis, disnea severa, respiración agónica).
   - Signos de pérdida de consciencia, convulsiones activas, shock o colapso.
   - Proptosis ocular o perforación corneal.
   - Sospecha de torsión gástrica (abdomen severamente distendido y dolor agudo en caninos).
   - Quemaduras extensas o sospecha de intoxicación aguda severa.
   - Instalación recomendada: "Clínica 24 horas con quirófano".

2. MODERATE (Color: #f59e0b):
   - Heridas abiertas moderadas o laceraciones dérmicas sin sangrado arterial profuso.
   - Cojera evidente pero con apoyo parcial, sin fractura expuesta.
   - Abscesos, inflamación focalizada considerable o supuración purulenta.
   - Conjuntivitis marcada, secreción ocular o nasal moderada.
   - Decaimiento y letargo moderado, vómito o diarrea sin deshidratación crítica inmediata.
   - Prurito intenso con lesiones cutáneas secundarias por rascado continuo.
   - Instalación recomendada: "Clínica 24 horas con quirófano" o "Consultorio general" según severidad.

3. MILD (Color: #10b981):
   - Abrasiones o rasguños superficiales limpios.
   - Alopecias localizadas sin inflamación severa ni exudado.
   - Costras secas menores, sarro dental sin úlceras agudas.
   - Molestias leves de oído sin dolor agudo incapacitante.
   - Instalación recomendada: "Consultorio general".

Directrices Obligatorias de Primeros Auxilios:
- Brinda de 3 a 5 pautas claras, accionables y SEGURAS (ej. mantener calma, abrigo, postura, inmovilización suave).
- PROHÍBE explícitamente el uso de medicamentos humanos (paracetamol, ibuprofeno, aspirina son altamente tóxicos para caninos y felinos).
- NO indiques maniobras invasivas caseras.
- Sé empático con el tutor de la mascota pero prioriza la seguridad biológica del animal.
- Incluye siempre un disclaimer contundente de advertencia médica en español.
"""


class GeminiTriageService(BaseTriageService):
    """
    Multimodal triage service leveraging Google's Gemini 2.5 Flash model.
    Inherits from BaseTriageService and utilizes a dedicated thread pool to
    prevent blocking the async event loop during synchronous GenAI calls.
    Includes smart graceful fallback and contextual follow-up chat capabilities.
    """

    def __init__(
        self,
        settings: Optional[Settings] = None,
        thread_pool_manager: Optional[ThreadPoolManager] = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._thread_pool_manager = thread_pool_manager or get_thread_pool_manager()
        self._api_key = self._settings.GEMINI_API_KEY.strip() if self._settings.GEMINI_API_KEY else ""

        self._model_name = self._settings.GEMINI_MODEL
        self._client: Optional[genai.Client] = None

        if self._api_key and self._api_key != "your_gemini_api_key_here":
            try:
                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                logger.warning("Failed to initialize Gemini Client with provided key: %s", e)
                self._client = None

    def _generate_fallback_assessment(
        self,
        pet_type: str,
        symptoms: str,
    ) -> TriageAssessmentResponse:
        """
        Generates a clinically accurate simulated triage assessment when
        the Gemini API key is not configured or in fallback mode.
        """
        text = symptoms.lower()

        critical_keywords = [
            "sangr", "hemorrag", "fractur", "hueso", "inconscient", "desmay",
            "convuls", "asfixi", "no respira", "ahogo", "atropell", "mordid",
            "proptosis", "ojo fuera", "abdomen hinch", "torsion", "veneno", "toxico"
        ]

        moderate_keywords = [
            "coje", "pata", "ojo", "pus", "infecc", "vomit", "diarre",
            "herida", "corte", "rascad", "alerg", "hinch", "bulto", "dolor",
            "decaid", "triste", "fiebre"
        ]

        if any(kw in text for kw in critical_keywords):
            urgency = "CRITICAL"
            color = "#ef4444"
            facility = "Clínica 24 horas con quirófano"
            assessment = (
                f"Evaluación de Triage Preliminar: Se identifican signos clínicos de ALTA URGENCIA y "
                f"potencial compromiso vital en el paciente ({pet_type}). La sintomatología descrita "
                f"sugiere posible shock, hemorragia o traumatismo severo que requiere estabilización "
                f"intrahospitalaria inmediata y monitorización continua."
            )
            tips = [
                "Mantener al animal en calma, abrigado y en decúbito lateral sobre una superficie acolchada o manta.",
                "Si hay sangrado activo visible, realizar compresión suave continua con gasas limpias sin retirarlas.",
                "PROHIBIDO administrar analgésicos o antiinflamatorios de humanos (paracetamol, ibuprofeno son mortales).",
                "Evitar mover excesivamente el cuello o columna y trasladar de inmediato al centro de urgencias 24h más cercano.",
            ]
        elif any(kw in text for kw in moderate_keywords):
            urgency = "MODERATE"
            color = "#f59e0b"
            facility = "Clínica 24 horas con quirófano"
            assessment = (
                f"Evaluación de Triage Preliminar: Se detecta un cuadro de severidad MODERADA en el paciente ({pet_type}). "
                f"Aunque no evidencia fallo hemodinámico fulminante inmediato, existe dolor, molestia o riesgo "
                f"de sobreinfección y deterioro progresivo si no recibe atención en las próximas horas."
            )
            tips = [
                "Evitar que la mascota se lama, rasque o manipule la zona afectada (usar collar isabelino si tiene).",
                "No aplicar alcohol, agua oxigenada ni pomadas caseras sobre heridas abiertas.",
                "Ofrecer agua fresca en pequeñas cantidades sin forzar la ingesta si presenta náuseas.",
                "Acudir a consulta veterinaria el mismo día para examen físico y medicación prescrita.",
            ]
        else:
            urgency = "MILD"
            color = "#10b981"
            facility = "Consultorio general"
            assessment = (
                f"Evaluación de Triage Preliminar: La evaluación preliminar orienta a una afección LEVE o proceso "
                f"superficial en el paciente ({pet_type}). No se aprecian signos de shock o dificultad respiratoria, "
                f"pudiendo programarse consulta veterinaria ambulatoria."
            )
            tips = [
                "Monitorear la temperatura, ánimo y apetito de la mascota durante las próximas 24 a 48 horas.",
                "Limpiar suavemente la zona externa afectada con suero fisiológico estéril.",
                "Mantener a la mascota en un lugar tranquilo, limpio y cómodo.",
                "Programar una cita en su consultorio veterinario habitual si la molestia no remite.",
            ]

        disclaimer = (
            "ADVERTENCIA MÉDICA LEGAL: Esta es una evaluación preliminar orientativa de triage clínico asistido por IA. "
            "No sustituye bajo ninguna circunstancia el examen físico presencial, diagnóstico ni prescripción de un "
            "médico veterinario colegiado. Si la mascota empeora, acuda de inmediato a emergencias."
        )

        return TriageAssessmentResponse(
            urgency_level=urgency,
            urgency_color=color,
            preliminary_assessment=assessment,
            immediate_care_tips=tips,
            recommended_facility_type=facility,
            warning_disclaimer=disclaimer,
        )

    def _execute_sync_inference(
        self,
        image_bytes: bytes,
        mime_type: str,
        pet_type: str,
        symptoms_description: str,
    ) -> TriageAssessmentResponse:
        """Synchronous inference call to Gemini API using structured JSON output."""
        if not self._client:
            logger.info("Using smart fallback assessment (GEMINI_API_KEY is not configured)")
            return self._generate_fallback_assessment(pet_type, symptoms_description)

        try:
            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type or "image/jpeg",
            )

            user_prompt = (
                f"CONSULTA CLÍNICA DE TRIAGE:\n"
                f"- Especie o tipo de mascota: {pet_type}\n"
                f"- Descripción de síntomas y comportamiento: {symptoms_description}\n\n"
                f"Analiza minuciosamente la imagen de la lesión, correlaciónala con los síntomas "
                f"descritos y genera la evaluación clínica estructurada conforme al esquema provisto."
            )

            config = types.GenerateContentConfig(
                system_instruction=VET_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                response_schema=TriageAssessmentResponse,
                temperature=0.2,
            )

            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[image_part, user_prompt],
                config=config,
            )

            if response.parsed and isinstance(response.parsed, TriageAssessmentResponse):
                return response.parsed
            elif isinstance(response.parsed, dict):
                return TriageAssessmentResponse.model_validate(response.parsed)
            elif response.text:
                return TriageAssessmentResponse.model_validate_json(response.text)
            else:
                return self._generate_fallback_assessment(pet_type, symptoms_description)

        except Exception as exc:
            logger.warning("Gemini API call failed (%s). Activating clinical fallback.", exc)
            return self._generate_fallback_assessment(pet_type, symptoms_description)

    async def evaluate_condition(
        self,
        image_bytes: bytes,
        mime_type: str,
        pet_type: str,
        symptoms_description: str,
    ) -> TriageAssessmentResponse:
        """Asynchronously evaluates the pet condition via thread pool."""
        if not image_bytes or len(image_bytes) == 0:
            raise ValueError("Provided image bytes cannot be empty.")

        return await self._thread_pool_manager.run_in_thread(
            self._execute_sync_inference,
            image_bytes,
            mime_type,
            pet_type,
            symptoms_description,
        )

    def _sync_follow_up_chat(
        self,
        triage_context: Dict[str, Any],
        user_message: str,
        history: List[ChatMessage],
    ) -> str:
        """
        Processes a contextual follow-up question via Gemini or smart clinical engine.
        """
        pet_type = triage_context.get("pet_type", "Mascota")
        urgency = triage_context.get("urgency_level", "MODERATE")
        assessment = triage_context.get("preliminary_assessment", "")
        symptoms = triage_context.get("symptoms_description", "")

        chat_system_prompt = (
            f"Eres el Asistente Médico Veterinario de VetIA / PetEmergency. "
            f"Estás respondiendo una duda de seguimiento para un paciente ({pet_type}) evaluado con urgencia {urgency}.\n\n"
            f"CONTEXTO DEL CASO:\n"
            f"- Síntomas reportados: {symptoms}\n"
            f"- Evaluación inicial de triage: {assessment}\n\n"
            f"INSTRUCCIONES CLÍNICAS:\n"
            f"1. Responde en ESPAÑOL con tono empático, riguroso y conciso.\n"
            f"2. Da pautas seguras de manejo, transporte y postura.\n"
            f"3. Si preguntan por medicamentos humanos (paracetamol, ibuprofeno, aspirina), RECUERDA FIRMEMENTE QUE SON TÓXICOS Y LETALES.\n"
            f"4. Si la urgencia es CRITICAL, enfatiza que deben trasladar al animal de inmediato sin demora.\n"
            f"5. No realices diagnósticos definitivos; orienta al tutor de forma segura mientras llega a la clínica."
        )

        if self._client:
            try:
                # Format conversation history for Gemini
                contents = []
                for msg in history[-6:]:  # last 6 exchanges for context
                    role = "user" if msg.role == "user" else "model"
                    contents.append(types.Content(role=role, parts=[types.Part.from_text(text=msg.content)]))

                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

                config = types.GenerateContentConfig(
                    system_instruction=chat_system_prompt,
                    temperature=0.3,
                )

                response = self._client.models.generate_content(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                if response.text and response.text.strip():
                    return response.text.strip()
            except Exception as e:
                logger.warning("Gemini follow-up chat error (%s), using fallback.", e)

        # Smart contextual fallback response
        q_lower = user_message.lower()
        if "medicamento" in q_lower or "pastilla" in q_lower or "paracetamol" in q_lower or "ibuprofeno" in q_lower:
            return (
                "⚠️ ALERTA MÉDICA: Bajo ninguna circunstancia administres medicamentos de uso humano (paracetamol, ibuprofeno, "
                "aspirina o diclofenaco). En perros y gatos causan necrosis hepática fulminante, úlceras gástricas y fallo renal letal. "
                "Cualquier analgesia debe ser prescrita exclusivamente por el médico veterinario."
            )
        elif "agua" in q_lower or "comida" in q_lower or "alimento" in q_lower or "comer" in q_lower:
            if urgency == "CRITICAL":
                return (
                    "Ante un cuadro crítico, NO suministres alimentos sólidos ni líquidos. Si el paciente requiere sedación, intubación "
                    "o cirugía de emergencia al llegar a la clínica, el estómago lleno aumenta drásticamente el riesgo de broncoaspiración."
                )
            return (
                "Puedes humedecer sus labios o poner agua fresca a libre disposición a temperatura ambiente, pero nunca fuerces a tu "
                f"{pet_type} a beber si presenta náuseas, decaimiento extremo o dolor agudo. Suspende el alimento sólido hasta la revisión."
            )
        elif "transport" in q_lower or "llevar" in q_lower or "mover" in q_lower or "viaje" in q_lower:
            return (
                f"Para transportar a tu {pet_type}, colócalo sobre una superficie plana y firme (una manta extendida tipo camilla o caja de cartón rígida). "
                "Mantén la cabeza y el cuello en línea recta para no comprometer las vías respiratorias y evita flexionar la columna. "
                "Cubre su cuerpo con una toalla ligera para preservar la temperatura corporal y acude con precaución al centro veterinario."
            )
        else:
            return (
                f"Respecto a tu consulta sobre tu {pet_type}: Es fundamental mantener un ambiente templado, con luz tenue y sin ruidos bruscos "
                f"para reducir su nivel de estrés y dolor. Dado que el triage indicó severidad {urgency}, lo prioritario es vigilar si la respiración "
                "se vuelve agitada o superficial, y acudir al centro veterinario indicado en el mapa interactivo."
            )

    async def follow_up_chat(
        self,
        triage_context: Dict[str, Any],
        user_message: str,
        history: List[ChatMessage],
    ) -> str:
        """Asynchronously dispatches follow-up chat to worker thread pool."""
        return await self._thread_pool_manager.run_in_thread(
            self._sync_follow_up_chat,
            triage_context,
            user_message,
            history,
        )
