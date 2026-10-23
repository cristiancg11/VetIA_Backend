"""Geospatial clinic localization and triage routing service."""

import math
from typing import Dict, List, Optional
from app.schemas.triage import ClinicLocationResponse, TriageUrgencyLevel


class ClinicLocatorService:
    """
    Object-oriented service responsible for locating nearby veterinary clinics,
    calculating geographical distances via Haversine formula, and prioritizing
    emergency facilities based on patient urgency level.
    """

    EARTH_RADIUS_KM: float = 6371.0

    def __init__(self) -> None:
        self._clinic_templates: List[Dict[str, object]] = [
            {
                "id": "vet-emergency-01",
                "name": "Hospital Veterinario de Urgencias 24H San Juan",
                "phone": "+57 300 123 4567",
                "hours": "Abierto 24 Horas / UCI & Quirófano",
                "d_lat": 0.0125,
                "d_lng": -0.0092,
                "address": "Av. Principal #45-12, Sector Norte",
                "is_emergency_facility": True,
            },
            {
                "id": "vet-emergency-02",
                "name": "Centro Quirúrgico y Traumatología Animal 'La Cruz'",
                "phone": "+57 311 987 6543",
                "hours": "Abierto 24 Horas / Emergencias Críticas",
                "d_lat": -0.0150,
                "d_lng": 0.0110,
                "address": "Calle 68 #22-80, Zona Médica",
                "is_emergency_facility": True,
            },
            {
                "id": "vet-general-03",
                "name": "Clínica Veterinaria & Diagnóstico Integral PetCare",
                "phone": "+57 315 456 7890",
                "hours": "Lun - Sáb: 7:00 AM - 8:00 PM (Dom: 9:00 AM - 5:00 PM)",
                "d_lat": 0.0078,
                "d_lng": 0.0135,
                "address": "Carrera 15 #85-30, Barrio El Prado",
                "is_emergency_facility": False,
            },
            {
                "id": "vet-general-04",
                "name": "Consultorio Veterinario General Huellitas Felices",
                "phone": "+57 320 654 3210",
                "hours": "Lun - Sáb: 8:00 AM - 6:00 PM",
                "d_lat": -0.0095,
                "d_lng": -0.0142,
                "address": "Diagonal 40 #19-45, Centro Residencial",
                "is_emergency_facility": False,
            },
        ]

    def calculate_haversine_distance(
        self,
        lat1: float,
        lon1: float,
        lat2: float,
        lon2: float,
    ) -> float:
        """
        Calculate great-circle distance between two geographical points using Haversine formula.
        """
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = (
            math.sin(delta_phi / 2.0) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
        )
        c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
        return round(self.EARTH_RADIUS_KM * c, 2)

    def find_nearby_clinics(
        self,
        latitude: float,
        longitude: float,
        urgency: Optional[TriageUrgencyLevel] = TriageUrgencyLevel.MODERATE,
    ) -> List[ClinicLocationResponse]:
        """
        Locates veterinary centers around the user coordinates and orders them
        by urgency classification and physical distance.
        """
        clinics: List[ClinicLocationResponse] = []

        for template in self._clinic_templates:
            clinic_lat = round(latitude + float(template["d_lat"]), 6)  # type: ignore
            clinic_lng = round(longitude + float(template["d_lng"]), 6)  # type: ignore
            distance = self.calculate_haversine_distance(
                latitude, longitude, clinic_lat, clinic_lng
            )

            clinics.append(
                ClinicLocationResponse(
                    id=str(template["id"]),
                    name=str(template["name"]),
                    phone=str(template["phone"]),
                    hours=str(template["hours"]),
                    distance_km=distance,
                    coordinates=[clinic_lat, clinic_lng],
                    address=str(template["address"]),
                    is_emergency_facility=bool(template["is_emergency_facility"]),
                )
            )

        # Critical urgency strictly prioritizes 24/7 emergency facilities
        if urgency == TriageUrgencyLevel.CRITICAL:
            clinics.sort(
                key=lambda clinic: (not clinic.is_emergency_facility, clinic.distance_km)
            )
        else:
            clinics.sort(key=lambda clinic: clinic.distance_km)

        return clinics
