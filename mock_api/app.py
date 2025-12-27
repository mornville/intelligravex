from __future__ import annotations

from datetime import date
from typing import List, Literal, Optional

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field


app = FastAPI(title="Intelligravex Mock APIs", version="0.1.0")


class UserData(BaseModel):
    firstName: str
    lastName: str
    ssn: str = Field(description="Mock SSN (do not use real SSNs).")
    dob: str = Field(description="Date of birth (YYYY-MM-DD).")


class Doctor(BaseModel):
    id: str
    name: str
    specialty: str
    city: str
    state: str
    phone: str
    accepts_new_patients: bool = True
    rating: float = 4.6


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/v1/user")
def get_user(
    user_id: str = Query("u_123", description="Mock user id"),
) -> dict:
    # Deterministic fake user (based on id hash) so it's stable across calls.
    # Keep this simple; the goal is to have predictable fields for integrations.
    seed = sum(ord(c) for c in user_id) % 1000
    first = ["Aarav", "Aisha", "Riya", "Kabir", "Neha", "Arjun"][seed % 6]
    last = ["Sharma", "Khan", "Gupta", "Verma", "Singh", "Mehta"][seed % 6]
    yyyy = 1990 + (seed % 20)
    mm = 1 + (seed % 12)
    dd = 1 + (seed % 28)
    dob = date(yyyy, mm, dd).isoformat()
    ssn = f"{100 + (seed % 900)}-{10 + (seed % 90):02d}-{1000 + (seed % 9000)}"
    return {"data": UserData(firstName=first, lastName=last, ssn=ssn, dob=dob).model_dump()}


DOCTORS: List[Doctor] = [
    Doctor(
        id="d_001",
        name="Dr. Maya Patel",
        specialty="Orthopedics",
        city="Mexico City",
        state="CDMX",
        phone="+52 55 5555 0101",
        accepts_new_patients=True,
        rating=4.7,
    ),
    Doctor(
        id="d_002",
        name="Dr. Luis Hernández",
        specialty="Orthopedics",
        city="Guadalajara",
        state="JAL",
        phone="+52 33 5555 0202",
        accepts_new_patients=True,
        rating=4.5,
    ),
    Doctor(
        id="d_003",
        name="Dr. Sofia Martínez",
        specialty="Cardiology",
        city="Mexico City",
        state="CDMX",
        phone="+52 55 5555 0303",
        accepts_new_patients=False,
        rating=4.8,
    ),
    Doctor(
        id="d_004",
        name="Dr. Rahul Jain",
        specialty="Orthopedics",
        city="Pune",
        state="MH",
        phone="+91 20 5555 0404",
        accepts_new_patients=True,
        rating=4.6,
    ),
]


@app.get("/v1/doctors")
def list_doctors(
    city: Optional[str] = Query(None, description="Filter by city"),
    specialty: Optional[str] = Query(None, description="Filter by specialty"),
    accepts_new_patients: Optional[bool] = Query(None, description="Filter by acceptance"),
    limit: int = Query(20, ge=1, le=200),
    sort: Literal["rating_desc", "rating_asc", "name_asc"] = Query("rating_desc"),
) -> dict:
    items = DOCTORS[:]

    if city:
        c = city.strip().lower()
        items = [d for d in items if d.city.lower() == c]
    if specialty:
        s = specialty.strip().lower()
        items = [d for d in items if d.specialty.lower() == s]
    if accepts_new_patients is not None:
        items = [d for d in items if d.accepts_new_patients == accepts_new_patients]

    if sort == "rating_desc":
        items.sort(key=lambda d: d.rating, reverse=True)
    elif sort == "rating_asc":
        items.sort(key=lambda d: d.rating)
    elif sort == "name_asc":
        items.sort(key=lambda d: d.name)

    items = items[:limit]
    return {"data": [d.model_dump() for d in items]}

