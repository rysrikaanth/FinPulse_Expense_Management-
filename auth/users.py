"""
User registry and authentication.

Users are stored in auth/user_db.json.
Passwords are SHA-256 hashed.

Roles:
  admin  — full access to all dashboard pages + all users' data
  user   — can only see their own claims and submit new ones
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

_AUTH_DIR = Path(__file__).resolve().parent
_USER_DB  = _AUTH_DIR / "user_db.json"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _load() -> dict:
    if _USER_DB.exists():
        with open(_USER_DB, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(db: dict) -> None:
    _AUTH_DIR.mkdir(parents=True, exist_ok=True)
    with open(_USER_DB, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def authenticate(username: str, password: str) -> Optional[dict]:
    """Return user dict if credentials are valid, else None."""
    db = _load()
    user = db.get(username)
    if user and user["password_hash"] == _hash(password):
        return {
            "username":       username,
            "role":           user["role"],
            "display_name":   user.get("display_name", username),
            "user_id":        user.get("user_id", username),
            "position":       user.get("position", ""),
            "department":     user.get("department", ""),
            "monthly_limits": user.get("monthly_limits", {}),
        }
    return None


def get_all_users() -> list[dict]:
    """Return list of all user records (without password hashes)."""
    db = _load()
    return [
        {
            "username":       u,
            "role":           v["role"],
            "display_name":   v.get("display_name", u),
            "user_id":        v.get("user_id", u),
            "position":       v.get("position", ""),
            "department":     v.get("department", ""),
            "monthly_limits": v.get("monthly_limits", {}),
        }
        for u, v in db.items()
    ]


def get_regular_users() -> list[dict]:
    return [u for u in get_all_users() if u["role"] == "user"]


def register_user(
    username: str,
    password: str,
    role: str = "user",
    display_name: str = "",
    user_id: str = "",
    position: str = "",
    department: str = "",
    monthly_limits: dict = None,
) -> None:
    """Register or update a user in the DB."""
    db = _load()
    db[username] = {
        "password_hash":  _hash(password),
        "role":           role,
        "display_name":   display_name or username,
        "user_id":        user_id or username,
        "position":       position,
        "department":     department,
        "monthly_limits": monthly_limits or {},
    }
    _save(db)


# ── Bootstrap default users ───────────────────────────────────────────────────

# Monthly expense limits per category per employee (INR)
# Based on realistic Indian corporate HR policy
EMPLOYEE_PROFILES = {
    "admin": {
        "password":       "admin@123",
        "role":           "admin",
        "display_name":   "Vikram Nair (Admin)",
        "user_id":        "admin",
        "position":       "Finance Controller",
        "department":     "Finance",
        "monthly_limits": {},   # admin has no personal expense limits
    },
    "rahul_m": {
        "password":       "rahul@2024",
        "role":           "user",
        "display_name":   "Rahul Mehta",
        "user_id":        "user_rahul",
        "position":       "Senior Software Engineer",
        "department":     "Engineering",
        "monthly_limits": {
            "meals":         8000,
            "transport":    10000,
            "accommodation":30000,
            "equipment":    15000,
            "training":      5000,
            "internet":      2000,
        },
    },
    "priya_s": {
        "password":       "priya@2024",
        "role":           "user",
        "display_name":   "Priya Sharma",
        "user_id":        "user_priya",
        "position":       "Sales Manager",
        "department":     "Sales & Business Development",
        "monthly_limits": {
            "meals":         20000,
            "transport":     40000,
            "accommodation": 60000,
            "client_gifts":   5000,
            "conference":    10000,
        },
    },
    "arjun_k": {
        "password":       "arjun@2024",
        "role":           "user",
        "display_name":   "Arjun Kumar",
        "user_id":        "user_arjun",
        "position":       "Junior Software Developer",
        "department":     "Engineering",
        "monthly_limits": {
            "meals":        4000,
            "transport":    3000,
            "internet":     1000,
            "training":     3000,
        },
    },
    "sneha_r": {
        "password":       "sneha@2024",
        "role":           "user",
        "display_name":   "Sneha Rao",
        "user_id":        "user_sneha",
        "position":       "Product Manager",
        "department":     "Product",
        "monthly_limits": {
            "meals":         12000,
            "transport":     15000,
            "accommodation": 25000,
            "software":      10000,
            "conference":    15000,
            "training":       5000,
        },
    },
}


def bootstrap(force: bool = False) -> None:
    """Seed the DB with admin + 4 employees. force=True overwrites existing."""
    db = _load()
    changed = False

    for username, prof in EMPLOYEE_PROFILES.items():
        if username not in db or force:
            db[username] = {
                "password_hash":  _hash(prof["password"]),
                "role":           prof["role"],
                "display_name":   prof["display_name"],
                "user_id":        prof["user_id"],
                "position":       prof.get("position", ""),
                "department":     prof.get("department", ""),
                "monthly_limits": prof.get("monthly_limits", {}),
            }
            changed = True

    if changed:
        _save(db)

