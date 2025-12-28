from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Top5Tech(BaseModel):
    external_id: str
    score: float


class Event(BaseModel):
    """
    Data model for storing events with original text,
    normalized text and embedding.
    """

    event_file_name: str

    event_text: Optional[str] = None  # Original unnormalized event text
    norm_text: Optional[str] = None  # Normalized SIEM event text
    norm_text_clean: bool = False

    embed: Optional[np.ndarray] = None  # Vector embedding of event/norm text
    prompt: Optional[str] = None  # Prompt to generate normalized SIEM event

    # Mitre
    norm_fields: Dict[str, Any] = Field(default_factory=dict)
    filtered_norm_text: Optional[str] = None  # Filtered text by important fields
    filtered_norm_embed: Optional[np.ndarray] = (
        None  # Vector embedding of filtered text
    )

    top_5_techs: List[Top5Tech] = Field(default_factory=list)
    tech_id: Optional[str] = None
    sub_tech_id: Optional[str] = None
    tech_score: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EventPack(BaseModel):
    """
    Data model for storing packs of events.
    """

    pack_path: Path

    events: List[Event] = Field(default_factory=list)

    pack_tech_id: Optional[str] = None
    pack_sub_tech_id: Optional[str] = None
    pack_tech_score: Optional[float] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


class MitrePattern(BaseModel):
    external_id: str
    name: str
    phases: List[str] = Field(default_factory=list)
    description: str

    text: str  # Concatenated fields in line
    embed: Optional[np.ndarray] = None  # Vector embedding of text

    model_config = ConfigDict(arbitrary_types_allowed=True)
