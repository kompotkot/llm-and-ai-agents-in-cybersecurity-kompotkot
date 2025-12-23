from pathlib import Path
from typing import List, Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Event(BaseModel):
    """
    Data model for storing events with original text,
    normalized text and embedding.
    """

    event_file_name: str

    event_text: str  # Original unnormalized event text
    norm_text: Optional[str] = None  # Normalized SIEM event text

    embed: Optional[np.ndarray] = None  # Vector embedding of event/norm text
    prompt: Optional[str] = None  # Prompt to generate normalized SIEM event

    mitre: Optional[dict] = None  # MITRE ATT&CK classification result

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EventPack(BaseModel):
    """
    Data model for storing packs of events.
    """

    pack_path: Path

    events: List[Event] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)
