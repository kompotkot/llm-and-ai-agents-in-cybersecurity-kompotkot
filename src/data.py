from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict


class ReferenceItem(BaseModel):
    """
    Data model for storing reference examples with original events,
    normalized events, and embeddings.
    """

    event_text: str  # Original unnormalized event text
    norm_text: Optional[str] = None  # Normalized SIEM event text
    embed: Optional[np.ndarray] = None  # Vector embedding of the event text

    model_config = ConfigDict(arbitrary_types_allowed=True)
