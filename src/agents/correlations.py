import json
import logging
from pathlib import Path
from typing import List, Optional, Set

import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from tqdm import tqdm

from .. import config, data, prompt

logger = logging.getLogger(__name__)


class CorrelationAgentState(BaseModel):
    """
    Global workflow state for embedding and correlation.

    This state tracks the training and prediction packs.
    """

    train_mitre_patterns: List[data.MitrePattern] = Field(default_factory=list)
    norm_field_packs: List[data.EventPack] = Field(default_factory=list)

    filtered_fields: Set[str] = Field(default_factory=set)


class CorrelationAgent:
    def __init__(
        self,
        embeddings_model: Optional[str] = None,
        embeddings_url: Optional[str] = None,
        slm_model: Optional[str] = None,
        slm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None,
        filtered_fields_path: Optional[str] = None,
        embeddings_path: Optional[str] = None,
        dump_embeddings: bool = False,
    ):
        self.elm = OllamaEmbeddings(
            model=embeddings_model or config.EMBED_MODEL,
            base_url=embeddings_url or config.EMBED_API_URI,
        )

        self.slm = ChatOllama(
            model=slm_model or config.SLM_MODEL,
            base_url=slm_url or config.SLM_API_URI,
        )

        self.llm = ChatOllama(
            model=llm_model or config.LLM_MODEL,
            base_url=llm_url or config.LLM_API_URI,
        )

        self.filtered_fields_path: Optional[str] = None
        if filtered_fields_path is not None:
            self.filtered_fields_path = filtered_fields_path
        else:
            """
            TODO(kompotkot): Add llm field extraction if not provided
            """
            raise Exception("Not in pipeline")

        self.embeddings_path: Optional[str] = None
        if embeddings_path is not None:
            self.embeddings_path = embeddings_path

        self.dump_embeddings = dump_embeddings

    def load_embed_train_mitre_patterns(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        embeddings_path = Path(self.embeddings_path)
        cnt = 0

        for mp in state.train_mitre_patterns:
            file_path = embeddings_path / f"{mp.external_id.replace('.', '_')}.npy"
            if not file_path.exists():
                logging.warning(f"There is not embedding for {mp.external_id}")
                continue

            mp.embed = np.load(file_path)
            cnt += 1

        if cnt == 0:
            raise Exception("Failed to load any embeddings for train mitre patterns")

        logging.info(f"Loaded {cnt} embeddings")

        return state

    def embed_train_mitre_patterns(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        """
        Embed all training Mitre patterns.
        """
        for mp in tqdm(state.train_mitre_patterns, desc="Train mitre patterns embed"):
            vector = self.elm.embed_query(mp.text)
            mp.embed = np.array(vector, dtype=np.float32)

            if self.dump_embeddings:
                dir_path = config.MITRE_COMPACT_PATTERNS_PATH.parent
                embed_file = dir_path / mp.external_id.replace(".", "_")
                logger.debug(
                    f"Embed for tain mitre patterns dumped at: {embed_file.relative_to(config.BASE_DIR)}"
                )
                np.save(embed_file, mp.embed)

        return state

    def load_filtered_fields(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        """
        TODO(kompotkot): Add llm field extraction if not provided
        """
        ff_path = Path(self.filtered_fields_path)
        ff_data = ff_path.read_text(encoding="utf-8")
        # Parse comma-separated string into a set
        state.filtered_fields = set(
            field.strip() for field in ff_data.split(",") if field.strip()
        )

        return state

    def filter_norm_fields(self, state: CorrelationAgentState) -> CorrelationAgentState:
        for pp in tqdm(state.norm_field_packs, desc="Packs filtering"):
            for pe in tqdm(pp.events, desc="Norm fields filtering", leave=False):
                # Filter keys from pe.norm_fields by state.filtered_fields
                filtered_items = []
                for key, value in pe.norm_fields.items():
                    if key in state.filtered_fields:
                        # Concatenate them in string like key: value separated by comma
                        filtered_items.append(f"{key}: {value}")

                event_file_path = pp.pack_path / pe.event_file_name

                if len(filtered_items) == 0:
                    logger.warning(
                        f"Not found any important fields in {str(event_file_path)}"
                    )
                    continue

                pe.filtered_norm_text = (
                    ", ".join(filtered_items) if filtered_items else ""
                )

                if logger.isEnabledFor(logging.DEBUG):
                    ff_norm_file = event_file_path.with_name(
                        event_file_path.name.replace(
                            "norm_fields_", "ff_norm_fields_"
                        ).replace(".json", ".txt")
                    )
                    ff_norm_file.write_text(pe.filtered_norm_text, encoding="utf-8")

        return state

    def build_graph(self):
        """
        Assemble and compile the full workflow graph.
        """
        builder = StateGraph(CorrelationAgentState)

        # Main pipeline nodes
        if self.embeddings_path is None:
            builder.add_node(
                "embed_train_mitre_patterns", self.embed_train_mitre_patterns
            )
        else:
            builder.add_node(
                "embed_train_mitre_patterns", self.load_embed_train_mitre_patterns
            )
        builder.add_node("filtered_fields", self.load_filtered_fields)
        builder.add_node("filter_norm_fields", self.filter_norm_fields)

        # The flow
        builder.add_edge(START, "embed_train_mitre_patterns")
        builder.add_edge("embed_train_mitre_patterns", "filtered_fields")
        builder.add_edge("filtered_fields", "filter_norm_fields")
        builder.add_edge("embed_train_mitre_patterns", END)

        return builder.compile()
