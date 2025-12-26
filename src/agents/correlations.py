import json
import logging
from pathlib import Path
from typing import List, Optional, Union

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


class CorrelationAgent:
    def __init__(
        self,
        embeddings_model: Optional[str] = None,
        embeddings_url: Optional[str] = None,
        slm_model: Optional[str] = None,
        slm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None,
        embeddings: Optional[str] = None,
        dump_embeddings: bool = False,
    ):
        self.embeddings = OllamaEmbeddings(
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

        self.embeddings: Optional[str] = None
        if embeddings is not None:
            self.embeddings = embeddings

        self.dump_embeddings = dump_embeddings

    def load_embed_train_mitre_patterns(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        embeddings_path = Path(self.embeddings)
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
        for mp in tqdm(state.train_mitre_patterns, desc="Train packs embed"):
            vector = self.embeddings.embed_query(mp.text)
            mp.embed = np.array(vector, dtype=np.float32)

            if self.dump_embeddings:
                dir_path = config.MITRE_COMPACT_PATTERNS_PATH.parent
                embed_file = dir_path / mp.external_id.replace(".", "_")
                logger.debug(
                    f"Embed for tain mitre patterns dumped at: {embed_file.relative_to(config.BASE_DIR)}"
                )
                np.save(embed_file, mp.embed)

        return state

    def build_graph(self):
        """
        Assemble and compile the full workflow graph.
        """
        builder = StateGraph(CorrelationAgentState)

        # Main pipeline nodes
        if self.embeddings is None:
            builder.add_node(
                "embed_train_mitre_patterns", self.embed_train_mitre_patterns
            )
        else:
            builder.add_node(
                "embed_train_mitre_patterns", self.load_embed_train_mitre_patterns
            )

        # The flow
        builder.add_edge(START, "embed_train_mitre_patterns")
        builder.add_edge("embed_train_mitre_patterns", END)

        return builder.compile()
