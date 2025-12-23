from typing import List, Optional

import numpy as np
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from . import config, data, prompt


def find_most_similar_embed(
    embed: np.ndarray, train_events: list[data.Event], top_k: int = 2
) -> list[data.Event]:
    """
    Finds the top-k most similar reference examples by comparing cosine similarity of embeddings.
    """
    if len(train_events) < top_k:
        top_k = len(train_events)

    # Calculate cosine similarity with all reference embeddings
    similarities = []
    for e in train_events:
        if e.embed is None:
            continue
        similarity = np.dot(embed, e.embed)
        similarities.append(similarity)

    # Find indices of top-k most similar references
    similarities_array = np.array(similarities)
    top_k_indices = np.argpartition(similarities_array, -top_k)[-top_k:]

    # Sort by similarity in descending order
    top_k_indices = top_k_indices[np.argsort(similarities_array[top_k_indices])[::-1]]

    return [train_events[idx] for idx in top_k_indices]


class EmbeddingState(BaseModel):
    train_packs: List[data.EventPack] = Field(default_factory=list)
    pred_packs: List[data.EventPack] = Field(default_factory=list)


class Orchestration:
    def __init__(
        self,
        embeddings_model: Optional[str] = None,
        embeddings_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None,
        debug: bool = False,
    ):
        if embeddings_model is None:
            embeddings_model = config.EMBED_MODEL

        if embeddings_url is None:
            embeddings_url = config.EMBED_API_URI

        self.embeddings = OllamaEmbeddings(
            model=embeddings_model,
            base_url=embeddings_url,
        )

        if llm_model is None:
            llm_model = config.LLM_MODEL

        if llm_url is None:
            llm_url = config.LLM_API_URI

        self.llm = ChatOllama(
            model=llm_model,
            base_url=llm_url,
        )

        self.debug = debug

    def embed_events(self, packs: list[data.EventPack]) -> None:
        for e in packs.events:
            event_text = e.event_text
            vector = self.embeddings.embed_query(event_text)
            e.embed = np.array(vector, dtype=np.float32)

    def embed_train_events(self, state: EmbeddingState) -> EmbeddingState:
        for p in state.train_packs:
            self.embed_events(p)
        return state

    def embed_pred_events(self, state: EmbeddingState) -> EmbeddingState:
        for p in state.pred_packs:
            self.embed_events(p)
        return state

    def prepare_pred_prompts(self, state: EmbeddingState) -> EmbeddingState:
        """
        Generate event prompts for prediction data.
        """
        train_events = [e for pack in state.train_packs for e in pack.events]

        for pp in state.pred_packs:
            for pe in pp.events:
                # Find most similar reference examples by comparing embeddings
                most_similar_trains = find_most_similar_embed(pe.embed, train_events)

                # Prepare examples list with (event, normalized) pairs
                examples = [(r.event_text, r.norm_text) for r in most_similar_trains]

                # Generate prompt with event and 3 similar examples
                pe.prompt = prompt.render_prompt_correlation(
                    event=pe.event_text,
                    examples=examples,
                    event_file_path=pp.pack_path / pe.event_file_name,
                    debug=self.debug,
                )
        return state

    def normalize_pred_events(self, state: EmbeddingState) -> EmbeddingState:
        # Prepare system prompt
        system_prompt = prompt.load_system_prompt(self.debug)

        for pp in state.pred_packs:
            for pe in pp.events:
                messages = [
                    system_prompt,
                    pe.prompt,
                ]
                response = self.llm.invoke(messages)

                event_file_path = pp.pack_path / pe.event_file_name

                # Save normalized fields to file in the same directory as event file
                norm_file = event_file_path.with_name(
                    event_file_path.name.replace("events_", "norm_fields_")
                )
                norm_file.write_text(response.content.strip(), encoding="utf-8")
                if self.debug:
                    print(
                        f"Normalized: {norm_file.parent.relative_to(config.BASE_DIR)}/{norm_file.name}"
                    )
        return state

    def build_graph(self):
        # Builds LangGraph
        builder = StateGraph(EmbeddingState)
        builder.add_node("embed_train_events", self.embed_train_events)
        builder.add_node("embed_pred_events", self.embed_pred_events)
        builder.add_node("prepare_pred_prompts", self.prepare_pred_prompts)
        builder.add_node("normalize_pred_events", self.normalize_pred_events)

        builder.add_edge(START, "embed_train_events")
        builder.add_edge("embed_train_events", "embed_pred_events")
        builder.add_edge("embed_pred_events", "prepare_pred_prompts")
        builder.add_edge("prepare_pred_prompts", "normalize_pred_events")

        builder.add_edge("normalize_pred_events", END)

        return builder.compile()
