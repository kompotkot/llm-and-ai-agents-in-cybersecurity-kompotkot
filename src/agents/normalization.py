import json
import logging
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .. import config, data, prompt

logger = logging.getLogger(__name__)


def find_most_similar_embed(
    embed: np.ndarray,
    train_events: list[data.Event],
    top_k: int = 2,
) -> list[data.Event]:
    """
    Finds the top-k most similar reference examples
    using cosine similarity (dot product of normalized vectors).
    """
    candidates = [e for e in train_events if e.embed is not None]

    if not candidates:
        return []

    top_k = min(top_k, len(candidates))

    similarities = np.array([np.dot(embed, e.embed) for e in candidates])

    top_k_indices = np.argpartition(similarities, -top_k)[-top_k:]
    top_k_indices = top_k_indices[np.argsort(similarities[top_k_indices])[::-1]]

    return [candidates[i] for i in top_k_indices]


def verify_json_structure(raw_text: Union[str, dict]) -> Optional[dict]:
    """
    Verify that raw_text is valid JSON and conforms to expected schema.

    Args:
        raw_text: text pretend to be a JSON, or already a dict

    Returns:
        Parsed dict if valid, otherwise None.
    """
    # If it's already a dict, return it
    if isinstance(raw_text, dict):
        return raw_text

    # If it's a string, try to parse it as JSON
    if isinstance(raw_text, str):
        try:
            return json.loads(raw_text)
        except Exception:
            return None

    return None


# Why is it a tool? Because it is cool
@tool
def remove_markdown_quotes(raw_text: Union[str, dict]) -> str:
    """
    Remove Markdown quotes from JSON object.

    Args:
        raw_text: text pretend to be a JSON, or already a dict

    Returns:
        Cleaned raw JSON-like string.
    """
    logger.debug("Triggered tool: remove_markdown_quotes")

    # If it's already a dict, convert it to JSON string
    if isinstance(raw_text, dict):
        try:
            return json.dumps(raw_text, ensure_ascii=False)
        except Exception:
            return None

    raw_text = raw_text.strip()

    first_brace = raw_text.find("{")
    last_brace = raw_text.rfind("}")

    # opening_brace_not_found or closing_brace_not_found or closing_before_opening
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return None

    candidate = raw_text[first_brace : last_brace + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None

    return json.dumps(parsed, ensure_ascii=False)


class CleanupJob(BaseModel):
    norm_file: Path
    raw_text: str


class NormalizationAgentState(BaseModel):
    """
    Global workflow state for embedding, normalization and validation.

    This state tracks the training and prediction packs as well as metadata
    required to orchestrate the cleanup loop.
    """

    train_packs: List[data.EventPack] = Field(default_factory=list)
    pred_packs: List[data.EventPack] = Field(default_factory=list)

    # A list of jobs corresponding to malformed norm_fields files that need to
    # be cleaned. Each job encapsulates the target file and its raw contents.
    cleanup_jobs: list[CleanupJob] = Field(default_factory=list)
    cleanup_attempts: int = 0  # The number of cleanup cycles that have run so far
    cleanup_attempts_max: int = 2  # The maximum number of cleanup cycles to run


class NormalizationAgent:
    def __init__(
        self,
        slm_model: Optional[str] = None,
        slm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_url: Optional[str] = None,
        dump_embeddings: bool = False,
    ):
        self.embed_model = SentenceTransformer(config.EMBED_MODEL)

        self.slm = ChatOllama(
            model=slm_model or config.SLM_MODEL,
            base_url=slm_url or config.SLM_API_URI,
        )

        if config.LLM_API_KEY is not None:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                api_key=config.LLM_API_KEY,
                base_url=config.LLM_API_URI,
            )
        else:
            self.llm = ChatOllama(
                model=llm_model or config.LLM_MODEL,
                base_url=llm_url or config.LLM_API_URI,
            )

        self.tools = [remove_markdown_quotes]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        self.dump_embeddings = dump_embeddings

    def embed_train_events(
        self, state: NormalizationAgentState
    ) -> NormalizationAgentState:
        """
        Embed all training events so they can be used
        for similarity-based few-shot selection.
        """
        for pp in tqdm(state.train_packs, desc="Train packs embed"):
            for pe in tqdm(pp.events, desc="Train events embed", leave=False):
                vector = self.embed_model.encode(
                    pe.event_text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                pe.embed = np.array(vector, dtype=np.float32)

                if self.dump_embeddings:
                    event_file_path = pp.pack_path / pe.event_file_name
                    embed_file = event_file_path.with_name(
                        event_file_path.name.replace(
                            "events_", "embed_train_events_"
                        ).replace(".json", ".npy")
                    )
                    np.save(embed_file, pe.embed)

        return state

    def embed_pred_events(
        self, state: NormalizationAgentState
    ) -> NormalizationAgentState:
        """
        Embed all prediction events so we can retrieve similar training examples per event.
        """
        for pp in tqdm(state.pred_packs, desc="Prediction packs embed"):
            for pe in tqdm(pp.events, desc="Prediction events embed", leave=False):
                vector = self.embed_model.encode(
                    pe.event_text,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                pe.embed = np.array(vector, dtype=np.float32)

                if self.dump_embeddings:
                    event_file_path = pp.pack_path / pe.event_file_name
                    embed_file = event_file_path.with_name(
                        event_file_path.name.replace(
                            "events_", "embed_pred_events_"
                        ).replace(".json", ".npy")
                    )
                    logger.debug(
                        f"Embed for pred event dumped at: {embed_file.relative_to(config.BASE_DIR)}"
                    )
                    np.save(embed_file, pe.embed)

        return state

    def prepare_pred_prompts(
        self, state: NormalizationAgentState
    ) -> NormalizationAgentState:
        """
        Build per-event normalization prompts for prediction data using nearest training examples.
        Prompts are stored on each prediction event for later LLM invocation.
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
                )

        return state

    def normalize_pred_events(
        self, state: NormalizationAgentState
    ) -> NormalizationAgentState:
        """
        Run the main LLM to generate `norm_fields_*.json` files from prepared prompts.
        """
        system_prompt = prompt.load_system_prompt()
        tax_fields_prompt = prompt.load_taxonomy_fields_prompt()
        system_prompt_content = system_prompt + "\n" + tax_fields_prompt
        system_prompt = SystemMessage(content=system_prompt_content)

        for pp in tqdm(state.pred_packs, desc="Packs under normalization"):
            for pe in tqdm(pp.events, desc="Events under normalization", leave=False):
                messages = [
                    system_prompt,
                    pe.prompt,
                ]
                response = self.slm.invoke(messages)

                event_file_path = pp.pack_path / pe.event_file_name

                # Save normalized fields to file in the same directory as event file
                norm_file = event_file_path.with_name(
                    event_file_path.name.replace("events_", "norm_fields_")
                )
                norm_file.write_text(response.content.strip(), encoding="utf-8")
                logger.debug(
                    f"Normalized: {norm_file.parent.relative_to(config.BASE_DIR)}/{norm_file.name}"
                )

        return state

    def scan_norm_fields(
        self, state: NormalizationAgentState
    ) -> NormalizationAgentState:
        """
        Validate generated norm_fields outputs.
        """
        for pp in state.pred_packs:
            for pe in pp.events:
                norm_file = Path(
                    pp.pack_path / pe.event_file_name.replace("events_", "norm_fields_")
                )
                raw_text = norm_file.read_text(encoding="utf-8")

                verified = verify_json_structure(raw_text)
                if verified is not None:
                    pe.norm_text_clean = True
                else:
                    pe.norm_text_clean = False

                logger.debug(
                    f"Scanned {norm_file.parent.relative_to(config.BASE_DIR)}/{norm_file.name} file as {pe.norm_text_clean}"
                )

        return state

    def prepare_cleanup_jobs(self, state: NormalizationAgentState) -> dict:
        """
        Collect invalid norm_fields files and prepare cleanup jobs.
        """
        jobs = []

        for pp in state.pred_packs:
            for pe in pp.events:
                if not pe.norm_text_clean:
                    norm_file = Path(
                        pp.pack_path
                        / pe.event_file_name.replace("events_", "norm_fields_")
                    )
                    raw_text = norm_file.read_text(encoding="utf-8")

                    jobs.append(
                        CleanupJob(
                            norm_file=norm_file,
                            raw_text=raw_text,
                        )
                    )

        return {"cleanup_jobs": jobs}

    def cleanup_llm_node(self, state: MessagesState) -> dict:
        """
        Invoke tool-enabled LLM for cleanup.
        """
        response = self.llm_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}

    def cleanup_invalid_norm_files(
        self, state: NormalizationAgentState
    ) -> NormalizationAgentState:
        """
        Sequentially process all cleanup jobs to repair malformed JSON outputs.

        This node iterates over the list of `cleanup_jobs` collected during
        scanning. For each job it prepares a new prompt instructing the LLM
        to clean the raw text, invokes the cleanup subgraph synchronously,
        writes the cleaned text back to disk and updates the corresponding
        normalized event's `norm_text_clean` key.
        """
        if not state.cleanup_jobs:
            return state

        system_prompt = prompt.load_system_clean_prompt()

        for job in tqdm(state.cleanup_jobs, desc="Cleanup jobs"):
            # Build a prompt instructing the LLM to clean the raw output
            clean_prompt = prompt.load_clean_prompt(
                job.raw_text,
                job.norm_file,
            )
            messages: list[BaseMessage] = [
                system_prompt,
                HumanMessage(content=clean_prompt),
            ]

            try:
                result = self.clean_agent.invoke({"messages": messages})
            except Exception as e:
                logger.debug(f"Cleanup subgraph failed for {job.norm_file}")
                logger.error(f"Cleanup subgraph error: {e}")
                continue

            # The cleaned text should be the content of the last message
            cleaned_messages = result.get("messages", [])
            if not cleaned_messages:
                logger.debug(f"No messages returned from cleanup for {job.norm_file}")
                continue

            cleaned_text = cleaned_messages[-1].content

            verified = verify_json_structure(cleaned_text)
            if verified is None:
                logger.warning(
                    f"The cleanup LLM failed to perform its duties for {job.norm_file}"
                )
                continue

            job.norm_file.write_text(cleaned_text, encoding="utf-8")

            # Mark the corresponding prediction event as clean
            for pp in state.pred_packs:
                for pe in pp.events:
                    expected_norm_file = pp.pack_path / pe.event_file_name.replace(
                        "events_", "norm_fields_"
                    )
                    if expected_norm_file == job.norm_file:
                        pe.norm_text_clean = True
                        break

        # Clear the job list and increment the attempts counter. We return
        # updates here rather than mutating the state directly, following
        # LangGraph best practices.
        return {
            "cleanup_jobs": [],
            "cleanup_attempts": state.cleanup_attempts + 1,
        }

    def build_cleanup_subgraph(self):
        """
        Assemble and compile cleanup subgraph.
        """

        builder = StateGraph(MessagesState)

        builder.add_node("llm_with_tools", self.cleanup_llm_node)
        builder.add_node("tools", ToolNode(self.tools))

        builder.add_edge(START, "llm_with_tools")
        builder.add_conditional_edges(
            "llm_with_tools",
            tools_condition,
            {
                "tools": "tools",
                "__end__": END,
            },
        )
        builder.add_edge("tools", "llm_with_tools")

        return builder.compile()

    def build_graph(self):
        """
        Assemble and compile the full workflow graph,
        including the norm_fields clean subgraph.
        """
        self.clean_agent = self.build_cleanup_subgraph()

        builder = StateGraph(NormalizationAgentState)

        # Main pipeline nodes
        builder.add_node("embed_train_events", self.embed_train_events)
        builder.add_node("embed_pred_events", self.embed_pred_events)
        builder.add_node("prepare_pred_prompts", self.prepare_pred_prompts)
        builder.add_node("normalize_pred_events", self.normalize_pred_events)
        builder.add_node("scan_norm_fields", self.scan_norm_fields)

        # Cleanup pipeline nodes
        builder.add_node("prepare_cleanup_jobs", self.prepare_cleanup_jobs)
        builder.add_node("cleanup_invalid_norm_files", self.cleanup_invalid_norm_files)

        # The flow: initial sequence up to scanning the normalization outputs
        builder.add_edge(START, "embed_train_events")
        builder.add_edge("embed_train_events", "embed_pred_events")
        builder.add_edge("embed_pred_events", "prepare_pred_prompts")
        builder.add_edge("prepare_pred_prompts", "normalize_pred_events")
        builder.add_edge("normalize_pred_events", "scan_norm_fields")

        # After scanning, decide whether to perform a cleanup cycle or end
        builder.add_conditional_edges(
            "scan_norm_fields",
            lambda s: (
                "cleanup"
                if any(
                    not pe.norm_text_clean for pp in s.pred_packs for pe in pp.events
                )
                and s.cleanup_attempts < s.cleanup_attempts_max
                else "__end__"
            ),
            {
                "cleanup": "prepare_cleanup_jobs",
                "__end__": END,
            },
        )

        # When cleanup is triggered, collect the jobs and then clean them. Once
        # cleaning is finished, loop back to scan the files again.
        builder.add_edge("prepare_cleanup_jobs", "cleanup_invalid_norm_files")
        builder.add_edge("cleanup_invalid_norm_files", "scan_norm_fields")

        return builder.compile()
