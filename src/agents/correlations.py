import json
import logging
from collections import defaultdict
from itertools import islice
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .. import config, data, prompt

logger = logging.getLogger(__name__)


def normalize_tactics(s: str) -> str:
    return ", ".join(part.strip().replace("-", " ").title() for part in s.split(","))


def build_dd_index(pats: List[data.MitrePattern]) -> NearestNeighbors:
    """
    Build retrieval index.
    """
    X = np.vstack([p.embed for p in pats])

    nn = NearestNeighbors(n_neighbors=10, metric="cosine")
    nn.fit(X)

    return nn


def retrieve_top_k(
    nn: NearestNeighbors,
    pats: List[data.MitrePattern],
    filtered_norm_embed: np.ndarray,
    k: int = 5,
) -> List[data.Top5Tech]:
    """
    Find Top-k technique for one normal field event.
    """
    distances, indices = nn.kneighbors(
        filtered_norm_embed.reshape(1, -1), n_neighbors=k
    )

    top_5_techs: List[data.Top5Tech] = []
    for dist, i in zip(distances[0], indices[0]):
        top_5_techs.append(
            data.Top5Tech(
                external_id=pats[i].external_id,
                score=float(1.0 - dist),  # cosine similarity
            )
        )

    return top_5_techs


def select_technique(
    event: data.Event, delta: float = 0.02
) -> Tuple[str, Optional[str], float]:
    """
    Select MITRE technique.
    """

    if len(event.top_5_techs) == 0:
        raise ValueError("Event has no top_5_techs")

    # Group by family
    families: dict[str, list[data.Top5Tech]] = defaultdict(list)
    for tech in event.top_5_techs:
        # T1055.012 -> T1055
        fam = tech.external_id.split(".")[0]
        families[fam].append(tech)

    # Select best family by max score
    best_family = None
    best_family_score = -1.0
    best_family_techs: list[data.Top5Tech] = []

    for fam, techs in families.items():
        fam_score = max(t.score for t in techs)

        if fam_score > best_family_score:
            best_family = fam
            best_family_score = fam_score
            best_family_techs = techs

    # Sort subtechniques inside the selected family
    sorted_subs = sorted(best_family_techs, key=lambda t: t.score, reverse=True)

    # Decide whether to emit subtechnique
    if len(sorted_subs) >= 2:
        score_diff = sorted_subs[0].score - sorted_subs[1].score

        if score_diff >= delta:
            return (best_family, sorted_subs[0].external_id, best_family_score)

    # Fallback: family only
    return best_family, None, best_family_score


def select_pack_technique(
    pack: data.EventPack,
    delta: float = 0.02,
    k: int = 5,
) -> Tuple[str, Optional[str], float]:
    """
    Select Mitre technique for the pack by aggregating evidence
    from all events in the pack.

    Strategy:
    - aggregate by family using SUM of scores across events
    - pick best family, then best technique/sub-tech inside that family
    - emit sub-tech only if it clearly wins inside the family
    """
    if not pack.events:
        raise ValueError("Pack has no events")

    family_scores: dict[str, float] = defaultdict(float)
    tech_scores: dict[str, float] = defaultdict(float)

    used_events = 0
    for pe in pack.events:
        if not pe.top_5_techs:
            continue

        used_events += 1
        for t in pe.top_5_techs[:k]:
            fam = t.external_id.split(".")[0]
            family_scores[fam] += t.score
            tech_scores[t.external_id] += t.score

    if used_events == 0:
        raise ValueError("Pack has no events with top_5_techs")

    best_family = max(family_scores.items(), key=lambda kv: kv[1])[0]
    fam_prefix = best_family + "."

    in_family = {
        tech_id: score
        for tech_id, score in tech_scores.items()
        if tech_id == best_family or tech_id.startswith(fam_prefix)
    }

    if not in_family:
        in_family = {best_family: family_scores[best_family]}

    sorted_family_techs = sorted(
        in_family.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    best_tech_id, best_tech_score = sorted_family_techs[0]
    second_score = sorted_family_techs[1][1] if len(sorted_family_techs) > 1 else -1.0

    fam_event_scores: list[float] = []
    for pe in pack.events:
        if not pe.top_5_techs:
            continue
        best = 0.0
        for t in pe.top_5_techs[:k]:
            if t.external_id == best_family or t.external_id.startswith(fam_prefix):
                if t.score > best:
                    best = t.score
        fam_event_scores.append(best)

    confidence = float(sum(fam_event_scores) / max(len(fam_event_scores), 1))

    if "." in best_tech_id and (best_tech_score - second_score) >= delta:
        return best_family, best_tech_id, confidence

    return best_family, None, confidence


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
        embeddings_mitre_path: Optional[str] = None,
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

        self.embeddings_mitre_path: Optional[str] = None
        if embeddings_mitre_path is not None:
            self.embeddings_mitre_path = embeddings_mitre_path

        self.dump_embeddings = dump_embeddings

    def load_embed_train_mitre_patterns(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        embeddings_mitre_path = Path(self.embeddings_mitre_path)
        cnt = 0

        for mp in state.train_mitre_patterns:
            file_path = (
                embeddings_mitre_path / f"{mp.external_id.replace('.', '_')}.npy"
            )
            if not file_path.exists():
                logging.warning(f"There is not embedding for {mp.external_id}")
                continue

            mp.embed = np.load(file_path)
            cnt += 1

        if cnt == 0:
            raise Exception("Failed to load any embeddings for train mitre patterns")

        logging.info(f"Loaded {cnt} mitre embeddings")

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
        cnt = 0
        for pp in tqdm(state.norm_field_packs, desc="Packs filtering"):
            for pe in tqdm(pp.events, desc="Norm fields filtering", leave=False):
                # Filter keys from pe.norm_fields by state.filtered_fields
                filtered_items = []
                for key, value in pe.norm_fields.items():
                    if key in state.filtered_fields:
                        # Convert keys like subject.account.name -> Subject Account Name
                        converted_key = " ".join(
                            part.capitalize().replace("_", " ")
                            for part in key.split(".")
                        )
                        # Concatenate them in string like key: value separated by comma
                        filtered_items.append(f"{converted_key}: {value}")

                event_file_path = pp.pack_path / pe.event_file_name

                if len(filtered_items) == 0:
                    logger.warning(
                        f"Not found any important fields in {str(event_file_path)}"
                    )
                    continue

                pe.filtered_norm_text = (
                    ", ".join(filtered_items).strip() if filtered_items else ""
                )

                if logger.isEnabledFor(logging.DEBUG):
                    ff_norm_file = event_file_path.with_name(
                        event_file_path.name.replace(
                            "norm_fields_", "ff_norm_fields_"
                        ).replace(".json", ".txt")
                    )
                    ff_norm_file.write_text(pe.filtered_norm_text, encoding="utf-8")

                cnt += 1

        logger.info(f"Filtered {cnt} norm fields")

        return state

    def embed_filter_norm_fields(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        for pp in tqdm(state.norm_field_packs, desc="Filtered packs embed"):
            for pe in tqdm(pp.events, desc="Filtered norm fields embed", leave=False):
                event_file_path = pp.pack_path / pe.event_file_name

                if not pe.filtered_norm_text:
                    raise Exception(
                        f"There is no filtered norm text for {str(event_file_path)}"
                    )

                vector = self.elm.embed_query(pe.filtered_norm_text)
                pe.filtered_norm_embed = np.array(vector, dtype=np.float32)

                if self.dump_embeddings:
                    embed_file = event_file_path.with_name(
                        event_file_path.name.replace(
                            "norm_fields_", "embed_filtered_norm_fields_"
                        ).replace(".json", ".npy")
                    )

                    np.save(embed_file, pe.filtered_norm_embed)

        return state

    def calc_neighbors_tech(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        nn_index = build_dd_index(state.train_mitre_patterns)

        cnt = 0
        for pp in state.norm_field_packs:
            for pe in pp.events:
                top_5_techs = retrieve_top_k(
                    nn_index, state.train_mitre_patterns, pe.filtered_norm_embed, k=5
                )

                pe.top_5_techs = top_5_techs

                # Keep per-event decision (useful for debugging), but final output is per-pack.
                technique, sub_technique, confidence = select_technique(pe)
                pe.tech_id = technique
                pe.sub_tech_id = sub_technique
                pe.tech_score = confidence

                if technique is None:
                    logger.warning(
                        f"There is no technique for event {pe.event_file_name}"
                    )

                cnt += 1

            pack_tech, pack_subtech, pack_conf = select_pack_technique(pp)
            pp.pack_tech_id = pack_tech
            pp.pack_sub_tech_id = pack_subtech
            pp.pack_tech_score = pack_conf

        logger.info(f"Found {cnt} tactics")

        return state

    def dump_raw_tech(self, state: CorrelationAgentState) -> CorrelationAgentState:
        mitre_by_id = {p.external_id: p for p in state.train_mitre_patterns}

        cnt = 0
        for pp in state.norm_field_packs:
            tactics: set[str] = set()

            technique_name = ""
            sub_technique_name = ""

            if pp.pack_tech_id and pp.pack_tech_id in mitre_by_id:
                pattern = mitre_by_id[pp.pack_tech_id]
                technique_name = pattern.name
                tactics.update(pattern.phases)

            if pp.pack_sub_tech_id and pp.pack_sub_tech_id in mitre_by_id:
                pattern = mitre_by_id[pp.pack_sub_tech_id]
                sub_technique_name = pattern.name
                tactics.update(pattern.phases)

            technique_full = technique_name
            if sub_technique_name:
                technique_full = f"{technique_name}: {sub_technique_name}"

            output = {
                "technique": technique_full,
                "tactic": ", ".join(sorted(tactics)),
                "importance": "low, medium, high",
            }
            pp.answer = output

            answer_file = pp.pack_path / "answer.json"

            with answer_file.open("w", encoding="utf-8") as f:
                json.dump(output, f)

            cnt += 1

        logger.info(f"Dumped {cnt} answers")

        return state

    def verify_tactic_importance(
        self, state: CorrelationAgentState
    ) -> CorrelationAgentState:
        system_prompt = prompt.load_system_verify_tactic_prompt()

        mitre_by_id = {p.external_id: p for p in state.train_mitre_patterns}

        cnt = 0
        for pp in tqdm(state.norm_field_packs, desc="Verify packs"):
            ff_norm_fields = []

            for pe in islice(pp.events, 3):
                ff_norm_fields.append(pe.filtered_norm_text)

            pattern = mitre_by_id[pp.pack_tech_id]

            verify_prompt = prompt.load_verify_tactic_prompt(
                technique=pp.answer["technique"],
                tactic=normalize_tactics(pp.answer["tactic"]),
                importance=pp.answer["importance"],
                description=pattern.description,
                ff_norm_fields="\n".join(ff_norm_fields),
                correlation_path=pp.pack_path / "tests",
            )

            messages: list[BaseMessage] = [
                system_prompt,
                HumanMessage(content=verify_prompt),
            ]

            result = self.llm.invoke(messages)

            llm_tactic = None
            llm_importance = None

            try:
                llm_response = json.loads(result.content)
                llm_tactic = llm_response.get("tactic")
                llm_importance = llm_response.get("importance")
            except (json.JSONDecodeError, AttributeError) as e:
                logger.warning(
                    f"Failed to parse LLM response as JSON for {pp.pack_path}: {e}"
                )

            # Update answer with LLM values if valid, otherwise keep original
            if llm_tactic is not None:
                pp.answer["tactic"] = llm_tactic.capitalize().strip()
            else:
                logger.warning(
                    f"LLM did not provide valid tactic for {pp.pack_path}, keeping original"
                )

            if llm_importance is not None:
                pp.answer["importance"] = llm_importance.lower().strip()
            else:
                logger.warning(
                    f"LLM did not provide valid importance for {pp.pack_path}, keeping original"
                )

            answer_file = pp.pack_path / "answer.json"
            with answer_file.open("w", encoding="utf-8") as f:
                json.dump(pp.answer, f)

            cnt += 1

        logger.info(f"Dumped {cnt} verified answers")

        return state

    def build_graph(self):
        """
        Assemble and compile the full workflow graph.
        """
        builder = StateGraph(CorrelationAgentState)

        # Main pipeline nodes
        if self.embeddings_mitre_path is None:
            builder.add_node(
                "embed_train_mitre_patterns", self.embed_train_mitre_patterns
            )
        else:
            builder.add_node(
                "embed_train_mitre_patterns", self.load_embed_train_mitre_patterns
            )
        builder.add_node("filtered_fields", self.load_filtered_fields)
        builder.add_node("filter_norm_fields", self.filter_norm_fields)
        builder.add_node("embed_filter_norm_fields", self.embed_filter_norm_fields)
        builder.add_node("calc_neighbors_tech", self.calc_neighbors_tech)
        builder.add_node("dump_raw_tech", self.dump_raw_tech)
        builder.add_node("verify_tactic_importance", self.verify_tactic_importance)

        # The flow
        builder.add_edge(START, "embed_train_mitre_patterns")
        builder.add_edge("embed_train_mitre_patterns", "filtered_fields")
        builder.add_edge("filtered_fields", "filter_norm_fields")
        builder.add_edge("filter_norm_fields", "embed_filter_norm_fields")
        builder.add_edge("embed_filter_norm_fields", "calc_neighbors_tech")
        builder.add_edge("calc_neighbors_tech", "dump_raw_tech")
        builder.add_edge("dump_raw_tech", "verify_tactic_importance")
        builder.add_edge("verify_tactic_importance", END)

        return builder.compile()
