import argparse
import json
import logging
from pathlib import Path

import yaml
from langfuse.langchain import CallbackHandler

from . import agents, config, data, prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mitre_compact_patterns(base_path: Path, pats: int = 0) -> list[data.MitrePattern]:
    patterns: list[data.MitrePattern] = []
    cnt = 0

    for path in base_path.glob("enterprise-attack/attack-pattern/*.json"):
        if pats != 0 and cnt >= pats:
            break

        with open(path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        # Iterate through objects in the bundle
        for obj in bundle.get("objects", []):
            if pats != 0 and cnt >= pats:
                break

            platforms = obj.get("x_mitre_platforms", [])
            if "Windows" not in platforms:
                continue

            ext_id = ""
            for ref in obj.get("external_references", []):
                if ref.get("source_name") == "mitre-attack":
                    ext_id = ref.get("external_id")
                    break

            phases = []
            for p in obj.get("kill_chain_phases", []):
                phases.append(p.get("phase_name"))

            desc = obj.get("description", "")
            desc = desc.replace("\n", "").strip()

            name = obj.get("name", "")
            phases_str = ", ".join(phases)

            pattern = data.MitrePattern(
                external_id=ext_id,
                name=obj.get("name", ""),
                phases=phases,
                description=desc,
                text=f"{ext_id}. Tactics: {phases_str}. Name: {name}. Description: {desc}",
            )
            patterns.append(pattern)
            cnt += 1

    return patterns


def load_test_data(
    base_path: Path,
    preds: int = 0,
) -> list[data.EventPack]:
    """
    Loads prediction data from directory structure by finding test directories and processing event files.

    Searches for pattern: base_path/*/tests/events_*.json files.
    """
    packs: list[data.EventPack] = []
    cnt = 0

    # Iterate through all test directories matching pattern: base_path/*/tests
    for test_dir in base_path.glob("*/tests"):
        if preds != 0 and cnt >= preds:
            break

        if not test_dir.is_dir():
            continue

        pack = data.EventPack(pack_path=test_dir.parent, events=[])

        # Scan for event files in current test directory
        for event_file in test_dir.glob("events_*.json"):
            if preds != 0 and cnt >= preds:
                break
            event_text = event_file.read_text(encoding="utf-8")

            pack.events.append(
                data.Event(
                    event_file_name=str(event_file.relative_to(pack.pack_path)),
                    event_text=event_text,
                )
            )
            cnt += 1

        packs.append(pack)

    logger.info(f"Found {cnt} events in {len(packs)} packs to normalize")

    return packs


def load_train_data(
    base_path: Path,
    references: int = 0,
) -> list[data.EventPack]:
    """
    Loads reference examples from directory structure by finding test directories and processing event files.

    Searches for pattern: base_path/*/tests/events_*.json and corresponding norm_fields_*.json files.
    """
    packs: list[data.EventPack] = []
    cnt = 0

    # Iterate through all test directories matching pattern: base_path/*/*/tests
    for test_dir in base_path.glob("*/*/tests"):
        if references != 0 and cnt >= references:
            break

        if not test_dir.is_dir():
            continue

        pack = data.EventPack(pack_path=test_dir.parent, events=[])

        # Scan for event files in current test directory
        for event_file in test_dir.glob("events_*.json"):
            if references != 0 and cnt >= references:
                break

            norm_file = event_file.with_name(
                event_file.name.replace("events_", "norm_fields_")
            )
            # Skip if normalized file doesn't exist
            if not norm_file.exists():
                continue

            # Read event and normalized text files
            event_text = event_file.read_text(encoding="utf-8")
            norm_text = norm_file.read_text(encoding="utf-8")

            pack.events.append(
                data.Event(
                    event_file_name=str(event_file.relative_to(pack.pack_path)),
                    event_text=event_text,
                    norm_text=norm_text,
                )
            )

            cnt += 1

        packs.append(pack)

    logger.info(f"Found {cnt} references in {len(packs)} packs")

    return packs


def correlate_handler(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger("src").setLevel(logging.DEBUG)

    embeddings_path = Path(args.embeddings)
    if not embeddings_path.is_dir():
        logger.error(f"There is no embeddings path: {str(embeddings_path)}")
        return

    # Generate compact MITRE ATT&CK patterns
    m_patterns = mitre_compact_patterns(config.MITRE_CTI_PATH, args.pats)
    with open(config.MITRE_COMPACT_PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in m_patterns], f)
        logger.info(
            f"Compact {len(m_patterns)} Mitre patterns at {config.MITRE_COMPACT_PATTERNS_PATH}"
        )

    # Initialize LLM Agent for correlations
    corr_agent = agents.CorrelationAgent(
        embeddings=args.embeddings, dump_embeddings=args.dump_embeddings
    )

    graph = corr_agent.build_graph()

    state = agents.CorrelationAgentState(train_mitre_patterns=m_patterns)

    if args.langfuse:
        langfuse_handler = CallbackHandler()
        result = graph.invoke(
            state,
            config={"callbacks": [langfuse_handler]},
        )


def normalize_handler(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger("src").setLevel(logging.DEBUG)

    # Initialize LLM Agent for normalization
    norm_agent = agents.NormalizationAgent(dump_embeddings=args.dump_embeddings)

    # Load all reference examples from the training data directory
    train_packs = load_train_data(config.TRAIN_DATA_PATH, args.references)

    # Load all events for prediction
    test_packs = load_test_data(config.TEST_DATA_PATH, args.preds)

    # # Prepare taxonomy prompt
    # taxonomy_ru_prompt, taxonomy_en_prompt = prompt.load_taxonomy_prompt(
    #     config.TAXONOMY_RU_PATH, config.TAXONOMY_EN_PATH
    # )

    graph = norm_agent.build_graph()

    state = agents.NormalizationAgentState(
        train_packs=train_packs, pred_packs=test_packs
    )

    if args.langfuse:
        langfuse_handler = CallbackHandler()
        result = graph.invoke(
            state,
            config={"callbacks": [langfuse_handler]},
        )


def utils_clean_handler(args: argparse.Namespace) -> None:
    data_path = args.path
    if data_path is None:
        data_path = config.TEST_DATA_PATH

    logger.info(f"Operating at: {data_path}")

    cnt = 0

    for name in [
        "system_prompt.txt",
        "system_clean_prompt.txt",
        "taxonomy_fields_prompt.txt",
    ]:
        system_prompt = data_path / name
    if system_prompt.exists():
        system_prompt.unlink()
        cnt += 1

    for test_dir in data_path.glob("*/tests"):
        if not test_dir.is_dir():
            continue

        for name_pattern in [
            "norm_fields_*.json",
            "prompt_*.txt",
            "clean_prompt_*.txt",
        ]:
            for f in test_dir.glob(name_pattern):
                if not f.exists():
                    continue

                f.unlink()
                cnt += 1

    logger.info(f"Removed {cnt} files")


def utils_test_handler(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger("src").setLevel(logging.DEBUG)

    logger.debug("Test")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent CLI")
    parser.set_defaults(func=lambda _: parser.print_help())
    subcommands = parser.add_subparsers(description="Agent commands")

    # Correlate command parser
    parser_correlate = subcommands.add_parser(
        "correlate", description="Generate correlation tactics and techniques"
    )
    parser_correlate.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Set this flag for debug",
    )
    parser_correlate.add_argument(
        "-e",
        "--embeddings",
        type=str,
        help="Path to directory with embeddings to load from",
    )
    parser_correlate.add_argument(
        "-p",
        "--pats",
        type=int,
        default=0,
        help="Amount of files to use for train",
    )
    parser_correlate.add_argument(
        "-l",
        "--langfuse",
        action="store_true",
        help="Set this flag for langfuse support",
    )
    parser_correlate.add_argument(
        "--dump-embeddings",
        action="store_true",
        help="Set this flag to dump generated embeddings",
    )
    parser_correlate.set_defaults(func=correlate_handler)

    # Normalize command parser
    parser_normalize = subcommands.add_parser(
        "normalize", description="Generate normalized SIEM fields"
    )
    parser_normalize.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Set this flag for debug",
    )
    parser_normalize.add_argument(
        "-l",
        "--langfuse",
        action="store_true",
        help="Set this flag for langfuse support",
    )
    parser_normalize.add_argument(
        "-r",
        "--references",
        type=int,
        default=0,
        help="Amount of files to use for train",
    )
    parser_normalize.add_argument(
        "-p",
        "--preds",
        type=int,
        default=0,
        help="Amount of files to use for prediction",
    )
    parser_normalize.add_argument(
        "--dump-embeddings",
        action="store_true",
        help="Set this flag to dump generated embeddings",
    )
    parser_normalize.set_defaults(func=normalize_handler)

    # Util command parser
    parser_utils = subcommands.add_parser("utils", description="Agent utils")
    parser_utils.set_defaults(func=lambda _: parser_utils.print_help())
    subcommands_utils = parser_utils.add_subparsers(description="Agent util commands")

    parser_utils_clean = subcommands_utils.add_parser(
        "clean", description="Clean data from newly generated files"
    )
    parser_utils_clean.add_argument(
        "-p",
        "--path",
        help="Path to file with test data",
    )
    parser_utils_clean.set_defaults(func=utils_clean_handler)

    parser_utils_test = subcommands_utils.add_parser(
        "test", description="For test purposes"
    )
    parser_utils_test.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Set this flag for debug",
    )
    parser_utils_test.set_defaults(func=utils_test_handler)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
