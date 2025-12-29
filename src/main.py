import argparse
import json
import logging
import re
from pathlib import Path

import yaml
from langchain_ollama import ChatOllama
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

            # Remove citations from desc (format: (Citation: ...))
            desc = re.sub(r"\(Citation:[^)]+\)", "", desc)
            # Remove urls from desc
            desc = re.sub(r"https?://[^\s\)]+", "", desc)
            # Clean up extra spaces that may result from removals
            desc = re.sub(r"\s+", " ", desc).strip()

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


def load_norm_fields(
    base_path: Path,
    preds: int = 0,
) -> list[data.EventPack]:
    packs: list[data.EventPack] = []
    cnt = 0

    for test_dir in base_path.glob("*/tests"):
        if preds != 0 and cnt >= preds:
            break

        if not test_dir.is_dir():
            continue

        pack = data.EventPack(pack_path=test_dir.parent, events=[])

        # Scan for norm field files in current test directory
        for nf_file in test_dir.glob("norm_fields_*.json"):
            if preds != 0 and cnt >= preds:
                break

            with nf_file.open(mode="r", encoding="utf-8") as f:
                nf_data = json.load(f)

            pack.events.append(
                data.Event(
                    event_file_name=str(nf_file.relative_to(pack.pack_path)),
                    norm_fields=nf_data,
                )
            )
            cnt += 1

        packs.append(pack)

    logger.info(f"Found {cnt} norm fields in {len(packs)} packs")

    return packs


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

    if args.embeddings_mitre is not None:
        embeddings_mitre_path = Path(args.embeddings_mitre)
        if not embeddings_mitre_path.is_dir():
            logger.error(
                f"There is no embeddings mitre path: {str(embeddings_mitre_path)}"
            )
            return

    # Generate compact MITRE ATT&CK patterns
    m_patterns = mitre_compact_patterns(config.MITRE_CTI_PATH, args.pats)
    with open(config.MITRE_COMPACT_PATTERNS_PATH, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in m_patterns], f)
        logger.info(
            f"Compact {len(m_patterns)} Mitre patterns at {config.MITRE_COMPACT_PATTERNS_PATH}"
        )

    norm_field_packs = load_norm_fields(config.TEST_DATA_PATH, args.preds)

    # Initialize LLM Agent for correlations
    corr_agent = agents.CorrelationAgent(
        filtered_fields_path=args.filtered_fields,
        embeddings_mitre_path=args.embeddings_mitre,
        dump_embeddings=args.dump_embeddings,
    )

    graph = corr_agent.build_graph()

    state = agents.CorrelationAgentState(
        train_mitre_patterns=m_patterns, norm_field_packs=norm_field_packs
    )

    callbacks = []
    if args.langfuse:
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)
    result = graph.invoke(
        state,
        config={"callbacks": callbacks},
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

    graph = norm_agent.build_graph()

    state = agents.NormalizationAgentState(
        train_packs=train_packs, pred_packs=test_packs
    )

    callbacks = []
    if args.langfuse:
        langfuse_handler = CallbackHandler()
        callbacks.append(langfuse_handler)
    result = graph.invoke(
        state,
        config={"callbacks": callbacks},
    )


def taxonomy_fields_handler(args: argparse.Namespace) -> None:
    data_path = args.path
    if data_path is None:
        data_path = config.TEST_DATA_PATH

    tax_path = config.TAXONOMY_EN_PATH
    if args.path is not None:
        tax_path = Path(args.path)
        if not tax_path.exists():
            logger.error(f"There is not file {str(tax_path)}")
            return

    data = yaml.safe_load(tax_path.read_text(encoding="utf-8"))
    taxonomy_text = json.dumps(data, indent="\t", ensure_ascii=False)

    llm = ChatOllama(
        model=config.LLM_MODEL,
        base_url=config.LLM_API_URI,
    )

    chain = prompt.PROMPT_IMPORTANT_FIELDS | llm
    response = chain.invoke({"taxonomy_guideline": taxonomy_text})
    filtered_fields = response.replace(" ", "")

    ff_file_path = tax_path.name.replace(".yml", "_filtered_fields.txt")
    ff_file_path.write_text(filtered_fields, encoding="utf-8")
    logger.info(f"Dumped filtered fields of Taxonomy guideline to {str(ff_file_path)}")


def utils_clean_handler(args: argparse.Namespace) -> None:
    data_dir = args.path
    if data_dir is None:
        data_dir = config.BASE_DIR / "data"

    train_data_path = data_dir / "macos_correlation_rules"
    test_data_path = data_dir / "windows_correlation_rules"

    logger.info(f"Operating at {train_data_path} and {test_data_path}")

    cnt = 0

    correlation_del = [
        "*.txt",
    ]
    correlation_test_del = [
        "*.txt",
    ]

    if args.norm_fields:
        correlation_test_del.append("norm_fields_*.json")
    if args.embeddings:
        correlation_test_del.append("*.npy")

        for test_dir in train_data_path.glob("*/*/tests"):
            if not test_dir.is_dir():
                continue
            for f in test_dir.glob("*.npy"):
                if f.exists():
                    f.unlink()
                    cnt += 1

    # Delete files from test data directory
    for name_pattern in correlation_del:
        for f in test_data_path.glob(name_pattern):
            if f.exists():
                f.unlink()
                cnt += 1

    # Process each correlation directory
    for correlation_dir in test_data_path.glob("correlation_*"):
        if not correlation_dir.is_dir():
            continue

        # Delete answers.json from correlation directory
        answer_file = correlation_dir / "answers.json"
        if answer_file.exists():
            answer_file.unlink()
            cnt += 1

        # Delete files from tests subdirectory
        test_dir = correlation_dir / "tests"
        if test_dir.is_dir():
            for name_pattern in correlation_test_del:
                for f in test_dir.glob(name_pattern):
                    if f.exists():
                        f.unlink()
                        cnt += 1

    logger.info(f"Removed {cnt} files")


def utils_dump_handler(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        logger.error(f"There is no directory {output_dir}")
        return

    output_correlation_dir = output_dir / "windows_correlation_rules"
    output_correlation_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        f"Created output correlation directory at {str(output_correlation_dir)}"
    )

    data_dir = config.BASE_DIR / "data" / "windows_correlation_rules"

    for correlation_dir in data_dir.glob("correlation_*"):
        if not correlation_dir.is_dir():
            continue

        # Create in output_dir same correlation_* dir
        output_corr_dir = output_correlation_dir / correlation_dir.name
        output_corr_dir.mkdir(parents=True, exist_ok=True)

        # Copy in it correlation_*/answers.json
        answer_file = correlation_dir / "answers.json"
        if answer_file.exists():
            (output_corr_dir / "answers.json").write_text(
                answer_file.read_text(encoding="utf-8"), encoding="utf-8"
            )

        # Create in it correlation_*/tests dir
        output_tests_dir = output_corr_dir / "tests"
        output_tests_dir.mkdir(parents=True, exist_ok=True)

        # Copy in it correlation_*/tests/events_*.json
        source_tests_dir = correlation_dir / "tests"
        if source_tests_dir.is_dir():
            for event_file in source_tests_dir.glob("events_*.json"):
                (output_tests_dir / event_file.name).write_text(
                    event_file.read_text(encoding="utf-8"), encoding="utf-8"
                )

        # Copy in it correlation_*/tests/norm_fields_*.json
        if source_tests_dir.is_dir():
            for norm_file in source_tests_dir.glob("norm_fields_*.json"):
                (output_tests_dir / norm_file.name).write_text(
                    norm_file.read_text(encoding="utf-8"), encoding="utf-8"
                )

        # Create in it correlation_*/i18n dir
        output_i18n_dir = output_corr_dir / "i18n"
        output_i18n_dir.mkdir(parents=True, exist_ok=True)

        # TODO(kompotkot): Finish it
        (output_i18n_dir / "i18n_en.yaml").touch()
        (output_i18n_dir / "i18n_ru.yaml").touch()

    logger.info(f"Dumped correlation rules to {str(output_correlation_dir)}")


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
        "--embeddings-mitre",
        type=str,
        help="Path to directory with mitre embeddings to load from",
    )
    parser_correlate.add_argument(
        "-f",
        "--filtered-fields",
        type=str,
        help="Path to file with filtered Taxonomy fields",
    )
    parser_correlate.add_argument(
        "--pats",
        type=int,
        default=0,
        help="Amount of files to use for train",
    )
    parser_correlate.add_argument(
        "-p",
        "--preds",
        type=int,
        default=0,
        help="Amount of files to use for prediction",
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

    # Taxonomy command parser
    parser_taxomomy = subcommands.add_parser("taxomomy", description="Agent taxonomy")
    parser_taxomomy.set_defaults(func=lambda _: parser_taxomomy.print_help())
    subcommands_taxomomy = parser_taxomomy.add_subparsers(
        description="Agent taxonomy commands"
    )

    parser_tax_fields = subcommands_taxomomy.add_parser(
        "fields", description="Return list of fields from Taxonomy guideline"
    )
    parser_tax_fields.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Set this flag for debug",
    )
    parser_tax_fields.add_argument(
        "-p",
        "--path",
        help="Path to Taxonomy guideline file",
    )
    parser_tax_fields.set_defaults(func=taxonomy_fields_handler)

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
        help="Path to file with data",
    )
    parser_utils_clean.add_argument(
        "--norm-fields",
        action="store_true",
        help="Set this flag do delete norm fields",
    )
    parser_utils_clean.add_argument(
        "--embeddings",
        action="store_true",
        help="Set this flag do delete embeddings",
    )
    parser_utils_clean.set_defaults(func=utils_clean_handler)

    parser_utils_dump = subcommands_utils.add_parser("dump", description="Dump result")
    parser_utils_dump.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Path to directory to dump results",
    )
    parser_utils_dump.set_defaults(func=utils_dump_handler)

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
