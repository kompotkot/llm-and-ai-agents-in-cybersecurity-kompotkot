import argparse
import logging
from pathlib import Path

from langfuse.langchain import CallbackHandler

from . import agents, config, data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

            event_text = event_file.read_text()

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
            event_text = event_file.read_text()
            norm_text = norm_file.read_text()

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


def normalize_handler(args: argparse.Namespace) -> None:
    if args.debug:
        logging.getLogger("src").setLevel(logging.DEBUG)

    # Initialize LLM Agent for normalization
    norm_agent = agents.NormalizationAgent()

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

    system_prompt = data_path / "system_prompt.txt"
    if system_prompt.exists():
        system_prompt.unlink()
        cnt += 1

    system_clean_prompt = data_path / "system_clean_prompt.txt"
    if system_clean_prompt.exists():
        system_clean_prompt.unlink()
        cnt += 1

    for test_dir in data_path.glob("*/tests"):
        if not test_dir.is_dir():
            continue

        # Scan for norm field files in current test directory
        for nf_file in test_dir.glob("norm_fields_*.json"):
            if not nf_file.exists():
                continue

            nf_file.unlink()
            cnt += 1

        # Scan for prompt files in current test directory
        for prompt_file in test_dir.glob("prompt_*.txt"):
            if not prompt_file.exists():
                continue

            prompt_file.unlink()
            cnt += 1

        # Scan for clean prompt files in current test directory
        for clean_prompt_file in test_dir.glob("clean_prompt_*.txt"):
            if not clean_prompt_file.exists():
                continue

            clean_prompt_file.unlink()
            cnt += 1

    logger.info(f"Removed {cnt} files")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent CLI")
    parser.set_defaults(func=lambda _: parser.print_help())
    subcommands = parser.add_subparsers(description="Agent commands")

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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
