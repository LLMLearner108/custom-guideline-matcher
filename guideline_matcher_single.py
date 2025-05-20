from itertools import product
import yaml
import pickle, json
from tqdm import tqdm
from typing import List
import asyncio
from utils import *
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

GUIDELINES = json.load(open(config["guidelines_path"]))
GUIDELINES_TO_CONSIDER = [
    x["guideline_id"] for x in GUIDELINES
]  # This can be restricted to a small subset if need be

# If debug mode is true, then sample a very small number of conversations
N = 5

AGENT, CUSTOMER = get_roles(config)


# Process a conversation (i.e. identify if the given guideline applies to the provided conversation) and send the result back
async def process_conversation(
    gp: CustomGuidelineMatcher, conversation: pd.Series, guideline_id: str
) -> Tuple:
    """Process a single conversation with the guideline proposer."""
    events = conversation_to_events(config, conversation, CUSTOMER, AGENT)

    # Create the respective guideline
    guidelines = [
        make_guideline(guideline_id, x["condition"], x["action"], config)
        for x in GUIDELINES
        if x["guideline_id"] == guideline_id
    ]

    # Get the completion result
    result = await gp.match_guidelines(
        agent=AGENT,
        customer=CUSTOMER,
        guidelines=guidelines,
        context_variables=[],
        interaction_history=events,
        terms=[],
        staged_events=[],
    )

    return result


def filter_records(conversations, logger) -> List[Tuple]:
    """Given the conversations and a logger object, identify which conversation-guideline pairs were already evaluated, filter them out and return the records for which completion"""
    if Path(config["output_path"]).exists():
        processed_records = pickle.load(open(config["output_path"], "rb"))
        processed_records = [
            (x[config["conversation_id_column"]], x["guideline_id"])
            for x in processed_records
        ]
    else:
        processed_records = []

    all_cids = conversations[config["conversation_id_column"]].unique().tolist()
    all_gids = GUIDELINES_TO_CONSIDER
    all_records = list(product(all_cids, all_gids))

    records_to_be_processed = [x for x in all_records if not x in processed_records]

    logger.info(f"Total Guideline Conversation Pairs: {len(all_records)}")
    logger.info(f"Completions available for: {len(processed_records)}")
    logger.info(f"Completions to be obtained for: {len(records_to_be_processed)}")

    return records_to_be_processed


async def main() -> None:
    """Main application logic i.e. guideline matching logic"""

    # Initialize the guideline proposer
    gp, logger = await create_guideline_matcher(config)

    # Get conversations and filter the conversation-guideline records
    conversations = get_conversations(config, N)
    logger.info(
        f"Processing {len(conversations)} conversations alongside {len(GUIDELINES_TO_CONSIDER)} guidelines"
    )

    records_to_be_processed = filter_records(conversations, logger)

    outputs = (
        pickle.load(open(config["output_path"], "rb"))
        if Path(config["output_path"]).exists()
        else []
    )

    for rec in tqdm(records_to_be_processed, total=len(records_to_be_processed)):

        conversation_id, guideline_id = rec

        conversation = conversations[
            conversations[config["conversation_id_column"]] == conversation_id
        ]

        result = await process_conversation(gp, conversation, guideline_id)

        for result_batch in result:
            prompt, completion, usage = result_batch
            outputs.append(
                {
                    config["conversation_id_column"]: conversation[
                        config["conversation_id_column"]
                    ].iloc[0],
                    "guideline_id": guideline_id,
                    "prompt": prompt,
                    "completion": completion.model_dump(),
                    "usage": usage,
                }
            )

    with open(config["output_path"], "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    asyncio.run(main())