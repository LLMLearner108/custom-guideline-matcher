from itertools import product
import yaml
import pickle, json
from tqdm import tqdm
from typing import List
from collections import defaultdict
import asyncio
from utils import *
from dotenv import load_dotenv

# Load the environment variables
load_dotenv()

with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

GUIDELINES = json.load(open(config["guidelines_path"]))
GUIDELINE_MAP = {x["guideline_id"]: x for x in GUIDELINES}
GUIDELINES_TO_CONSIDER = [
    x["guideline_id"] for x in GUIDELINES
]  # This can be restricted to a small subset if need be

# If debug mode is true, then sample a very small number of conversations
N = 5

AGENT, CUSTOMER = get_roles(config)


def make_guidelines(guideline_ids: List[str]) -> List[Guideline]:
    """Create a list of parlant guideline objects for those guidelines that need to be evaluated"""
    parlant_guidelines = []
    for gid in guideline_ids:
        json_guideline = GUIDELINE_MAP[gid]
        parlant_guidelines.append(
            make_guideline(
                gid, json_guideline["condition"], json_guideline["action"], config
            )
        )
    return parlant_guidelines


# Process a conversation (i.e. identify if the given guideline applies to the provided conversation) and send the result back
async def process_conversation(
    gp: CustomGuidelineMatcher, conversation: pd.Series, guidelines: List[Guideline]
) -> Tuple:
    """Process a single conversation with the guideline proposer."""
    events = conversation_to_events(config, conversation, CUSTOMER, AGENT)

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
        data = pickle.load(open(config["output_path"], "rb"))
        processed_records = []

        for record in data:
            cid = record[config["conversation_id_column"]]
            gids = record["guideline_ids"]
            for gid in gids:
                processed_records.append((cid, gid))
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

    # Figure out which guidelines need to be evaluated for which conversations based on the above list of tuples
    cid_gid_map = defaultdict(lambda: [])
    for cid, gid in records_to_be_processed:
        cid_gid_map[cid].append(gid)

    outputs = (
        pickle.load(open(config["output_path"], "rb"))
        if Path(config["output_path"]).exists()
        else []
    )

    recs = list(cid_gid_map.items())
    for rec in tqdm(recs, total=len(recs)):

        conversation_id, guideline_ids = rec

        conversation = conversations[
            conversations[config["conversation_id_column"]] == conversation_id
        ]

        guidelines = make_guidelines(guideline_ids)
        result = await process_conversation(gp, conversation, guidelines)

        for result_batch in result:
            prompt, completion, usage = result_batch
            guideline_ids = [x.guideline_id for x in completion.checks]
            outputs.append(
                {
                    config["conversation_id_column"]: conversation[
                        config["conversation_id_column"]
                    ].iloc[0],
                    "guideline_ids": guideline_ids,
                    "prompt": prompt,
                    "completion": completion.model_dump(),
                    "usage": usage,
                }
            )

    with open(config["output_path"], "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    asyncio.run(main())
