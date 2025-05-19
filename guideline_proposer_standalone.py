import asyncio
from datetime import datetime, timezone
from typing import Sequence, Tuple, List
from parlant.adapters.nlp.openai_service import OpenAIService
from parlant.core.agents import Agent, AgentId, CompositionMode
from parlant.core.common import generate_id
from parlant.core.contextual_correlator import ContextualCorrelator
from parlant.core.customers import Customer, CustomerId
from parlant.core.engines.alpha.guideline_matcher import (
    GenericGuidelineMatchesSchema,
    DefaultGuidelineMatchingStrategyResolver,
)
from parlant.core.guidelines import Guideline, GuidelineContent, GuidelineId
from parlant.core.loggers import LogLevel, FileLogger
from parlant.core.sessions import (
    Event,
    EventId,
    MessageEventData,
    EventSource,
    EventKind,
)
import pandas as pd
import pickle, json
from pathlib import Path
from tqdm import tqdm
from overriden_guideline_matcher import CustomGuidelineMatching, CustomGuidelineMatcher
from parlant.core.tags import TagId
from itertools import product

LOGGER_PATH = "logs/custom_guideline_matching.log"
AGENT_ID = "test-agent"
CUSTOMER_ID = "test-customer"
AGENT_DESCRIPTION = "You are a banking agent who is highly skilled in indulging in conversations with customers and providing them answers to their questions related to digital banking services provided by Chase Bank."
OUTPUT_PATH = "completions.pkl"

CONVERSATIONS_PATH = "conversations.csv"
CONVERSATION_ID_COLUMN = "conversation_id"
CONVERSATION_COLUMN = "script_dialog"
GUIDELINES_PATH = "guidelines.json"

GUIDELINES = json.load(open(GUIDELINES_PATH))
GUIDELINES_TO_CONSIDER = [
    x["guideline_id"] for x in GUIDELINES
]  # This can be restricted to a small subset if need be

DEBUG = True
N = 50


# Function to create a guideline matcher object
async def create_guideline_matcher() -> Tuple[CustomGuidelineMatcher, FileLogger]:
    """Create a guideline matcher object"""
    correlator = ContextualCorrelator()

    if Path(LOGGER_PATH).exists():
        Path(LOGGER_PATH).unlink()

    logger = FileLogger(Path(LOGGER_PATH), correlator, LogLevel.INFO)
    nlp_service = OpenAIService(logger)
    gm_schema_generator = await nlp_service.get_schematic_generator(
        GenericGuidelineMatchesSchema
    )
    gen_strategy = CustomGuidelineMatching(
        logger, schematic_generator=gm_schema_generator
    )
    strategy = DefaultGuidelineMatchingStrategyResolver(
        generic_strategy=gen_strategy, logger=logger
    )

    return CustomGuidelineMatcher(logger, strategy), logger


# Create a test agent and test customer
AGENT = Agent(
    id=AgentId(AGENT_ID),
    name="Test Agent",
    max_engine_iterations=1,
    composition_mode=CompositionMode.FLUID,
    description=AGENT_DESCRIPTION,
    tags=[TagId(f"agent:{AGENT_ID}")],
    creation_utc=datetime.now(timezone.utc),
)


CUSTOMER = Customer(
    id=CustomerId(CUSTOMER_ID),
    name="Test Customer",
    extra={},
    tags=[],
    creation_utc=datetime.now(timezone.utc),
)


# Function to make a guideline given the condition and action
def make_guideline(guideline_id: str, condition: str, action: str) -> Guideline:
    """Given a guideline id, condition and action, create a parlant guideline object"""
    return Guideline(
        id=GuidelineId(guideline_id),
        creation_utc=datetime.now(timezone.utc),
        enabled=True,
        content=GuidelineContent(condition=condition, action=action),
        tags=[TagId(f"agent:{AGENT_ID}")],
        metadata={},
    )


# Function to load the data and sample some random conversations
def get_conversations() -> pd.DataFrame:
    """Get the conversations from the disk"""
    conversations = pd.read_csv(CONVERSATIONS_PATH)
    if DEBUG:
        num_samples = min(len(conversations), N)
        conversations = conversations.sample(num_samples, random_state=42).reset_index(
            drop=True
        )
    return conversations


# Create events from conversation
def conversation_to_events(conversation: pd.Series) -> Sequence[Event]:
    """Convert a conversation to a sequence of events."""
    next_offset = -1
    events = []

    for line in conversation[CONVERSATION_COLUMN].iloc[0].split("\n"):
        if not line:
            continue

        next_offset += 1

        if line.startswith("user: "):
            message = line[6:]  # Remove "user: " prefix
            role = CUSTOMER
        elif line.startswith("assistant: "):
            message = line[11:]  # Remove "assistant: " prefix
            role = AGENT
        else:
            continue

        events.append(
            Event(
                id=EventId(generate_id()),
                source=EventSource.AI_AGENT if role == AGENT else EventSource.CUSTOMER,
                kind=EventKind.MESSAGE,
                creation_utc=datetime.now(timezone.utc),
                offset=next_offset,
                correlation_id=str(conversation[CONVERSATION_ID_COLUMN]),
                data=MessageEventData(
                    message=message,
                    participant={
                        "id": role.id,
                        "display_name": role.name,
                    },
                ),
                deleted=False,
            )
        )

    return events


# Process a conversation (i.e. identify if the given guideline applies to the provided conversation) and send the result back
async def process_conversation(
    gp: CustomGuidelineMatcher, conversation: pd.Series, guideline_id: str
) -> Tuple:
    """Process a single conversation with the guideline proposer."""
    events = conversation_to_events(conversation)

    # Create the respective guideline
    guidelines = [
        make_guideline(guideline_id, x["condition"], x["action"])
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
    if Path(OUTPUT_PATH).exists():
        processed_records = pickle.load(open(OUTPUT_PATH, "rb"))
        processed_records = [
            (x[CONVERSATION_ID_COLUMN], x["guideline_id"]) for x in processed_records
        ]
    else:
        processed_records = []

    all_cids = conversations[CONVERSATION_ID_COLUMN].unique().tolist()
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
    gp, logger = await create_guideline_matcher()

    # Get conversations and filter the conversation-guideline records
    conversations = get_conversations()
    logger.info(
        f"Processing {len(conversations)} conversations alongside {len(GUIDELINES_TO_CONSIDER)} guidelines"
    )

    records_to_be_processed = filter_records(conversations, logger)

    outputs = pickle.load(open(OUTPUT_PATH, "rb")) if Path(OUTPUT_PATH).exists() else []

    for rec in tqdm(records_to_be_processed, total=len(records_to_be_processed)):

        conversation_id, guideline_id = rec
        
        conversation = conversations[
            conversations[CONVERSATION_ID_COLUMN] == conversation_id
        ]
        
        result = await process_conversation(gp, conversation, guideline_id)

        for result_batch in result:
            prompt, completion, usage = result_batch
            outputs.append(
                {
                    CONVERSATION_ID_COLUMN: conversation[CONVERSATION_ID_COLUMN].iloc[0],
                    "guideline_id": guideline_id,
                    "prompt": prompt,
                    "completion": completion.model_dump(),
                    "usage": usage,
                }
            )

    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(outputs, f)


if __name__ == "__main__":
    asyncio.run(main())
