from datetime import datetime, timezone
from typing import Sequence, Tuple, Dict
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
from pathlib import Path
from overriden_guideline_matcher import CustomGuidelineMatching, CustomGuidelineMatcher
from parlant.core.tags import TagId


# Function to create a guideline matcher object
async def create_guideline_matcher(config) -> Tuple[CustomGuidelineMatcher, FileLogger]:
    """Create a guideline matcher object"""
    correlator = ContextualCorrelator()

    if Path(config["logger_path"]).exists():
        Path(config["logger_path"]).unlink()

    logger = FileLogger(Path(config["logger_path"]), correlator, LogLevel.INFO)
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


# Create events from conversation
def conversation_to_events(
    config: Dict, conversation: pd.Series, customer_role, agent_role
) -> Sequence[Event]:
    """Convert a conversation to a sequence of events."""
    next_offset = -1
    events = []

    for line in conversation[config["conversation_column"]].iloc[0].split("\n"):
        if not line:
            continue

        next_offset += 1

        if line.startswith("user: "):
            message = line[6:]  # Remove "user: " prefix
            role = customer_role
        elif line.startswith("assistant: "):
            message = line[11:]  # Remove "assistant: " prefix
            role = agent_role
        else:
            continue

        events.append(
            Event(
                id=EventId(generate_id()),
                source=(
                    EventSource.AI_AGENT if role == agent_role else EventSource.CUSTOMER
                ),
                kind=EventKind.MESSAGE,
                creation_utc=datetime.now(timezone.utc),
                offset=next_offset,
                correlation_id=str(conversation[config["conversation_id_column"]]),
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


def get_roles(config: Dict) -> Tuple:
    """Create an agent and a customer"""
    agent = Agent(
        id=AgentId(config["agent_id"]),
        name="Test Agent",
        max_engine_iterations=1,
        composition_mode=CompositionMode.FLUID,
        description=config["agent_description"],
        tags=[TagId(f"agent:{config['agent_id']}")],
        creation_utc=datetime.now(timezone.utc),
    )

    customer = Customer(
        id=CustomerId(config["customer_id"]),
        name="Test Customer",
        extra={},
        tags=[],
        creation_utc=datetime.now(timezone.utc),
    )

    return agent, customer


# Function to load the data and sample some random conversations
def get_conversations(config: Dict, N: int = 5) -> pd.DataFrame:
    """Get the conversations from the disk"""
    conversations = pd.read_csv(config["conversations_path"])
    if config["debug"]:
        num_samples = min(len(conversations), N)
        conversations = conversations.sample(num_samples, random_state=42).reset_index(
            drop=True
        )
    return conversations


# Function to make a guideline given the condition and action
def make_guideline(
    guideline_id: str, condition: str, action: str, config: Dict
) -> Guideline:
    """Given a guideline id, condition and action, create a parlant guideline object"""
    return Guideline(
        id=GuidelineId(guideline_id),
        creation_utc=datetime.now(timezone.utc),
        enabled=True,
        content=GuidelineContent(condition=condition, action=action),
        tags=[TagId(f"agent:{config['agent_id']}")],
        metadata={},
    )
