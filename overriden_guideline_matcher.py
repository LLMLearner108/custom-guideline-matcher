from parlant.core.engines.alpha.guideline_matcher import (
    GenericGuidelineMatchingBatch,
    GuidelineMatchingContext,
    GuidelineMatchingStrategy,
    GenericGuidelineMatching,
    GuidelineMatchingStrategyResolver,
)

from parlant.core.sessions import Event
from parlant.core.emissions import EmittedEvent
from parlant.core.loggers import Logger
from parlant.core.context_variables import ContextVariable, ContextVariableValue
from parlant.core.guidelines import Guideline, GuidelineId
from parlant.core.agents import Agent
from parlant.core.customers import Customer
from parlant.core.glossary import Term
from parlant.core import async_utils
from dataclasses import asdict

from typing import Tuple, Sequence
import math, time
from typing_extensions import override


class CustomGuidelineMatchingBatch(GenericGuidelineMatchingBatch):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    async def process(self) -> Tuple:
        prompt = self._build_prompt(shots=await self.shots())

        with self._logger.operation(
            f"CustomGuidelineMatchingBatch: {len(self._guidelines)} guidelines"
        ):
            inference = await self._schematic_generator.generate(
                prompt=prompt,
                hints={"temperature": 0.15},
            )

        if not inference.content.checks:
            self._logger.warning(
                "Completion:\nNo checks generated! This shouldn't happen."
            )
        else:
            self._logger.debug(
                f"Completion:\n{inference.content.model_dump_json(indent=2)}"
            )

        return (prompt.build(), inference.content, asdict(inference.info))


class CustomGuidelineMatching(GenericGuidelineMatching):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_batch(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> CustomGuidelineMatchingBatch:
        return CustomGuidelineMatchingBatch(
            logger=self._logger,
            schematic_generator=self._schematic_generator,
            guidelines=guidelines,
            context=context,
        )

    @override
    async def create_batches(
        self,
        guidelines: Sequence[Guideline],
        context: GuidelineMatchingContext,
    ) -> Sequence[CustomGuidelineMatchingBatch]:
        batches = []

        guidelines_dict = {g.id: g for g in guidelines}
        batch_size = self._get_optimal_batch_size(guidelines_dict)
        guidelines_list = list(guidelines_dict.items())
        batch_count = math.ceil(len(guidelines_dict) / batch_size)

        for batch_number in range(batch_count):
            start_offset = batch_number * batch_size
            end_offset = start_offset + batch_size
            batch = dict(guidelines_list[start_offset:end_offset])
            batches.append(
                self._create_batch(
                    guidelines=list(batch.values()),
                    context=context,
                )
            )

        return batches

    @override
    def _get_optimal_batch_size(self, guidelines: dict[GuidelineId, Guideline]) -> int:
        return 3


class CustomGuidelineMatcher:
    def __init__(
        self,
        logger: Logger,
        strategy_resolver: GuidelineMatchingStrategyResolver,
    ) -> None:
        self._logger = logger
        self.strategy_resolver = strategy_resolver

    async def match_guidelines(
        self,
        agent: Agent,
        customer: Customer,
        context_variables: Sequence[tuple[ContextVariable, ContextVariableValue]],
        interaction_history: Sequence[Event],
        terms: Sequence[Term],
        staged_events: Sequence[EmittedEvent],
        guidelines: Sequence[Guideline],
    ) -> Tuple:
        if not guidelines:
            return ()

        t_start = time.time()

        with self._logger.scope("GuidelineMatcher"):
            with self._logger.operation("Creating batches"):
                guideline_strategies: dict[
                    str, tuple[GuidelineMatchingStrategy, list[Guideline]]
                ] = {}
                for guideline in guidelines:
                    strategy = await self.strategy_resolver.resolve(guideline)
                    if strategy.__class__.__name__ not in guideline_strategies:
                        guideline_strategies[strategy.__class__.__name__] = (
                            strategy,
                            [],
                        )
                    guideline_strategies[strategy.__class__.__name__][1].append(
                        guideline
                    )

                batches = []
                for _, (strategy, guidelines) in guideline_strategies.items():
                    batch = await strategy.create_batches(
                        guidelines,
                        context=GuidelineMatchingContext(
                            agent,
                            customer,
                            context_variables,
                            interaction_history,
                            terms,
                            staged_events,
                        ),
                    )
                    batches.append(batch)

            with self._logger.operation("Processing batches"):
                batch_tasks = [batch.process() for batch in batches[0]]
                batch_results = await async_utils.safe_gather(*batch_tasks)

        t_end = time.time()

        return batch_results
