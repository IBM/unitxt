from typing import List, Optional, Generator
from dataclasses import asdict
import random
from abc import abstractmethod

from .stream import MultiStream, Stream
from .operator import SourceOperator, StreamSource
from .card import TaskCard, ICLCard
from .common import CommonRecipe

class BaseFusion(SourceOperator):
    """
    BaseFusion operator that combines multiple streams into one.
    
    Args:
        include_splits: List of splits to include. If None, all splits are included.
    """
    include_splits: Optional[List[str]] = None
    
    @abstractmethod
    def fusion_generator(self, split) -> Generator:
        pass
    
    def splits(self) -> Generator:
        splits = []
        for origin in self.origins:
            for s in origin().keys():
                if s not in splits:
                    if self.include_splits is None or s in self.include_splits:
                        splits.append(s)
        return splits
              

    def process(self, ) -> MultiStream:
        result = {}
        for split in self.splits():
            result[split] = Stream(self.fusion_generator, gen_kwargs={'split': split})
        return MultiStream(result)

class FixedFusion(BaseFusion):
    """
    FixedFusion operator that combines multiple streams into one based on a fixed number of examples per task.
    
    Args:
        orgins: List of StreamSource objects.
        examples_per_task: Number of examples per task. If None, all examples are returned.
        splits: List of splits to include. If None, all splits are included.
    """
    examples_per_task: Optional[int] = None
    
    def fusion_generator(self, split) -> Generator:
        for origin in self.orgins:
            iterator = iter(origin()[split])
            if self.examples_per_task is not None:
                for i in range(self.examples_per_task):
                    yield next(iterator)
            else:
                yield from iterator
    

class WeightedFusion(BaseFusion):
    """
    Fusion operator that combines multiple streams based 
    
    Args:
        orgins: List of StreamSource objects.
        weights: List of weights for each origin.
        total_examples: Total number of examples to return. If None, all examples are returned.
    """
    origins: List[StreamSource] = None
    weights: List[float] = None
    total_examples: int = None
    
    def verify(self):
        super().verify()
        assert self.origins is not None, "origins must be specified"
        assert self.weights is not None, "weights must be specified"
        assert len(self.origins) == len(self.weights), "origins and weights must have the same length"
    
    def fusion_generator(self, split) -> Generator:
        iterators = [iter(origin()[split]) for origin in self.origins]
        total_examples = 0
        while (self.total_examples is None or total_examples <= self.total_examples) \
            and len(iterators) > 0:
            iterator = random.choices(population=iterators, weights=self.weights)[0]
            try:
                yield next(iterator)
                total_examples += 1
            except StopIteration:
                iterators.remove(iterator)
    
class TasksFusion(SourceOperator):
    """
    TasksFusion operator that combines multiple tasks into one.
    
    Args:
        tasks: List of TaskCard objects.
        config: ICLCard object.
        examples_per_task: Number of examples per task. If None, all examples are returned.
        include_splits: List of splits to include. If None, all splits are included.
    """
    tasks: List[TaskCard]
    config: ICLCard
    examples_per_task: Optional[int] = None
    include_splits: Optional[List[str]] = None
    
    def prepare(self):
        self.recipes = []
        for task in self.tasks:
            recipe = CommonRecipe(
                card=task,
                **asdict(self.config)
            )
            
        self.fusion = FixedFusion(
            origins=self.recipes,
            examples_per_task=self.examples_per_task,
            include_splits=self.include_splits
        )

    def process(self) -> MultiStream:
        return self.fusion()

    
    
            
    
    
    