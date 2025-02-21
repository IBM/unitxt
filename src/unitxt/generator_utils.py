from typing import Any, Dict, List

from .dataclass import Dataclass, OptionalField


class ReusableGenerator(Dataclass):
    generator: callable
    gen_argv: List[Any] = OptionalField(default_factory=list)
    gen_kwargs: Dict[str, Any] = OptionalField(default_factory=dict)
    already_processed_from_feeding_stream: int = 0
    feeding_stream: Any = None

    def activate(self):
        return self.generator(*self.gen_argv, **self.gen_kwargs)

    def __iter__(self):
        from .operator import InstanceOperator
        if hasattr(self.generator, "__self__") and isinstance(self.generator.__self__, InstanceOperator):
            self.feeding_stream = self.gen_kwargs["stream"]
            # print("self.generator.__self__::", self.generator.__self__)
            # print("self.feeding_stream::", self.feeding_stream)
            for i, instance in enumerate(self.feeding_stream):
                if i < self.already_processed_from_feeding_stream:
                    yield instance
                instance = self.generator.__self__.process(instance)
                self.already_processed_from_feeding_stream += 1
                yield instance
        else:
            for instance in self.activate():
                yield instance
            # yield from self.activate()

    def __call__(self):
        yield from iter(self)


class CopyingReusableGenerator(ReusableGenerator):
    pass
    # def __iter__(self):
    #     for instance in self.activate():
    #         yield recursive_copy(instance)
