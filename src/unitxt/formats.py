from .artifact import Artifact


class Format(Artifact):
    pass


class SizeLimitingFormat(Format):
    size_limiter: Artifact = None


class ICLFormat(SizeLimitingFormat):
    prefix: str = ""
    input_prefix: str = ""
    output_prefix: str = ""
    target_prefix: str = " "
    instruction_prefix: str = ""
    input_output_separator: str = "\n"
    demo_separator: str = "\n\n"
    suffix: str = ""
    add_instruction_at_start: bool = True
    add_instruction_after_demos: bool = False

    def single_source_str(self, source):
        return (
            self.input_prefix
            + source
            + self.input_output_separator
            + self.output_prefix
        )

    def single_source_str_with_instruction(self, source, instruction):
        return (
            self.input_prefix
            + instruction
            + self.demo_separator
            + source
            + self.input_output_separator
            + self.output_prefix
        )

    def format(self, instance, demos_instances=None):
        if demos_instances is None:
            demos_instances = []
        source = self.prefix

        instruction = ""
        if "instruction" in instance:
            instruction = instance.pop("instruction")
            assert (
                "instruction" != None
            ), f"instruction field can not be none : {instance}"

        if self.add_instruction_at_start and instruction != "":
            source += self.instruction_prefix + instruction + self.demo_separator

        if self.add_instruction_after_demos and instruction != "":
            query_str = self.single_source_str_with_instruction(
                instance["source"], instruction
            )
        else:
            query_str = self.single_source_str(instance["source"])

        for demo_instance in demos_instances:
            demo_str = (
                self.single_source_str(demo_instance["source"])
                + self.target_prefix
                + demo_instance["target"]
                + self.demo_separator
            )

            if self.size_limiter is not None:
                if not self.size_limiter.check(
                    source + demo_str + query_str + instance["target"]
                ):
                    continue

            source += demo_str

        source += query_str
        source += self.suffix
        return source
