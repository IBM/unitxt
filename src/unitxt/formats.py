from .artifact import Artifact


class Format(Artifact):
    pass


class SizeLimitingFormat(Format):
    size_limiter: Artifact = None


class ICLFormat(SizeLimitingFormat):
    prefix: str = ""
    input_prefix: str = ""
    output_prefix: str = ""
    instruction_prefix: str = ""
    input_output_separator: str = None
    demo_separator: str = "\n\n"
    suffix: str = ""
    add_instruction_at_start: bool = True
    add_instruction_after_demos: bool = False

    def single_source_str(self, source, input_output_separator, instruction=None):
        source_str = self.input_prefix
        if instruction is not None and instruction != "":
            source_str += self.instruction_prefix + instruction + self.demo_separator
        source_str += source
        if self.input_output_separator is not None and self.input_output_separator != "":
            source_str += self.input_output_separator
        else:
            source_str += input_output_separator
        source_str += self.output_prefix
        return source_str

    def format(self, instance, demos_instances=[]):
        source = self.prefix
        input_output_separator = instance.pop("input_output_separator")

        instruction = ""
        if "instruction" in instance:
            instruction = instance.pop("instruction")
            assert "instruction" != None, f"instruction field can not be none : {instance}"

        if self.add_instruction_at_start and instruction != "":
            source += self.instruction_prefix + instruction + self.demo_separator

        if self.add_instruction_after_demos and instruction != "":
            query_str = self.single_source_str(
                instance["source"], input_output_separator=input_output_separator, instruction=instruction
            )
        else:
            query_str = self.single_source_str(instance["source"], input_output_separator=input_output_separator)

        for demo_instance in demos_instances:
            demo_str = (
                self.single_source_str(demo_instance["source"], input_output_separator=input_output_separator)
                + demo_instance["target"]
                + self.demo_separator
            )

            # Disabled unimplemented functionality
            #             if self.size_limiter is not None:
            #                if not self.size_limiter.check(source + demo_str + query_str + instance["target"]):
            #                    continue

            source += demo_str

        source += query_str
        source += self.suffix
        return source
