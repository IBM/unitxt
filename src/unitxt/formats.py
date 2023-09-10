from .artifact import Artifact


class Format(Artifact):
    pass


class SizeLimitingFormat(Format):
    size_limiter: Artifact = None


class ICLFormat(SizeLimitingFormat):
    input_prefix: str = ""
    output_prefix: str = ""
    target_prefix: str = " "
    instruction_prefix: str = ""
    input_output_separator: str = "\n"
    demo_separator: str = "\n\n"

    def single_source_str(self, source):
        return self.input_prefix + source + self.input_output_separator + self.output_prefix

    def format(self, instance, demos_instances=[]):
        source = ""

        query_str = self.single_source_str(instance["source"])

        if "instruction" in instance:
            instruction = instance.pop("instruction")
            source += self.instruction_prefix + instruction + self.demo_separator

        for demo_instance in demos_instances:
            demo_str = (
                self.single_source_str(demo_instance["source"])
                + self.target_prefix
                + demo_instance["target"]
                + self.demo_separator
            )

            if self.size_limiter is not None:
                if not self.size_limiter.check(source + demo_str + query_str + instance["target"]):
                    continue

            source += demo_str

        source += query_str

        return source
