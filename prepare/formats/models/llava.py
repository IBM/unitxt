from unitxt import add_to_catalog
from unitxt.formats import HFSystemFormat

format = HFSystemFormat(model_name="llava-hf/llava-1.5-7b-hf")

add_to_catalog(format, "formats.models.llava", overwrite=True)


format = HFSystemFormat(model_name="llava-hf/llava-interleave-qwen-0.5b-hf")

add_to_catalog(format, "formats.models.llava_interleave", overwrite=True)
