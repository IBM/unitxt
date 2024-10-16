from enum import Enum

class DirectEvaluatorNameEnum(Enum):
    MIXTRAL = ('Mixtral', 'mixtral_8_7b', 'mistralai/mixtral-8x7b-instruct-v01')
    GRANITE = ('Granite 20b', 'granite_20b', 'ibm/granite-20b-code-instruct')
    LLAMA3_8B = ('Llama3-8b', 'llama3_8b', 'meta-llama/llama-3-8b-instruct')
    LLAMA3_70B = ('Llama3-70b', 'llama3_70b', 'meta-llama/llama-3-70b-instruct')
    PROMETHEUS = ('Prometheus', 'prometheus_8_7b', 'kaist-ai/prometheus-8x7b-v2')
    GPT4 = ("GPT-4o", "gpt_4o", 'gpt-4o-2024-05-13')

    def __new__(cls, value, json_name, model_id):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.json_name = json_name
        obj.model_id = model_id
        return obj

AvailableDirectEvaluators = [DirectEvaluatorNameEnum.MIXTRAL, DirectEvaluatorNameEnum.LLAMA3_8B,
                             DirectEvaluatorNameEnum.LLAMA3_70B, DirectEvaluatorNameEnum.PROMETHEUS, 
                             DirectEvaluatorNameEnum.GPT4]