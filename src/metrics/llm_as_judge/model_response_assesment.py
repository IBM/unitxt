dummy = 1

'''class LlmAsJudgeModel(ABC):
    """Abstract base class for LLmAsJudgeModel.

Args:
        model_id (str): The model id that will be used a judge.
        system_prompt (str): Optional. The system prompt that will be added to every api call.
    """

    def __init__(self, model_id: str, system_prompt: str = ""):
        self.model_id = model_id
        self.system_prompt = system_prompt

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int) -> str:
        """Abstract method for making generate evaluation response from the model.
        Subclasses must implement this method.

Args:
        evaluation_prompt (str): The evaluation prompt that will be sent to the language model.
        max_tokens (str): The max_token argument that will be used with the model.

Returns:
            Tuple[float, str]: A tuple containing the score as a float and the reasoning as a string.
        """
        pass


class MockLlmAsJudgeModel(LlmAsJudgeModel):
    def generate(self, prompt: str, max_tokens: int = 256) -> Tuple[int, str]:
        return (
            int(hash(prompt) % 10),
            f"MockLlmAsJudgeApiCall reasoning for system prompt {self.system_prompt}"
            f" and evaluation prompt {prompt}",
        )


class OpenAiLlmAsJudgeModel(LlmAsJudgeModel):
    def generate(self, evaluation_prompt: str, max_tokens: int = 256) -> str:
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        assert api_key is not None, (
            "Error while trying to run OpenAiLlmAsJudgeApiCall.generate. "
            "Please set the environment param 'OPENAI_API_KEY'."
        )
        openai.api_key = api_key

        MAX_API_RETRY = 5
        RETRY_INTERVAL_IN_SEC = 10

        last_error: str = ""
        for i in range(MAX_API_RETRY):
            try:
                response = openai.ChatCompletion.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": evaluation_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=max_tokens,
                )
                resp = response["choices"][0]["message"]["content"]  # type: ignore
                return resp

            except Exception as e:
                logger.error(
                    f"Failed to get response. Retrying after {RETRY_INTERVAL_IN_SEC} seconds."
                )
                logger.error(e)
                last_error = str(e)
                time.sleep(RETRY_INTERVAL_IN_SEC)

        logger.error(f"Failed after {MAX_API_RETRY} retries.")
        return f"Failed to retrieve an evaluation. Error: {last_error}"


class LlmAsJudge(BulkInstanceMetric):
    """LlamaIndex based metric class for evaluating correctness.

Attributes:
        reduction_map (dict): A dictionary specifying the reduction method for the metric.
        main_score (str): The main score used for evaluation.
        _requirements_list (List[str]): A list specifying any additional requirements for the metric.

Methods:
        prepare(self): Initialization method for the metric.
        compute(self, references, predictions, additional_inputs): Method to compute the metric.

    Usage:
        metric = LlamaIndexCorrectnessMetric()
        scores = metric.compute(references, prediction, additional_inputs)
    """

    model_id: str = ""
    model_type: Literal["ibm_bam", "openai", "huggingface"] = ""

    template: str = ""
    processing_card: str = ""

    _model_name_normalized = model_id.replace(".", "_").replace("-", "_")
    main_score: str = f"llm_as_judge-card_{processing_card}-template_{template}-model_id_{_model_name_normalized}"

    reduction_map: Dict[str, List[str]] = {"mean": [main_score]}

    _supported_openai_models: List[str] = ["gpt-3.5-turbo", "gpt-4"]
    _supported_ibm_bam_models: List[str] = []
    _mock_models: List[str] = ["mock"]

    _requirements_list: List[str] = ["openai", "genai", "transformers"]
    _llm: RemoteLlmAsJudgeApiCall

    def _assert_allow_passing_data_to_remote_api(self):
        assert settings.allow_passing_data_to_remote_api, (
            f"LlmAsJudge metric cannot run send data to remote APIs ({self.model_type}) when"
            f" unitxt.settings.allow_passing_data_to_remote_api=False."
            f" Set UNITXT_ALLOW_PASSING_DATA_TO_REMOTE_API environment variable, if you want to allow this. "
        )

    def prepare(self):
        """Initialization method for the metric. Initializes the CorrectnessEvaluator with the OpenAI model."""
        super().prepare()

        assert self.model_type in type(self.model_type)

        if self.model_type == "openai":
            assert self.model_id in self._supported_openai_models, (
                f"LlmAsJudge metric does not support {self.model_id} from type {self.model_type},"
                f" currently only the following {self.model_type} models are"
                f" supported: {self._supported_openai_models}."
            )
            self._assert_allow_passing_data_to_remote_api()

            self._llm = OpenAiLlmAsJudgeApiCall(
                model_name=self.model_id, api_key="", system_prompt=""
            )
        if self.model_type == "ibm_bam":
            assert self.model_id in self._supported_ibm_bam_models, (
                f"LlmAsJudge metric does not support {self.model_id} from type {self.model_type},"
                f" currently only the following {self.model_type} models are"
                f" supported: {self._supported_ibm_bam_models}."
            )
            self._assert_allow_passing_data_to_remote_api()
            raise NotImplementedError()
        elif self.model_type == "huggingface":
            raise NotImplementedError()
        else:
            raise NotImplementedError(
                f"LlmAsJudge metric does not support {self.model_name}, currently only {self._supported_external_api_models} are supported"
            )

        self.evaluator = CorrectnessEvaluator(
            llm=_llm, parser_function=self._custom_parser
        )

    def compute(
        self,
        references: List[str],
        prediction: str,
        task_data: Dict,
    ) -> Dict[str, Any]:
        """Method to compute the correctness metric.

Args:
            references (List[str]): List of reference instances.
            prediction (str): List of predicted instances.
            task_data (Dict): List of additional input data.

Returns:
            Dict[str, Any]: List of computed scores and feedback.

Raises:
            AssertionError: If the input does not meet the expected format.
        """
        # treat the references as the questions and the predictions as answers
        # assume a single reference

        query = task_data["question"]

        contexts = None
        if "contexts" in task_data:
            contexts = task_data["contexts"]

        per_reference_results = []
        for reference_response in references:
            per_reference_results.append(
                self.evaluator.evaluate(
                    query=query,
                    response=prediction,
                    contexts=contexts,
                    reference=reference_response,
                )
            )
        result = max([results.score for results in per_reference_results])

        return {
            self.main_score: result / 5,
            # "score_name": self.main_score,
            # "feedback": result.feedback, # removed since this cannot be tested
        }
'''
