"""Stand-in for vllm in CanItEdit's tests (78_llm_inference): its tests replace `LLM` with a mock,
but the code imports `LLM` and `SamplingParams` at the top, so only these names are needed."""


class SamplingParams:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class LLM:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("vllm stub: the tests are expected to mock LLM")
