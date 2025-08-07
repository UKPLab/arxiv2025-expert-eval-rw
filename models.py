from openai import AzureOpenAI
import openai
import utils
import re
from vllm import LLM, SamplingParams, sampling_params
from vllm.sampling_params import GuidedDecodingParams
import torch
import time
import json


class AzureModel:
    """
    Azure client object for OpenAI models
    """
    def __init__(self, endpoint, api_key, api_version, deployment_name, temperature):

        self.client = AzureOpenAI(azure_endpoint=endpoint, api_key=api_key, api_version=api_version)
        self.deployment_name = deployment_name
        self.temperature = temperature

    def __call__(self, system_prompt, user_prompt, response_format):
        """
        Inference call for the model in a chat completion style
        :param system_prompt: System prompt for the task
        :param user_prompt: Task specific input prompt including few-shot examples
        :param response_format: JSON Schema for evaluators, None for generators
        :return: Generated response and a dictionary including token counts and estimated cost
        """
        messages = [{'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt}]

        if self.deployment_name == 'o3-mini':
            # o3-mini does not support temperature adjustment
            try:
                completion = self.client.chat.completions.create(model=self.deployment_name,
                                                                 messages=messages,
                                                                 response_format=response_format)
            except openai.RateLimitError:
                # Retrying for rare cases of rate limit errors.
                time.sleep(1)
                completion = self.client.chat.completions.create(model=self.deployment_name,
                                                                 messages=messages,
                                                                 response_format=response_format)
        else:
            try:
                completion = self.client.chat.completions.create(model=self.deployment_name,
                                                                 messages=messages,
                                                                 temperature=self.temperature,
                                                                 response_format=response_format)
            except openai.RateLimitError:
                time.sleep(1)
                completion = self.client.chat.completions.create(model=self.deployment_name,
                                                                 messages=messages,
                                                                 temperature=self.temperature,
                                                                 response_format=response_format)

        if response_format is None:
            response = completion.choices[0].message.content
        else:
            try:
                # Trying whether response format followed.
                response = json.loads(completion.choices[0].message.content)
            except:
                response = {key: "" for key in response_format['json_schema']['schema']['required']}

        cost = {'prompt_tokens': completion.usage.prompt_tokens,
                'completion_tokens': completion.usage.completion_tokens,
                'total_cost': utils.calculate_cost(completion.model, completion.usage.prompt_tokens, completion.usage.completion_tokens)}

        return response, cost


class VLLModel:
    """
    vLLM model object for open local models
    """
    def __init__(self, deployment_name, temperature, context):

        self.deployment_name = deployment_name
        self.model = LLM(model=self.deployment_name, dtype=torch.bfloat16, trust_remote_code=True,
                         quantization="bitsandbytes", max_model_len=context, gpu_memory_utilization=0.8)
        self.sample_params = self.model.get_default_sampling_params()
        self.sample_params.max_tokens = 2048
        self.sample_params.temperature = temperature

    def __call__(self, system_prompt, user_prompt, response_format):
        """
        Inference call for the model in a chat completion style
        :param system_prompt: System prompt for the task
        :param user_prompt: Task specific input prompt including few-shot examples
        :param response_format: JSON Schema for evaluators, None for generators
        :return: Generated response and a dictionary including token counts and estimated cost
        """

        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]

        if response_format is not None:
            # OpenAI and vLLM do not use the exactly same format
            response_format = response_format['json_schema']['schema']
            self.sample_params.guided_decoding = GuidedDecodingParams(json=response_format)
        else:
            self.sample_params.guided_decoding = None

        completion = self.model.chat(messages, self.sample_params, use_tqdm=False)

        if response_format is None:
            # If model produces reasoning specific content, we are filtering it
            clean = re.sub(r'<think>.*?</think>', '', completion[0].outputs[0].text, flags=re.DOTALL)
            response = clean.strip()
        else:
            try:
                response = json.loads(completion[0].outputs[0].text)
            except:
                response = {key: "" for key in response_format['required']}

        cost = {'prompt_tokens': len(completion[0].prompt_token_ids),
                'completion_tokens': len(completion[0].outputs[0].token_ids),
                'total_cost': 0}

        return response, cost
