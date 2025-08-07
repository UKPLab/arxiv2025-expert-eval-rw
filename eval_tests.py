import argparse
import utils
import os
import json
import eval_modules
import models
from tqdm import tqdm
from dotenv import load_dotenv
from sklearn.metrics import classification_report


def run_contribution_type_eval(model, prompts, exp_type, data, majority, output_path):

    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}
    evals = {'result': {}, 'answers': {'gold': [], 'pred': []}, 'evals': {}}

    if exp_type == 'zero-shot':
        sys_prompts = prompts["system_prompts_zs"]
        examples = None
    elif exp_type == 'few-shot':
        sys_prompts = prompts["system_prompts_fs"]
        examples = prompts["examples"]
    else:
        raise ValueError('exp_type must be either zero-shot or few-shot')

    for idx in tqdm(data, total=len(data)):
        evals['evals'][idx], type_cost = eval_modules.contribution_type_eval(model, sys_prompts, examples, data[idx]['related_work'], majority)
        total_cost['prompt_tokens'] += type_cost['prompt_tokens']
        total_cost['completion_tokens'] += type_cost['completion_tokens']
        total_cost['total_cost'] += type_cost['total_cost']

        if len(evals['evals'][idx]['evaluations']) > 0:
            final_type = utils.majority_voting(evals['evals'][idx]['evaluations'])
        else:
            final_type = -1

        evals['evals'][idx]['final_type'] = final_type
        evals['answers']['gold'].append(int(data[idx]['expected_type']))
        evals['answers']['pred'].append(final_type)

    evals['result'] = classification_report(evals['answers']['gold'], evals['answers']['pred'], output_dict=True)

    utils.save(evals, os.path.join(output_path, 'contribution_type_result.json'))
    utils.save(total_cost, os.path.join(output_path, 'contribution_type_cost.json'))


def run_contribution_check_eval(model, prompts, exp_type, data, majority, output_path):

    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}
    evals = {'result': {}, 'answers': {'gold': [], 'pred': []}, 'evals': {}}

    if exp_type == 'zero-shot':
        sys_prompts = prompts["system_prompts_zs"]
        examples = None
    elif exp_type == 'few-shot':
        sys_prompts = prompts["system_prompts_fs"]
        examples = prompts["examples"]
    else:
        raise ValueError('exp_type must be either zero-shot or few-shot')

    for idx in tqdm(data, total=len(data)):
        raw_eval, check_cost = eval_modules.contribution_check_eval(model, sys_prompts, examples, data[idx]['text'], int(data[idx]['contribution_type']), majority)
        total_cost['prompt_tokens'] += check_cost['prompt_tokens']
        total_cost['completion_tokens'] += check_cost['completion_tokens']
        total_cost['total_cost'] += check_cost['total_cost']

        if len(raw_eval) != 1:
            raise ValueError(f"Problem of paragraph count in data instance {raw_eval}")
        else:
            key = list(raw_eval.keys())[0]
            evals['evals'][idx] = raw_eval[key]

            if len(evals['evals'][idx]['scores']) > 0:
                final_score = utils.majority_voting(evals['evals'][idx]['scores'])
            else:
                final_score = -1

            evals['evals'][idx]['final_score'] = final_score
            evals['answers']['gold'].append(int(data[idx]['expected_result']))
            evals['answers']['pred'].append(final_score)

    evals['result'] = classification_report(evals['answers']['gold'], evals['answers']['pred'], output_dict=True)

    utils.save(evals, os.path.join(output_path, 'contribution_check_result.json'))
    utils.save(total_cost, os.path.join(output_path, 'contribution_check_cost.json'))


def run_coherence_eval(model, prompts, exp_type, data, majority, output_path):

    score_map = {'yes': 1, 'no': 0}
    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}
    evals = {'result': {}, 'answers': {'gold': [], 'pred': []}, 'evals': {}}

    for idx in tqdm(data, total=len(data)):
        evals['evals'][idx] = {'scores': [], 'reasons': []}

        if exp_type == 'zero-shot':
            sys_prompt = prompts['system_prompt_zs']
            user_prompt = f"PAPER CONTEXT: {data[idx]['context']}\n\nCITATION SENTENCE: {data[idx]['sentence']}\n\n" \
                          f"CITED PAPER {data[idx]['citation_number']}"
        elif exp_type == 'few-shot':
            sys_prompt = prompts['system_prompt_fs']
            user_prompt = f"{prompts['example']}\n\nPAPER CONTEXT: {data[idx]['context']}\n\nCITATION SENTENCE: {data[idx]['sentence']}\n\n" \
                          f"CITED PAPER {data[idx]['citation_number']}"
        else:
            raise ValueError('exp_type must be either zero-shot or few-shot')

        for turn in range(majority):

            raw_eval, cost = model(system_prompt=sys_prompt,
                                   user_prompt=user_prompt,
                                   response_format={"type": "json_schema", "json_schema": utils.get_general_evaluation_schema()})

            total_cost['prompt_tokens'] += cost['prompt_tokens']
            total_cost['completion_tokens'] += cost['completion_tokens']
            total_cost['total_cost'] += cost['total_cost']

            if raw_eval['evaluation'] in ['yes', 'no']:
                evals['evals'][idx]['scores'].append(score_map[raw_eval['evaluation']])
                evals['evals'][idx]['reasons'].append(raw_eval['reasoning'])


        if len(evals['evals'][idx]['scores']) > 0:
            final_score = utils.majority_voting(evals['evals'][idx]['scores'])
        else:
            final_score = -1

        evals['evals'][idx]['final_score'] = final_score

        evals['answers']['gold'].append(data[idx]['expected_result'])
        evals['answers']['pred'].append(final_score)

    evals['result'] = classification_report(evals['answers']['gold'], evals['answers']['pred'], output_dict=True)

    utils.save(evals, os.path.join(output_path, 'coherence_result.json'))
    utils.save(total_cost, os.path.join(output_path, 'coherence_cost.json'))


def main(args):

    output_path = os.path.join(args.output_path, args.deployment_name, args.exp_type)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    load_dotenv(args.env_file)

    contribution_type_data = utils.read_json(os.path.join(args.dataset_path,'contribution_type_data.json'))
    contribution_check_data = utils.read_json(os.path.join(args.dataset_path, 'contribution_check_data.json'))
    contribution_coherence_data = utils.read_json(os.path.join(args.dataset_path, 'coherence_data.json'))

    prompts = utils.read_json(args.prompt_file)['eval_test']

    if args.deployment_name in ['gpt-4o', 'o3-mini']:
        model = models.AzureModel(endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                  api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                  api_version=args.api_version,
                                  deployment_name=args.deployment_name,
                                  temperature=args.temperature)

    elif args.deployment_name in ['meta-llama/Llama-3.3-70B-Instruct', 'google/gemma-3-27b-it']:
        model = models.VLLModel(deployment_name=args.deployment_name,
                                temperature=args.temperature,
                                context=65536)

    else:
        raise ValueError(f"Deployment name {args.deployment_name} not supported.")

    print('Evaluating contribution type...')
    run_contribution_type_eval(model=model,
                               prompts=prompts['contribution'],
                               exp_type = args.exp_type,
                               data=contribution_type_data,
                               majority=args.majority_vote,
                               output_path=output_path)

    print('Evaluating contribution checks...')
    run_contribution_check_eval(model=model,
                                prompts=prompts['contribution'],
                                exp_type=args.exp_type,
                                data=contribution_check_data,
                                majority=args.majority_vote,
                                output_path=output_path)

    print('Evaluating coherence...')
    run_coherence_eval(model=model,
                       prompts=prompts['coherence'],
                       exp_type=args.exp_type,
                       data=contribution_coherence_data,
                       majority=args.majority_vote,
                       output_path=output_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', required=True, type=str)
    parser.add_argument('--deployment_name', required=True, type=str)
    parser.add_argument('--api_version', default='2025-03-01-preview')
    parser.add_argument('--prompt_file', default='eval_exp_prompts.json', type=str)
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--majority_vote', default=3, type=int)
    parser.add_argument('--temperature', default=0.8)
    parser.add_argument('--exp_type', choices=['zero-shot','few-shot'], default='few-shot')


    arguments = parser.parse_args()
    main(arguments)