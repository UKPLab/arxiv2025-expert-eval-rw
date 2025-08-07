import os
import utils
import random
import models
import argparse
from dotenv import load_dotenv
from collections import Counter
from tqdm import tqdm


def generate_contribution_type_data(model, dataset, prompts, target_number):
    """
    Generating synthetic positioning type data
    :param model: LLM model object
    :param dataset: Related work dataset
    :param prompts: Prompt dictionary
    :param target_number: Number of instances to be generated
    """
    eval_data = {}
    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}
    paper_ids = list(dataset.keys())
    values = list(prompts['contribution_type'].keys())

    if target_number > len(paper_ids):
        # Upsampling from the dataset
        # Assigning a positioning type for each paper
        # Avoid duplicates for the same papers and balancing the dataset labels
        initial_mapping = utils.random_map(paper_ids, values)
        new_mapping = {}
        duplicate_ids = random.sample(paper_ids, target_number-len(paper_ids))
        value_usage = Counter()

        for idx in duplicate_ids:
            prev_value = initial_mapping[idx]
            candidates = [v for v in values if v != prev_value]

            if not candidates:
                raise ValueError(f"No alternative value available for paper '{idx}'.")

            min_usage = min(value_usage[v] for v in candidates)
            least_used = [v for v in candidates if value_usage[v] == min_usage]
            chosen_value = random.choice(least_used)
            new_mapping[idx] = chosen_value
            value_usage[chosen_value] += 1

        final_mapping = list(initial_mapping.items()) + list(new_mapping.items())

    else:
        final_mapping = utils.random_map(random.sample(paper_ids, target_number), values)

    for i, items in enumerate(tqdm(final_mapping, total=len(final_mapping))):

        system_prompt = prompts['contribution_type'][items[1]]

        if items[1] in ["1", "2"]:
            # Rewriting the related work sections according to desired contribution type
            user_prompt = (f"ABSTRACT: {dataset[items[0]]['abstract']['clean']}\n\n"
                           f"INTRODUCTION: {dataset[items[0]]['introduction']['clean']}\n\n"
                           f"RELATED WORK: {dataset[items[0]]['related_work']['clean_numbered']}\n\n")
        else:
            # Rewriting the related work sections so that there is no positioning statement (negative example)
            user_prompt = f"RELATED WORK: {dataset[items[0]]['related_work']['clean_numbered']}\n\n"

        generation, cost = model(system_prompt=system_prompt, user_prompt=user_prompt, response_format=None)
        total_cost['prompt_tokens'] += cost['prompt_tokens']
        total_cost['completion_tokens'] += cost['completion_tokens']
        total_cost['total_cost'] += cost['total_cost']

        eval_data[i+1] = {'related_work': generation, 'paper_id':items[0], 'expected_type': items[1]}

    return eval_data, total_cost


def generate_contribution_check_data(contribution_type_data, target_number):
    """
    Generating synthetic contribution-positioning evaluation data
    :param contribution_type_data: Synthetically generated contribution type data
    :param target_number: Number of instances to be generated
    """
    eval_data = {}
    dist = [sum(x) for x in zip([target_number//4] * 4, ([1] * (target_number%4)) + ([0]*(4-(target_number%4))))]

    direct_eval_positives = []
    direct_eval_negatives = []
    pairwise_eval_positives = []
    pairwise_eval_negatives = []
    sampled_data = []

    for idx in contribution_type_data:
        # Based on the positioning type labeling paragraphs
        # Balancing the dataset with positive and negative examples for each type
        paragraphs = [paragraph for paragraph in contribution_type_data[idx]['related_work'].split('\n') if paragraph != '']
        if contribution_type_data[idx]['expected_type'] == '1':
            direct_eval_positives += [{'text': paragraph, 'contribution_type': '1', 'expected_result': 1} for paragraph in paragraphs]
        elif contribution_type_data[idx]['expected_type'] == '2':
            direct_eval_negatives += [{'text': paragraph, 'contribution_type': '1', 'expected_result': 0} for paragraph in paragraphs[:-1]] # Except final paragraph which includes contribution
            pairwise_eval_positives += [{'text': f"{paragraph}\n\n{paragraphs[-1]}", 'contribution_type': '2', 'expected_result': 1} for paragraph in paragraphs[:-1]]
        elif contribution_type_data[idx]['expected_type'] == '3':
            direct_eval_negatives += [{'text': paragraph, 'contribution_type': '1', 'expected_result': 0} for paragraph in paragraphs]
            pairwise_eval_negatives += [{'text': f"{paragraph}\n\n{paragraphs[-1]}", 'contribution_type': '2', 'expected_result': 0} for paragraph in paragraphs[:-1]]
        else:
            raise ValueError(f"Contribution type '{contribution_type_data[idx]['expected_type']}' not supported.")

    sampled_data += random.sample(direct_eval_positives, dist[0])
    sampled_data += random.sample(pairwise_eval_positives, dist[1])
    sampled_data += random.sample(direct_eval_negatives, dist[2])
    sampled_data += random.sample(pairwise_eval_negatives, dist[3])
    random.shuffle(sampled_data)

    for idx in range(target_number):
        eval_data[idx+1] = sampled_data[idx]

    return eval_data


def generate_coherence_data(dataset, target_number):
    """
    Generating synthetic contribution-positioning evaluation data
    :param dataset: Related work dataset
    :param target_number: Number of instances to be generated
    """
    eval_data = {}
    paper_ids = list(dataset.keys())
    dist = ([0]*(target_number//2)) + ([1]*(target_number - (target_number//2)))

    if target_number > len(paper_ids):
        paper_ids += random.sample(paper_ids, target_number-len(paper_ids))

    # Sampling citations sentences and balancing the dataset by
    # deliberately producing negative examples by changing paper contexts
    for idx, paper_id in enumerate(paper_ids):
        sentences_to_sample = utils.sentences_per_citation(dataset[paper_id]['related_work']['clean_numbered'], len(dataset[paper_id]['cited_papers_in_rw']))
        citation_number = random.sample(list(sentences_to_sample.keys()), 1)[0]

        sentence = random.sample(sentences_to_sample[citation_number], 1)[0]
        result = dist.pop(random.randrange(len(dist)))
        context = ''
        if result:
            context_number = citation_number
        else:
            # Removing the correct number and other possible citation numbers in the same sentence to guarantee negative samples
            removals = utils.extract_citation_numbers(sentence)
            sampling_pool = [i for i in range(1,len(dataset[paper_id]['cited_papers_in_rw'])+1) if i not in removals]
            context_number = random.sample(sampling_pool, 1)[0]

        for i, key in enumerate(dataset[paper_id]['cited_papers_in_rw']):
            if i+1 == context_number:
                context = (f"{dataset[paper_id]['cited_papers_in_rw'][key]['abstract']}\n"
                           f"{dataset[paper_id]['cited_papers_in_rw'][key]['introduction']}")
                break
        if context == '':
            raise ValueError('Context not found')

        eval_data[idx+1] = {'sentence': sentence, 'context': context, 'citation_number': citation_number, 'expected_result': result}

    return eval_data


def main(args):

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    load_dotenv(args.env_file)

    dataset = utils.read_json(args.dataset_file)
    prompts = utils.read_json(args.prompt_file)['eval_data']
    model = models.AzureModel(endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                              api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                              api_version=args.api_version,
                              deployment_name=args.deployment_name,
                              temperature=0.8)

    print(f"Generating Contribution Type Data...")
    contribution_type_data, contribution_type_cost = generate_contribution_type_data(model, dataset, prompts, args.target_number)
    utils.save(contribution_type_data, os.path.join(args.output_path, 'contribution_type_data.json'))
    utils.save(contribution_type_cost, os.path.join(args.output_path, 'contribution_type_data_cost.json'))

    print(f"Generating Contribution Check Data...")
    contribution_check_data = generate_contribution_check_data(contribution_type_data, args.target_number)
    utils.save(contribution_check_data, os.path.join(args.output_path, 'contribution_check_data.json'))

    print(f"Generating Coherence Data...")
    coherence_data = generate_coherence_data(dataset, args.target_number)
    utils.save(coherence_data, os.path.join(args.output_path, 'coherence_data.json'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_file', required=True, type=str)
    parser.add_argument('--deployment_name', default='gpt-4o', type=str)
    parser.add_argument('--api_version', default='2025-03-01-preview')
    parser.add_argument('--prompt_file', default='eval_exp_prompts.json', type=str)
    parser.add_argument('--dataset_file', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--target_number', default=50, type=int)

    arguments = parser.parse_args()
    main(arguments)

