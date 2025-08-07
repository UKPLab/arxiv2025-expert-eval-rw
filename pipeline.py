import argparse
import os
import utils
import eval_modules
import models
from tqdm import tqdm
from datetime import datetime
from dotenv import load_dotenv


def run_citation_eval(draft, paper_data):
    """
    Calling citation evaluation module
    :param draft: Generated related work draft
    :param paper_data: Data including cited paper information
    :return: Evaluation results
    """
    return eval_modules.citation_eval(draft, paper_data)


def run_coherence_eval(model, sys_prompt_eval, examples, draft, paper_data, sys_prompt_sum, majority):
    """
    Calling coherence evaluation module and deciding final result via majority voting
    :param model: LLM object
    :param sys_prompt_eval: System prompt for evaluation
    :param examples: Contrastive few-shot examples
    :param draft: Generated related work draft
    :param paper_data: Data including cited paper information
    :param sys_prompt_sum: System prompts for reasoning summary
    :param majority: Number of votes for majority
    :return: Evaluation results and cost information
    """
    
    # Running coherence evaluation for each citation for #majority times
    evals, eval_cost = eval_modules.coherence_eval(model, sys_prompt_eval, examples, draft, paper_data, majority)

    for cited_paper_id in evals:
        for sentence in evals[cited_paper_id]:
            if len(evals[cited_paper_id][sentence]['scores']) > 0:
                # Final evaluation decision by majority voting for each citation
                final_score = utils.majority_voting(evals[cited_paper_id][sentence]['scores'])

                user_prompt = f""
                for idx, (score, reason) in enumerate(zip(evals[cited_paper_id][sentence]['scores'], evals[cited_paper_id][sentence]['reasons'])):
                    if score == final_score:
                        user_prompt = f"{user_prompt}REASONING {idx+1}: {reason}\n\n"

                # Summary of reasoning for winner votes
                sum_reason, sum_cost = generate(model, sys_prompt_sum, user_prompt)

                evals[cited_paper_id][sentence]['final_score'] = final_score
                evals[cited_paper_id][sentence]['final_reason'] = sum_reason

                eval_cost['prompt_tokens'] += sum_cost['prompt_tokens']
                eval_cost['completion_tokens'] += sum_cost['completion_tokens']
                eval_cost['total_cost'] += sum_cost['total_cost']
            else:
                evals[cited_paper_id][sentence]['final_score'] = None
                evals[cited_paper_id][sentence]['final_reason'] = None

    return evals, eval_cost


def run_contribution_eval(model, sys_prompts_eval, examples, expected_type, draft, sys_prompt_sum, majority):
    """
    Calling contribution-positioning evaluation module and deciding final result via majority voting
    :param model: LLM object
    :param sys_prompts_eval: System prompts for evaluation
    :param examples: Contrastive few-shot examples
    :param expected_type: Expected positioning type
    :param draft: Generated related work draft
    :param sys_prompt_sum: System prompts for reasoning summary
    :param majority: Number of votes for majority
    :return: Evaluation results and cost information
    """

    evals = {'expected_type': int(expected_type)}
    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}

    # Running positioning type evaluation for #majority times
    type_eval, type_cost = eval_modules.contribution_type_eval(model, sys_prompts_eval, examples, draft, majority)
    total_cost['prompt_tokens'] += type_cost['prompt_tokens']
    total_cost['completion_tokens'] += type_cost['completion_tokens']
    total_cost['total_cost'] += type_cost['total_cost']

    if len(type_eval['evaluations']) > 0:
        # Final decided positioning type by majority voting
        final_type = utils.majority_voting(type_eval['evaluations'])
        
        # Summary of reasoning for winner votes
        user_prompt = f""
        for idx, (evaluation, reason) in enumerate(zip(type_eval['evaluations'], type_eval['reasons'])):
            if evaluation == final_type:
                user_prompt = f"{user_prompt}REASONING {idx+1}: {reason}\n\n"

        sum_reason, sum_cost = generate(model, sys_prompt_sum, user_prompt)
        total_cost['prompt_tokens'] += sum_cost['prompt_tokens']
        total_cost['completion_tokens'] += sum_cost['completion_tokens']
        total_cost['total_cost'] += sum_cost['total_cost']

        type_eval['final_type'] = final_type
        type_eval['final_reason'] = sum_reason
    else:
        type_eval['final_type'] = None
        type_eval['final_reason'] = None

    evals['type'] = type_eval
    if evals['type']['final_type'] in [1, 2]:
        # If positioning statement exist, we check paragraphs according to final type
        evals['check'], check_cost = eval_modules.contribution_check_eval(model, sys_prompts_eval, examples, draft, final_type, majority)

        total_cost['prompt_tokens'] += type_cost['prompt_tokens']
        total_cost['completion_tokens'] += type_cost['completion_tokens']
        total_cost['total_cost'] += type_cost['total_cost']

        for paragraph in evals['check']:
            if len(evals['check'][paragraph]['scores']) > 0:
                # Final evaluation decision by majority voting
                final_score = utils.majority_voting(evals['check'][paragraph]['scores'])

                # Summary of reasoning for winner votes
                user_prompt = f""
                for idx, (score, reason) in enumerate(zip(evals['check'][paragraph]['scores'], evals['check'][paragraph]['reasons'])):
                    if score == final_score:
                        user_prompt = f"{user_prompt}REASONING {idx+1}: {reason}\n\n"
                
                sum_reason, sum_cost = generate(model, sys_prompt_sum, user_prompt)

                evals['check'][paragraph]['final_score'] = final_score
                evals['check'][paragraph]['final_reason'] = sum_reason

                total_cost['prompt_tokens'] += sum_cost['prompt_tokens']
                total_cost['completion_tokens'] += sum_cost['completion_tokens']
                total_cost['total_cost'] += sum_cost['total_cost']

            else:
                evals['check'][paragraph]['final_score'] = None
                evals['check'][paragraph]['final_reason'] = None

        return evals, total_cost

    else:
        return evals, total_cost


def aggregate(citation_eval, coherence_eval, contribution_eval, expected_type):
    """
    Gathering all evaluation results into single report
    :param citation_eval: Citation check evaluation results
    :param coherence_eval: Coherence evaluation results
    :param contribution_eval: Contribution-positioning evaluation results
    :param expected_type: Expected positioning type
    :return: Evaluation report (string)
    """

    cont_type_dict = {"1": "In each paragraph, the contributions of the main paper or its position in the existing literature are highlighted in relation to the papers and topics addressed in that paragraph.",
                      "2": "In the final paragraph, the contributions of the main paper or its position in the existing literature are highlighted. While highlighting, the topics addressed in each previous paragraph are referred.",
                      "3": "No contribution statement"}

    report = "<START OF DRAFT REPORT>"

    if len(citation_eval['missing_papers']) == 0:
        report = f"{report}\n\n\n## MISSING CITATIONS\nNone"
    else:
        report = f"{report}\n\n\n## MISSING CITATIONS\n{citation_eval['missing_papers']}"

    if len(citation_eval['hallucinated_papers']) == 0:
        report = f"{report}\n\n## HALLUCINATED CITATIONS\nNone"
    else:
        report = f"{report}\n\n## HALLUCINATED CITATIONS\n{citation_eval['hallucinated_papers']}"

    if citation_eval['total_length'][0] in ['Too short', 'Too long']:
        report = f"{report}\n\n## SECTION LENGTH\n{citation_eval['total_length'][0]} {citation_eval['total_length'][1]}"

    report = f"{report}\n\n## CITATION EMPHASIS EVALUATION"

    for i in citation_eval['citation_emphasis']:
        if citation_eval['citation_emphasis'][i] in ['Excessive emphasis', 'Insufficient emphasis']:
            report = f"{report}\n* CITED PAPER {i}: {citation_eval['citation_emphasis'][i]}"

    report = f"{report}\n\n## COHERENCE PROBLEMATIC SENTENCES"
    coherence_flag = True
    for cited_paper_id in coherence_eval:
        for sentence in coherence_eval[cited_paper_id]:
            if coherence_eval[cited_paper_id][sentence]['final_score'] == 0:
                coherence_flag = False
                report = f"{report}\n\n* CITED PAPER: {cited_paper_id}\n SENTENCE: {sentence}\n " \
                         f"REASONING: {coherence_eval[cited_paper_id][sentence]['final_reason']}"

    if coherence_flag:
        report = f"{report}\nNone"

    report = f"{report}\n\n## CONTRIBUTION - POSITIONING EVALUATION"

    report = f"{report}\n\n### Type"

    report = f"{report}\n* Intended Contribution - Positioning Type: {cont_type_dict[expected_type]}"

    report = f"{report}\n* Draft's Contribution - Positioning Type: {cont_type_dict[str(contribution_eval['type']['final_type'])]}"

    report = f"{report}\n* Draft's Contribution - Positioning Type Reasoning: {contribution_eval['type']['final_reason']}"

    if expected_type == str(contribution_eval['type']['final_type']):
        report = f"{report}\n* Type Result: Matched"
    else:
        report = f"{report}\n* Type Result: Mismatched"

    if contribution_eval['type']['final_type'] != 3:
        report = f"{report}\n\n### Draft Contribution - Positioning Problematic Paragraphs (In case the type matches)"

        contribution_flag = True
        for paragraph in contribution_eval['check']:
            if contribution_eval['check'][paragraph]['final_score'] == 0:
                contribution_flag = False
                report = f"{report}\n\n* {paragraph}\nREASONING: {contribution_eval['check'][paragraph]['final_reason']}"

        if contribution_flag:
            report = f"{report}\nNone"

    report = f"{report}\n\n\n<END OF DRAFT REPORT>"

    return report


def generate(model, system_prompt, user_prompt):
    """
    Calling LLM object for inference
    :param model: LLM object
    :param system_prompt: System prompt for inference
    :param user_prompt: Other required inputs for the task
    """
    generation, cost = model(system_prompt=system_prompt, user_prompt=user_prompt, response_format=None)
    return generation, cost


def run_pipeline(generator_model, coh_eval_model, cont_eval_model, config, dataset):
    """
    Running evaluation an generation pipeline for the dataset
    :param generator_model: Generator model object
    :param coh_eval_model: Coherence evaluation model object
    :param cont_eval_model: Contribution-positioning evaluation model object
    :param config: Experiment configuration dictionary
    :param dataset: Related work dataset
    """
    cont_dist = utils.random_map(list(dataset.keys()), ['1', '2'])
    change_dict = {'1': '2', '2': '1'}

    # Main pipeline loop
    for main_paper_id in (pbar := tqdm(dataset, total=len(dataset))):
        pbar.set_description(f"Processing {main_paper_id}")

        # Evaluation records based on iterations
        record = {i: {} for i in range(1,config['num_iterations']+1)}
        cost = {'individual': {i: {} for i in range(1,config['num_iterations']+1)},
                'total': 0}

        for i in range(1, config['num_iterations']+1):

            print(f"Paper: {main_paper_id} Iteration {i}/{config['num_iterations']} Generating draft...")

            if i == 1:
                # Prompt adjustment for the first draft
                # If new paper introduction experiment, making 25% of the papers
                if config['add_new_paper']:
                    active_data = dataset[main_paper_id].copy()
                    paper_limit = len(dataset[main_paper_id]['cited_papers_in_rw']) - round(len(dataset[main_paper_id]['cited_papers_in_rw'])/4)

                    active_data['cited_papers_in_rw'] = {key: dataset[main_paper_id]['cited_papers_in_rw'][key]
                                                         for no, key in enumerate(dataset[main_paper_id]['cited_papers_in_rw']) if (no + 1) <= paper_limit}
                else:
                    active_data = dataset[main_paper_id].copy()

                expected_type = cont_dist[main_paper_id]
                cont_type_inst = config['prompts']['contribution']['instruction'][cont_dist[main_paper_id]]
                draft_sys_prompt = config['prompts']['first_draft']['system_prompt'].format(contribution=cont_type_inst)
                draft_user_prompt = utils.set_paper_info_prompt(active_data)

            else:
                # prompt adjustment for the next drafts with feedback
                # Based on the experiment type, new papers are added and style change applied.

                if config['add_new_paper'] and i > config['num_iterations']/2:
                    active_data = dataset[main_paper_id].copy()

                if config['style_change'] and i > config['num_iterations']/2:
                    cont_type_inst = config['prompts']['contribution']['instruction'][change_dict[cont_dist[main_paper_id]]]
                    expected_type = change_dict[cont_dist[main_paper_id]]

                if not config['report_feedback']:
                    feedback = record[i-1]['feedback']
                else:
                    feedback = record[i-1]['eval_report']

                draft_sys_prompt = config['prompts']['next_draft']['system_prompt'].format(contribution=cont_type_inst)
                draft_user_prompt = f"{utils.set_paper_info_prompt(active_data)}\n\n" \
                                    f"PREVIOUS DRAFT: {record[i-1]['draft']}\n\n" \
                                    f"FEEDBACK: {feedback}"

            # Generating draft
            record[i]['draft'], cost['individual'][i]['generation_cost'] = generate(model=generator_model,
                                                                                    system_prompt=draft_sys_prompt,
                                                                                    user_prompt=draft_user_prompt)
            cost['total'] += cost['individual'][i]['generation_cost']['total_cost']

            # Citation checking evaluation
            print(f"Paper: {main_paper_id} Iteration {i}/{config['num_iterations']} Citation check...")
            record[i]['citation_eval'] = run_citation_eval(draft=record[i]['draft'], paper_data=active_data)

            # Coherence evaluation
            print(f"Paper: {main_paper_id} Iteration {i}/{config['num_iterations']} Coherence check...")
            record[i]['coherence_eval'], cost['individual'][i]['coherence_cost'] = run_coherence_eval(model=coh_eval_model,
                                                                                                      sys_prompt_eval=config['prompts']['coherence']['system_prompt'],
                                                                                                      examples=config['prompts']['coherence']['example'],
                                                                                                      draft=record[i]['draft'],
                                                                                                      paper_data=active_data,
                                                                                                      sys_prompt_sum = config['prompts']['summary']['system_prompt'],
                                                                                                      majority=config['majority_vote'])
            cost['total'] += cost['individual'][i]['coherence_cost']['total_cost']

            # Positioning evalations
            print(f"Paper: {main_paper_id} Iteration {i}/{config['num_iterations']} Contribution check...")
            record[i]['contribution_eval'], cost['individual'][i]['contribution_cost'] =  run_contribution_eval(model_type=cont_eval_model,
                                                                                                                sys_prompts_eval=config['prompts']['contribution']['system_prompts'],
                                                                                                                examples=config['prompts']['contribution']['examples'],
                                                                                                                expected_type=expected_type,
                                                                                                                draft=record[i]['draft'],
                                                                                                                sys_prompt_sum=config['prompts']['summary']['system_prompt'],
                                                                                                                majority=config['majority_vote'])
            cost['total'] += cost['individual'][i]['contribution_cost']['total_cost']

            # Generating evaluation report
            record[i]['eval_report'] = aggregate(citation_eval=record[i]['citation_eval'],
                                                 coherence_eval=record[i]['coherence_eval'],
                                                 contribution_eval=record[i]['contribution_eval'],
                                                 expected_type=expected_type)

            # Generating feedback
            if not config['report_feedback']:
                print(f"Paper: {main_paper_id} Iteration {i}/{config['num_iterations']} Generating feedback...")
                record[i]['feedback'], cost['individual'][i]['feedback_cost'] = generate(model=generator_model,
                                                                                         system_prompt=config['prompts']['feedback']['system_prompt'],
                                                                                         user_prompt=record[i]['eval_report'])
                cost['total'] += cost['individual'][i]['feedback_cost']['total_cost']

        # Save the completed papers so far to avoid waste sources in case of API problems
        utils.save(record, os.path.join(config['output_path'], f"records/{main_paper_id}.json"))
        utils.save(cost, os.path.join(config['output_path'], f"costs/{main_paper_id}.json"))


def main(args):

    if os.path.exists(os.path.join(args.output_path, 'config.json')):

        # If a config file already exists, already finished papers exluded from pipeline
        config = utils.read_json(os.path.join(args.output_path, 'config.json'))

        dataset = utils.read_json(config['dataset_file'])
        processed_papers_ids = [file.removesuffix('.json') for file in os.listdir(os.path.join(config['output_path'], 'records'))]
        dataset = {key: dataset[key] for key in dataset if key not in processed_papers_ids}

    else:
        # It there is no config file in the specified output path,
        # it will be created with experiment parameters
        config = args.__dict__.copy()
        config.pop('data_count')
        config['output_path'] = os.path.join(config['output_path'], f"{config['deployment_name']}--{config['exp_name']}-{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        os.makedirs(os.path.join(config['output_path'], 'records'))
        os.makedirs(os.path.join(config['output_path'], 'costs'))

        dataset = utils.read_json(config['dataset_file'])
        config['prompts'] = utils.read_json(config['prompt_file'])
        utils.save(config, os.path.join(config['output_path'], 'config.json'))

    if args.data_count is not None:
        if args.data_count < len(dataset):
            sub_dataset = {}
            for i, key in zip(range(args.data_count), dataset):
                sub_dataset[key] = dataset[key]
        else:
            sub_dataset = dataset
    else:
        sub_dataset = dataset

    load_dotenv(config['env_file'])

    # Setting main generator model
    if config['deployment_name'] in ['gpt-4o', 'o3-mini']:
        generator_model = models.AzureModel(endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                            api_version=config['api_version'],
                                            deployment_name=config['deployment_name'],
                                            temperature=config['temperature'])


    elif config['deployment_name'] in ['meta-llama/Llama-3.3-70B-Instruct', 'google/gemma-3-27b-it']:
        generator_model = models.VLLModel(deployment_name=config['deployment_name'],
                                          temperature=config['temperature'],
                                          context=65536)

    else:
        raise ValueError(f"Deployment name {config['deployment_name']} not supported.")

    # Setting coherence evaluation model as gpt-4o based on the preliminary evaluation results
    coh_eval_model = models.AzureModel(endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                        api_version=config['api_version'],
                                        deployment_name='gpt-4o',
                                        temperature=config['temperature'])

    # Setting contribution-positioning evaluation model as o3-mini based on  the preliminary evaluation results
    cont_eval_model = models.AzureModel(endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                                        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                                        api_version=config['api_version'],
                                        deployment_name='o3-mini',
                                        temperature=config['temperature'])

    # Starting pipeline loop
    run_pipeline(generator_model, coh_eval_model, cont_eval_model, config, sub_dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', reqired=True, type=str)
    parser.add_argument('--env_file', reqired=True, type=str)
    parser.add_argument('--deployment_name', reqired=True, type=str)
    parser.add_argument('--api_version', default='2025-03-01-preview', type=str)
    parser.add_argument('--prompt_file', default='prompts.json', type=str)
    parser.add_argument('--dataset_file', required=True, type=str)
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--data_count', default=44, type=int)
    parser.add_argument('--num_iterations', default=5, type=int)
    parser.add_argument('--majority_vote', default=3, type=int)
    parser.add_argument('--temperature', default=0.8)
    parser.add_argument('--add_new_paper', action='store_true')
    parser.add_argument('--style_change', action='store_true')

    arguments = parser.parse_args()
    main(arguments)