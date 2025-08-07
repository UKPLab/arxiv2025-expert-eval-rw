import utils


def citation_eval(draft, paper_data):
    """
    Citation evaluation. Regex based evaluation.
    Hallucinated and missing paper ratio.
    Length
    Citation emphasis
    :param draft: Generated related work draft
    :param paper_data: Data including cited paper information along with gold related work draft
    :return: Dictionary of evaluation results
    """
    evals = {}
    # Determining number of unique citation counts for both gold and generated related work
    gold_cited_paper_count = len(paper_data['cited_papers_in_rw'])
    draft_cited_papers = utils.extract_citation_numbers(draft)

    evals['hallucinated_papers'] = []

    for idx in draft_cited_papers:
        if idx > gold_cited_paper_count:
            evals['hallucinated_papers'].append(idx)

    evals['missing_papers'] = sorted(list(set(range(1,gold_cited_paper_count+1)) - set(draft_cited_papers)))

    draft_distribution, draft_word_count = utils.word_distribution_per_citation(draft, gold_cited_paper_count)
    gold_distribution, gold_word_count = utils.word_distribution_per_citation(paper_data['related_work']['clean_numbered'], gold_cited_paper_count)

    # Checking whether total length falls in 25% tolerance interval
    if (gold_word_count > draft_word_count) and ((gold_word_count - draft_word_count) / gold_word_count > 0.25):
        evals['total_length'] = ['Too short', f"Increase the content by {5*round((gold_word_count - draft_word_count)/draft_word_count * 100/5)}%"]
    elif (gold_word_count < draft_word_count) and ((draft_word_count - gold_word_count) / gold_word_count > 0.25):
        evals['total_length'] = ['Too long', f"Reduce the content by {5*round((draft_word_count - gold_word_count)/draft_word_count * 100/5)}%"]
    else:
        evals['total_length'] = 'Adequate emphasis'

    evals['citation_emphasis'] = {}

    # Checking whether each citation emphasis falls in 25% tolerance interval
    for i, (draft_value, gold_value) in enumerate(zip(draft_distribution, gold_distribution)):
        if (gold_value > draft_value) and ((gold_value - draft_value) / gold_value > 0.25):
            evals['citation_emphasis'][i+1] = 'Insufficient emphasis'
        elif (gold_value < draft_value) and ((draft_value - gold_value) / gold_value > 0.25):
            evals['citation_emphasis'][i+1] = 'Excessive emphasis'
        else:
            evals['citation_emphasis'][i+1] = 'Adequate emphasis'

    return evals


def coherence_eval(model, system_prompt, examples, draft, paper_data, turns):
    """
    Coherence evaluation of citation sentences. LLM-based evaluation.
    :param model: LLM model object
    :param system_prompt: System prompt for coherence evaluation
    :param examples: Contrastive few-shot examples
    :param draft: Generated related work draft
    :param paper_data: Data including cited paper information
    :param turns: Number of evaluation turns to be used in majority voting
    :return: Dictionary of evaluation results and token numbers with cost
    """

    # For each cited paper, determining citation sentences they belong
    sentences_to_check = utils.sentences_per_citation(draft, len(paper_data['cited_papers_in_rw']))

    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}

    evals = {}
    score_map = {'yes': 1, 'no': 0}

    for i, key in enumerate(paper_data['cited_papers_in_rw']):
        evals[i+1] = {}
        for sentence in sentences_to_check[i+1]:
            # For each sentence we record each score and reasoning
            evals[i+1][sentence] = {'scores': [], 'reasons': []}


            # Checking whether citation sentences grounded in cited paper's abstract and information
            if examples is None:
                user_prompt = f"PAPER CONTEXT: {paper_data['cited_papers_in_rw'][key]['abstract']}\n" \
                              f"{paper_data['cited_papers_in_rw'][key]['introduction']}\n\n" \
                              f"CITATION SENTENCE: {sentence}\n\nCITED PAPER {i+1}"
            else:
                user_prompt = f"{examples}\n\nPAPER CONTEXT: {paper_data['cited_papers_in_rw'][key]['abstract']}\n" \
                              f"{paper_data['cited_papers_in_rw'][key]['introduction']}\n\n" \
                              f"CITATION SENTENCE: {sentence}\n\nCITED PAPER {i+1}"

            # Repeating evaluation for specified number of turns
            for turn in range(turns):
                raw_eval, cost = model(system_prompt=system_prompt,
                                       user_prompt=user_prompt,
                                       response_format={"type": "json_schema", "json_schema": utils.get_general_evaluation_schema()})

                total_cost['prompt_tokens'] += cost['prompt_tokens']
                total_cost['completion_tokens'] += cost['completion_tokens']
                total_cost['total_cost'] += cost['total_cost']

                if raw_eval['evaluation'] in ['yes', 'no']:
                    evals[i+1][sentence]['scores'].append(score_map[raw_eval['evaluation']])
                    evals[i+1][sentence]['reasons'].append(raw_eval['reasoning'])

    return evals, total_cost


def contribution_type_eval(model, system_prompts, examples, draft, turns):
    """
    Positioning type evaluation of related work draft. LLM-based evaluation.
    :param model: LLM model object
    :param system_prompts: System prompt for positioning type evaluation
    :param examples: Contrastive few-shot examples
    :param draft: Generated related work draft
    :param turns: Number of evaluation turns to be used in majority voting
    :return: Dictionary of evaluation results and token numbers with cost
    """

    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}
    evals = {'evaluations': [], 'reasons': []}

    if examples is None:
        user_prompt = f"DRAFT: {draft}\n\n"
    else:
        user_prompt = f"{examples['contribution_type']}\n\nDRAFT: {draft}\n\n"

    # Classifying position statement type in the generated draft
    # Repeating evaluation for specified number of turns
    for turn in range(turns):
        raw_eval, cost = model(system_prompt=system_prompts['contribution_type'],
                               user_prompt=user_prompt,
                               response_format={"type": "json_schema", "json_schema": utils.get_contribution_type_evaluation_schema()})

        total_cost['prompt_tokens'] += cost['prompt_tokens']
        total_cost['completion_tokens'] += cost['completion_tokens']
        total_cost['total_cost'] += cost['total_cost']

        if raw_eval['evaluation'] in [1, 2, 3]:
            evals['evaluations'].append(raw_eval['evaluation'])
            evals['reasons'].append(raw_eval['reasoning'])


    return evals, total_cost

def contribution_check_eval(model, system_prompts, examples, draft, type, turns):
    """
    Contribution-positioning evaluation of related work paragraphs. LLM-based evaluation.
    :param model: LLM model object
    :param system_prompts: System prompt for positioning type evaluation
    :param examples: Contrastive few-shot examples
    :param draft: Generated related work draft
    :param type: Positioning type
    :param turns: Number of evaluation turns to be used in majority voting
    :return: Dictionary of evaluation results and token numbers with cost
    """

    evals = {}
    total_cost = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_cost': 0}
    score_map = {'yes': 1, 'no': 0}

    if type == 1:
        # For the 1st type positioning we check whether each paragraph includes contribution-positioning statement
        paragraphs = [paragraph for paragraph in draft.split('\n') if paragraph != '']
        for paragraph in paragraphs:
            evals[paragraph] = {'scores': [], 'reasons': []}

            if examples is None:
                user_prompt = f"DRAFT: {paragraph}\n\n"
            else:
                user_prompt = f"{examples['direct_eval']}\n\nDRAFT: {paragraph}\n\n"

            for turn in range(turns):
                raw_eval, cost = model(system_prompt=system_prompts['direct_eval'],
                                       user_prompt=user_prompt,
                                       response_format={"type": "json_schema", "json_schema": utils.get_general_evaluation_schema()})

                total_cost['prompt_tokens'] += cost['prompt_tokens']
                total_cost['completion_tokens'] += cost['completion_tokens']
                total_cost['total_cost'] += cost['total_cost']

                if raw_eval['evaluation'] in ['yes', 'no']:
                    evals[paragraph]['scores'].append(score_map[raw_eval['evaluation']])
                    evals[paragraph]['reasons'].append(raw_eval['reasoning'])

    elif type == 2:
        # For the 2nd type positioning we check whether final paragraph refers previous paragraphs while positioning the paper
        paragraphs = [paragraph for paragraph in draft.split('\n') if paragraph != '']
        final_paragraph = paragraphs[-1]

        for paragraph in paragraphs[:-1]:
            key = f"CONTEXT PARAGRAPH: {paragraph}\nFINAL PARAGRAPH: {final_paragraph}"
            evals[key] = {'scores': [], 'reasons': []}

            if examples is None:
                user_prompt = f"CONTEXT: {paragraph}\nFINAL: {final_paragraph}\n\n"
            else:
                user_prompt = f"{examples['pairwise_eval']}\n\nCONTEXT: {paragraph}\nFINAL: {final_paragraph}\n\n"

            for turn in range(turns):
                raw_eval, cost = model(system_prompt=system_prompts['pairwise_eval'],
                                       user_prompt=user_prompt,
                                       response_format={"type": "json_schema", "json_schema": utils.get_general_evaluation_schema()})

                total_cost['prompt_tokens'] += cost['prompt_tokens']
                total_cost['completion_tokens'] += cost['completion_tokens']
                total_cost['total_cost'] += cost['total_cost']

                if raw_eval['evaluation'] in ['yes', 'no']:
                    evals[key]['scores'].append(score_map[raw_eval['evaluation']])
                    evals[key]['reasons'].append(f"REASONING: {raw_eval['reasoning']}")

    return evals, total_cost
