import json
import spacy
import re
import numpy as np
import random
from collections import Counter


nlp = spacy.load("en_core_sci_sm")


def read_json(path):
    """
    Reading a json file
    :param path: Directory of file to be read
    :return: json file
    """
    with open(path, 'r') as fr:
        file = json.load(fr)
    return file


def save(item, path):
    """
    Saving a json file
    :param item: Object to be saved
    :param path: Directory of file to be saved
    :return: json file
    """
    with open(path, 'w') as fw:
        json.dump(item, fw, indent=4)


def majority_voting(votes):
    """
    Majority voting for evaluation
    :param votes: List of evaluation verdicts
    :return: Majority voting result
    """
    counts = Counter(votes)
    max_count = max(counts.values())
    most_common_items = [item for item, freq in counts.items() if freq == max_count]

    if len(most_common_items) == 1:
        majority_vote = most_common_items[0]
    else:
        majority_vote = random.choice(most_common_items)

    return majority_vote

def split_sentences(text):
    """
    Splitting given text into sentences using spacy library
    :param text: String text to be split
    :return: List of string sentences
    """
    return [sent.text for sent in nlp(text).sents]


def calculate_cost(model_name, prompt_tokens, completion_tokens):
    """
    Calculating cost given model name and prompt tokens. Based on price changes this function needs to be updated.
    :param model_name: OpenAI model name only gpt-4o and o3-mini is used.
    :param prompt_tokens: Number of prompt tokens
    :param completion_tokens: Number of completion tokens
    :return: Calculated cost
    """
    if model_name == 'gpt-4o-2024-11-20':
        total_cost = prompt_tokens * 2.5 / 1000000 + completion_tokens * 10 / 1000000
    elif model_name == 'o3-mini-2025-01-31':
        total_cost = prompt_tokens * 1.1 / 1000000 + completion_tokens * 4.4 / 1000000
    else:
        raise KeyError(f"Unknown model name {model_name}")

    return total_cost


def set_paper_info_prompt(data):
    """
    Preparing prompt for paper input to generate related work drafts
    :param data: Data instance including main paper and cited paper information
    :return: Prompt string with abstract and introduction of main and cited papers
    """
    main_paper_prompt = f"MAIN PAPER TITLE: {data['metadata']['title']}\n" \
                        f"MAIN PAPER ABSTRACT: {data['abstract']['clean']}\n" \
                        f"MAIN PAPER INTRODUCTION: {data['introduction']['clean']}\n\n"

    cited_papers_prompt = '\n\n'.join([f"CITED PAPER {no+1} TITLE: {data['cited_papers_in_rw'][key]['title']}\n"
                                       f"CITED PAPER {no+1} ABSTRACT: {data['cited_papers_in_rw'][key]['abstract']}\n"
                                       f"CITED PAPER {no+1} INTRODUCTION: {data['cited_papers_in_rw'][key]['introduction']}"
                                       for no, key in enumerate(data['cited_papers_in_rw'])])

    return main_paper_prompt + cited_papers_prompt


def get_general_evaluation_schema():
    """
    Constructing structured output schema for coherence and contribution-positioning evaluation
    Includes binary evaluation along with reasoning
    :return: Dictionary of evaluation schema
    """
    schema = {
        "name": "Evaluation",
        "description": "Preference following evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                },
                "evaluation": {
                    "type": "string",
                    "enum": ["yes", "no"]
                }
            },
            "required": ["reasoning","evaluation"],
            "additionalProperties": False,
        }
    }
    return schema


def get_contribution_type_evaluation_schema():
    """
    Constructing structured output schema for contribution-positioning type evaluation
    Includes ternary evaluation along with reasoning
    :return: Dictionary of evaluation schema
    """
    schema = {
        "name": "Contribution-Type",
        "description": "Contribution type evaluation",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                },
                "evaluation": {
                    "type": "integer",
                    "enum": [1, 2, 3]
                }
            },
            "required": ["reasoning","evaluation"],
            "additionalProperties": False,
        }
    }

    return schema


def extract_citation_numbers(text):
    """
    Given citation sentence detect which citation numbers are used
    Adapted to different scenarios such as [1-3, 5, 7–9] -> 1,2,3,5,7,8,9
    :param text: Citation sentence
    :return: List of unique citation numbers in that sentence
    """
    citations = set()
    # To catch citation ranges with different dash types
    dash_chars = ['-', '‐', '‑', '‒', '–', '—', '―', '−']
    # Regular expression to catch citation marks with square brackets
    regex = r'\[([0-9,\s\u002D\u2010\u2011\u2012\u2013\u2014\u2015\u2212]+)\]'
    matches = re.findall(regex, text)

    for match in matches:
        for content in match.strip().split(','):
            content = content.replace(' ', '')
            if content.isdigit():
                # Single citations or multiple discrete citations
                citations.add(int(content))
            else:
                # Normalizing dashes in citation ranges
                for dash in dash_chars:
                    content = content.replace(dash, '-')

                range_parts = content.split('-')
                if len(range_parts) == 2:
                    start, end = map(int, range_parts)
                    for i in range(start, end+1):
                        citations.add(i)
                else:
                    raise ValueError(f"Invalid citation mark {content}")

    return sorted(list(citations))


def word_distribution_per_citation(text, cited_paper_count):
    """
    Given a related work section calculating how much content allocated per citation
    Counting the number of words in sentences for each respective citation
    :param text: Related work section
    :param cited_paper_count: Number of unique cited papers in gold related work section
    :return: Distribution of words per citation along with total number of words in the related work section
    """
    total_word_count = 0
    distribution = np.zeros(cited_paper_count)

    paragraphs = [paragraph for paragraph in text.split('\n') if paragraph != '']

    for paragraph in paragraphs:
        # If there is no citation in current sentence, we assume that those sentences are still referring previous citations
        current_papers = []
        sentences = split_sentences(paragraph)
        for sentence in sentences:
            cited_paper_ids = extract_citation_numbers(sentence)
            word_count = len(sentence.split())
            total_word_count += word_count

            if len(cited_paper_ids) > 0:
                current_papers = cited_paper_ids
            # Initial cases where no citations are mentioned yet
            if len(current_papers) > 0:
                for idx in current_papers:
                    if idx <= cited_paper_count:
                        distribution[idx-1] += word_count

    return distribution/total_word_count, total_word_count


def sentences_per_citation(text, cited_paper_count):
    """
    Given a related work section, identifying sentences in which the citations appear
    :param text: Related work section
    :param cited_paper_count: Number of unique cited papers in gold related work section
    :return: Dictionary where keys are citation numbers and values are list of sentences
    """
    sent_citation_dict = {i: [] for i in range(1, cited_paper_count+1)}

    paragraphs = text.split('\n')
    paragraphs = [paragraph for paragraph in paragraphs if paragraph != '']

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph)
        for sentence in sentences:
            cited_paper_ids = extract_citation_numbers(sentence)
            for cited_paper_id in cited_paper_ids:
                if cited_paper_id <= cited_paper_count:
                    sent_citation_dict[cited_paper_id].append(sentence)

    return sent_citation_dict


def random_map(keys, values, seed=0):

    base_count = len(keys) // len(values)
    remainder = len(keys) % len(values)
    value_dist = base_count * values

    random.seed(seed)
    value_dist += random.sample(values, remainder)
    random.seed(seed)
    random.shuffle(value_dist)

    return dict(zip(keys, value_dist))