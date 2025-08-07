import argparse
import os
import utils


def main(args):

    config = utils.read_json(os.path.join(args.output_path, 'config.json'))

    record_files = sorted(os.listdir(os.path.join(args.output_path, 'records')))

    num_iterations = config['num_iterations']

    results = {'hallucinated_paper_ratio': [[0]*num_iterations for i in range(len(record_files))],
               'missing_paper_ratio': [[0]*num_iterations for i in range(len(record_files))],
               'length': [[0]*num_iterations for i in range(len(record_files))],
               'citation_emphasis_ratio': [[0]*num_iterations for i in range(len(record_files))],
               'coherence_ratio': [[0]*num_iterations for i in range(len(record_files))],
               'contribution_existence': [[0]*num_iterations for i in range(len(record_files))],
               'contribution_type_correctness': [[0]*num_iterations for i in range(len(record_files))],
               'contribution_ratio': [[0]*num_iterations for i in range(len(record_files))],
               'hard_const_complete': [[0]*num_iterations for i in range(len(record_files))]}

    for i, file in enumerate(record_files):
        record = utils.read_json(os.path.join(args.output_path, 'records', file))
        total_citation = len(record["1"]['citation_eval']['citation_emphasis'])
        for iteration in range(num_iterations):
            # Number of hallucinated papers / number of papers needs to be cited
            results['hallucinated_paper_ratio'][i][iteration] = len(record[str(iteration+1)]['citation_eval']['hallucinated_papers']) / total_citation

            # Number of missing papers / number of papers needs to be cited
            results['missing_paper_ratio'][i][iteration] = len(record[str(iteration + 1)]['citation_eval']['missing_papers']) / total_citation

            # Binary evaluation whether draft length is in tolerance interval
            results['length'][i][iteration] = int(record[str(iteration + 1)]['citation_eval']['total_length'] == "Adequate emphasis")

            # Number of citations with adequate emphasis / number of papers needs to be cited
            results['citation_emphasis_ratio'][i][iteration] = sum(value == "Adequate emphasis" for value in record[str(iteration + 1)]['citation_eval']['citation_emphasis'].values()) / total_citation

            # Average of coherent sentences over all citations
            if sum(len(record[str(iteration + 1)]['coherence_eval'][citation]) for citation in record[str(iteration + 1)]['coherence_eval']) > 0:
                results['coherence_ratio'][i][iteration] = sum(record[str(iteration + 1)]['coherence_eval'][citation][sentence]['final_score']
                                                               for citation in record[str(iteration + 1)]['coherence_eval']
                                                               for sentence in record[str(iteration + 1)]['coherence_eval'][citation]) / sum(len(record[str(iteration + 1)]['coherence_eval'][citation])
                                                                                                                                         for citation in record[str(iteration + 1)]['coherence_eval'])
            else:
                results['coherence_ratio'][i][iteration] = 0

            # Binary evaluation of contribution-positioning existence
            results['contribution_existence'][i][iteration] = int(record[str(iteration + 1)]['contribution_eval']['type']['final_type'] != 3)

            # Binary evaluation of correctness of contribution-positioning type
            results['contribution_type_correctness'][i][iteration] = int(record[str(iteration + 1)]['contribution_eval']['type']['final_type'] == record[str(iteration + 1)]['contribution_eval']['expected_type'])

            # Average of correct positioned paragraphs over all paragraphs depending on positioning type
            if results['contribution_type_correctness'][i][iteration]:
                results['contribution_ratio'][i][iteration] = sum(record[str(iteration + 1)]['contribution_eval']['check'][paragraph]['final_score']
                                                               for paragraph in record[str(iteration + 1)]['contribution_eval']['check']) / len(record[str(iteration + 1)]['contribution_eval']['check'])

            # Hard constraints
            if results['hallucinated_paper_ratio'][i][iteration] == 0 and results['missing_paper_ratio'][i][iteration] == 0 \
                and results['coherence_ratio'][i][iteration] == 1 and results['contribution_existence'][i][iteration] == 1:

                results['hard_const_complete'][i][iteration] = 1


    utils.save(results, os.path.join(args.output_path, 'eval.json'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--full_dataset_file', required=True, type=str)
    arguments = parser.parse_args()
    main(arguments)