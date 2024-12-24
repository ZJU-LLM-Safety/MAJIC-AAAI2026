import os 
import json

file_name = 'markov_test_literary.json'

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.dirname(__file__))
    with open(os.path.join(dir_path, 'results', file_name), 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    sum = 0

    for result in results:
        if result['best_score'] == 1.0 and result['type'] == 'markov':
            sum += 1

    print(sum)