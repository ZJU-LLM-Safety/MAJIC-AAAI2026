import os 
import json

file_name = 'markov_test_initvec_02_gemma.json'

if __name__ == '__main__':
    with open(file_name, 'r', encoding='utf-8') as f:
        results = json.load(f)
    sum = 0
    for result in results:
        if result['best_score'] == 1.0 and result['type'] == 'markov':
            sum += 1

    print(sum)