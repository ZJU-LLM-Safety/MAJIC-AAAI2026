import json
import os
import re

def analyze_json_files_gpt4o(directory, keyword1="results", keyword2="gpt4o"):
    """
    Analyzes JSON files in a directory, matching filenames based on keywords,
    extracting IDs where "best_score" is 1.0, and calculating coverage. Sorts
    output by filename number and IDs within each file.

    Args:
        directory: The directory containing the JSON files.
        keyword1: The first keyword for filename matching.
        keyword2: The second keyword for filename matching.

    Returns:
        A dictionary containing:
        - "all_covered_ids": A sorted list of all unique IDs covered across all matched files.
        - "file_covered_ids": A dictionary mapping filenames to sorted lists of covered IDs in each file, sorted by the number in the filename.
        - "coverage_percentage": The percentage of IDs (0-49) covered across all matched files.
    """

    all_covered_ids = set()
    file_covered_ids = {}
    all_possible_ids = set(range(50))

    for filename in os.listdir(directory):
        if keyword1 in filename and keyword2 in filename:
            try:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                covered_ids = set()
                for item in data:
                    if "best_score" in item and item["best_score"] == 1.0:
                        covered_ids.add(item["id"])

                all_covered_ids.update(covered_ids)
                file_covered_ids[filename] = sorted(list(covered_ids)) # Store as sorted list

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {filename}: {e}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    coverage_percentage = (len(all_covered_ids) / len(all_possible_ids)) * 100

    # Sort file_covered_ids by the number extracted from the filename
    sorted_file_covered_ids = dict(sorted(file_covered_ids.items(), key=lambda item: int(re.search(r"_(\d+)_", item[0]).group(1))))

    return {
        "all_covered_ids": sorted(list(all_covered_ids)), # Return as sorted list
        "file_covered_ids": sorted_file_covered_ids,
        "coverage_percentage": coverage_percentage
    }

def analyze_json_files_llama3(directory, keyword1="results", keyword2="llama3"):
    all_covered_ids = set()
    file_covered_ids = {}
    all_possible_ids = set(range(50))

    for filename in os.listdir(directory):
        if keyword1 in filename and keyword2 in filename:
            try:
                filepath = os.path.join(directory, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                covered_ids = set()
                for item in data:
                    if "best_score" in item and item["best_score"] == 1.0:
                        covered_ids.add(item["id"])

                all_covered_ids.update(covered_ids)
                file_covered_ids[filename] = sorted(list(covered_ids)) # Store as sorted list

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in {filename}: {e}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    coverage_percentage = (len(all_covered_ids) / len(all_possible_ids)) * 100

    # Sort file_covered_ids by the number extracted from the filename
    sorted_file_covered_ids = dict(sorted(file_covered_ids.items(), key=lambda item: int(re.search(r"_(\d+)_", item[0]).group(1))))

    return {
        "all_covered_ids": sorted(list(all_covered_ids)), # Return as sorted list
        "file_covered_ids": sorted_file_covered_ids,
        "coverage_percentage": coverage_percentage
    }


# Example usage (replace with your directory)
directory = "./results"
# results = analyze_json_files_gpt4o(directory) # 得到4o的结果汇总
results = analyze_json_files_llama3(directory) # 得到llama3的结果汇总


print("Covered IDs in each file:")
for filename, ids in results["file_covered_ids"].items():
    print(f"{filename}: {ids}")

print("\nAll covered IDs:", results["all_covered_ids"])
print(f"\nCoverage percentage: {results['coverage_percentage']:.2f}%")