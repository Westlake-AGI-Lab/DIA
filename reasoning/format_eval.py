import json
import re
import os
import argparse


def read_files(file_path):
    files = os.listdir(file_path)
    files = [f for f in files if "results" not in f]

    resps_list = []
    latency_list = []
    doc_ids = []
    for file in files:
        with open(os.path.join(file_path, file), 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'resps' in data:
                        if data['doc_id'] not in doc_ids:
                            doc_ids.append(data['doc_id'])
                            resps_list.append(data['resps'][0][0])
                            if 'generation_latency_seconds' in data:
                                latency_list.append(float(data['generation_latency_seconds']))
                            else:
                                latency_list.append(0.0)
                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    continue
    print(f'len(resps_list): {len(resps_list)}, len(latency_list): {len(latency_list)}')
    return resps_list, latency_list


def format_reward(resps_list):
    total = len(resps_list)
    correct = 0
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    for answer in resps_list:
        if re.match(pattern, answer, re.DOTALL):
            correct += 1
    return correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate format reward for reasoning DIA outputs")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Path to the lm_eval results directory containing JSONL files")
    args = parser.parse_args()

    resps_list, latency_list = read_files(args.results_dir)
    latency_list = [latency for latency in latency_list if latency != 0.0]
    average_latency = sum(latency_list) / len(latency_list) if latency_list else 0.0
    print('--------------------------------')
    print('Task: MATH | Format reward evaluation')
    print(f"reward: {format_reward(resps_list)}, average_latency: {average_latency}")
