import math
import os
from collections import defaultdict
from typing import NoReturn

import jsonlines
import numpy as np
from nltk.metrics import agreement

def fleiss_kappa(worker_tags: list[list[str]]) -> float:
    """
    Calculate the Fleiss' Kappa score of several(>=2) workers' annotations on the same sequence.
    Example of worker_tags:
    [
        ['B-POS', 'I-POS', 'O'    ], # worker 1
        ['B-POS', 'O',     'O'    ], # worker 2
        ['B-POS', 'I-POS', 'I-POS'], # worker 3
    ]
    """
    assert len(worker_tags) >= 2

    # Check if all workers give no annotation spans on the sequence.
    is_only_O = True
    for tags in worker_tags:
        if not is_only_O:
            break
        for tag in tags:
            if tag != 'O':
                is_only_O = False
                break
    if is_only_O:
        return 1.0

    # The NLTK implementation.
    data = []
    for worker_idx, tags in enumerate(worker_tags):
        for tag_idx, tag in enumerate(tags):
            data.append((worker_idx, tag_idx, tag))
    task = agreement.AnnotationTask(data=data)
    return task.multi_kappa()

def to_tags(line: dict) -> list[str]:
    """
    Convert a line to a tag sequence.
    """
    seq = line['text'].split('    ')[0].split()
    # Annotate all sub-spans and concatenate them.
    span2tags = {}
    for length in range(1, len(seq) + 1):
        for start in range(len(seq)):
            if start + length > len(seq):
                continue
            span2tags[(start, start + length)] = ['O' for _ in range(length)]
    for label in line['label']:
        start = math.floor(label[0] / 3)
        end = math.ceil(label[1] / 3)
        pos = label[2]
        span2tags[(start, end)] = [str(pos) for _ in range(end - start)]

    concat_tags = []
    for tags in span2tags.values():
        concat_tags.extend(tags)

    return concat_tags

def agreement_score(data_dir: str) -> dict[str, float]:
    """
    Calculate agreement score for every group of workers.
    """
    group2score: dict[str, float] = {}
    for current_dir, sub_dirs, filenames in os.walk(data_dir):
        if sub_dirs:
            continue
        seq2worker_tags: dict[int, list[list[str]]] = defaultdict(list)
        # Collect annotations.
        for filename in filenames:
            if filename == 'admin.jsonl':
                continue
            with jsonlines.open(os.path.join(current_dir, filename)) as reader:
                for line in reader:
                    seq2worker_tags[line['id']].append(to_tags(line))
        # Calculate scores.
        seq2score: dict[int, float] = {}
        for seq, worker_tags in seq2worker_tags.items():
            seq2score[seq] = fleiss_kappa(worker_tags)
        group2score[current_dir.split('/')[-1]] = float(np.mean(list(seq2score.values())))
    return group2score

if __name__ == '__main__':
    print(agreement_score('data'))