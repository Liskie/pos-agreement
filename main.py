import argparse

from agreement import agreement_score

argparser = argparse.ArgumentParser()

argparser.add_argument('-d', '--data_dir', type=str, default='data')

if __name__ == '__main__':
    args = argparser.parse_args()

    group2score = agreement_score(args.data_dir)

    print(f'Agreement scores:')
    for group, score in group2score.items():
        print(f'{group}: {score:.03f}')

    with open('out/results.csv', 'w') as writer:
        writer.write('Group, Score\n')
        for group, score in group2score.items():
            writer.write(f'{group}, {score:.03f}\n')