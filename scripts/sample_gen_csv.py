import random
import argparse
import csv

def sampling(inputfile,outfile):
    infile = open(inputfile, "r")
    reader = csv.DictReader(infile,delimiter='^')
    emo_dict = {}

    for index,row in enumerate(reader):
        if not index==0:
            if row['emotion'] not in emo_dict.keys():
                emo_dict[row['emotion']] = [[row['id'],row['input'],
                                             row['output'],row['target']]]
            else:
                emo_dict[row['emotion']].append([row['id'],row['input'],
                                                 row['output'],row['target']])

    infile.close()
    outfile = open(outfile, mode='w')
    out_writer = csv.writer(outfile, delimiter='^',
                            quotechar='"', quoting=csv.QUOTE_MINIMAL)
    out_writer.writerow(['id', 'emotion', 'input', 'output', 'target'])

    for emo in emo_dict.keys():
        samples = emo_dict[emo]
        conv1, conv2 = random.sample(samples,2)
        out_writer.writerow([conv1[0],emo, conv1[1],
                             conv1[2],conv1[3]])
        out_writer.writerow([conv2[0],emo, conv2[1],
                             conv2[2],conv2[3]])

    outfile.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outfile",
        type=str,
        required=True,
        help="File where the sampled answers while be stored.",
    )

    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="Input generated file to be sampled.",
    )
    options = parser.parse_args()
    sampling(options.inputfile, options.outfile)
