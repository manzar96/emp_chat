import argparse
import json
import csv

def read_responses(inputfile):

    with open(inputfile) as json_file:
        all_data = []
        data = [json.loads(line) for line in json_file]
        for sample in data:
            for turn in sample['dialog']:
                input = turn[0]['text']
                target = turn[0]['eval_labels'][0]
                response = turn[1]['text']
                all_data.append([input, response, target])
        return all_data


def write_responses(outfile,data):

    with open(outfile,'w') as outfile:
        for line in data:
            outfile.write(line[0] + "\t\t" + line[1] + "\t\t" + line[2] +
                          "\n")
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
        help="Input generated sampled file to be loaded.",
    )
    options = parser.parse_args()
    data = read_responses(options.inputfile)
    write_responses(options.outfile,data)