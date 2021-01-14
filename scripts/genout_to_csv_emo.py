import random
import argparse
import csv

def gen_csv(inputfile,outfile):
    infile = open(inputfile, "r")
    lines = infile.readlines()

    outfile = open(outfile, mode='w')
    out_writer = csv.writer(outfile, delimiter='^',
                                 quotechar='"', quoting=csv.QUOTE_MINIMAL)
    out_writer.writerow(['id','emotion','input','output','target'])



    for index,line in enumerate(lines):
        inp, out, trgt, emo_label = line[:-1].split("\t\t")
        out_writer.writerow([index,emo_label,inp,out,trgt])


    infile.close()
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
    gen_csv(options.inputfile, options.outfile)
