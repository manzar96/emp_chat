import random
import argparse


def sampling(inputfile,outfile):
    infile = open(inputfile, "r")
    lines = infile.readlines()
    emo_dict = {}

    for line in lines:
        inp, out, trgt, emo_label = line[:-1].split("\t\t")
        if emo_label not in emo_dict.keys():
            emo_dict[emo_label] = [[inp,out,trgt]]
        else:
            emo_dict[emo_label].append([inp,out,trgt])

    infile.close()
    outf = open(outfile, "w")
    for emo in emo_dict.keys():
        outf.write("Emotion: "+emo+"\n\n")
        samples = emo_dict[emo]
        conv1, conv2 = random.sample(samples,2)
        outf.write("Conversation 1\n")
        outf.write("Input:  "+conv1[0]+"\n")
        outf.write("Generated response:  "+conv1[1]+"\n")
        outf.write("Target response:  "+conv1[2]+"\n\n")
        outf.write("Conversation 2\n")
        outf.write("Input:  "+conv2[0]+"\n")
        outf.write("Generated response:  "+conv2[1]+"\n")
        outf.write("Target response:  "+conv2[2]+"\n\n")
    outf.close()

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
