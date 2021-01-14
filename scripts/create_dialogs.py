import argparse
import csv


def dialogs(inputfile1,inputfile2,outfile):

    infile2 = open(inputfile2, mode='r')
    lines2 = infile2.readlines()[1:]
    dict2 = {}
    for line in lines2:
        index,inp,out,target = line[:-1].split("^")
        dict2[index] = [inp,out,target]

    infile2.close()

    infile1 = open(inputfile1, mode='r')
    lines1 = infile1.readlines()[1:]

    outf = open(outfile, "w")
    for line in lines1:
        index1,emo1, inp1, out1, target1 = line[:-1].split("^")
        outf.write("Emotion: "+emo1+"\n")
        outf.write("Conversation\n")
        outf.write("Input:  "+inp1+"\n")
        outf.write("Generated response 1:  "+out1+"\n")
        outf.write("Generated response 2:  "+dict2[index1][1]+"\n")
        outf.write("Target response:  "+target1+"\n\n")
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
        "--inputfile1",
        type=str,
        required=True,
        help="Input generated sampled file to be loaded.",
    )


    parser.add_argument(
        "--inputfile2",
        type=str,
        required=True,
        help="Input generated file from which to get gen responses.",
    )
    options = parser.parse_args()
    dialogs(options.inputfile1,options.inputfile2,options.outfile)