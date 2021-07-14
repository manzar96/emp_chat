import os
import numpy as np
import argparse
from core.metrics.metrics import calc_sentence_bleu_score, \
    calc_word_error_rate, distinct_1, distinct_2, avg_len, bleu_parlai
from transformers import T5Tokenizer


def calc_metrics(options,tokenizer):
    outfile = open(os.path.join(options.inputfile), "r")
    lines = outfile.readlines()
    bleu1=[]
    bleu2=[]
    bleu3=[]
    bleu4 = []
    word_error_rate = []
    all_outputs = []
    for line in lines:
        inp, out, trgt,emo = line[:-1].split("\t\t")
        all_outputs.append(out)
        # inp = tokenizer.encode(inp)
        # out = tokenizer.encode(out)
        # trgt = tokenizer.encode(trgt)
        trgt = trgt.split(" ")
        out = out.split(" ")
        bleu1.append(calc_sentence_bleu_score(trgt, out, n=1))
        bleu2.append(calc_sentence_bleu_score(trgt, out, n=2))
        bleu3.append(calc_sentence_bleu_score(trgt, out, n=3))
        bleu4.append(calc_sentence_bleu_score(trgt, out, n=4))
        # bleu1.append(bleu_parlai([trgt], out, n=1))
        # bleu2.append(bleu_parlai([trgt], out, n=2))
        # bleu3.append(bleu_parlai([trgt], out, n=3))
        # bleu4.append(bleu_parlai([trgt], out, n=4))

        word_error_rate.append(calc_word_error_rate(trgt, out))
    print("BLEU1: {}".format(np.mean(bleu1)))
    print("BLEU2: {}".format(np.mean(bleu2)))
    print("BLEU3: {}".format(np.mean(bleu3)))
    print("BLEU4: {}".format(np.mean(bleu4)))
    print("Average BLEU score: {}".format( (np.mean(bleu1)+np.mean(
        bleu2)+np.mean(bleu3)+np.mean(bleu4))/4.0 ) )
    #print("Word Error Rate: {}".format(np.mean(word_error_rate)))

    distinct1 = distinct_1(all_outputs)
    distinct2 = distinct_2(all_outputs)
    avg_length = avg_len(all_outputs)
    print("Distinct 1: ",distinct1)
    print("Distinct 2: ",distinct2)
    print("Average Length: ",avg_length)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inputfile",
        type=str,
        required=True,
        help="CSV file where the generated responses are stored",
    )
    options = parser.parse_args()

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    calc_metrics(options,tokenizer=tokenizer)

