from nltk.translate.bleu_score import sentence_bleu,corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import Levenshtein as Lev
import re
re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def calc_sentence_bleu_score(reference,hypothesis,n=4):
    """
    This function receives as input a reference sentence(list) and a
    hypothesis(list) and
    returns bleu score.
    (list of words should be given)
    Bleu score formula: https://leimao.github.io/blog/BLEU-Score/
    """
    reference = [reference]
    weights = [1/n for _ in range(n)]
    smoothie = SmoothingFunction(epsilon=1e-12).method1
    # return sentence_bleu(reference, hypothesis, weights)
    return corpus_bleu([reference], [hypothesis],smoothing_function=smoothie,
    weights = weights)


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    """

    s = s.lower()
    s = re_punc.sub(' ', s)
    s = re_art.sub(' ', s)
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '
    s = ' '.join(s.split())
    return s


def bleu_parlai(reference,hypothesis,n=4):
    reference = [normalize_answer(a).split(" ") for a in reference]
    hypothesis = normalize_answer(hypothesis).split(" ")
    weights = [1/n for _ in range(n)]
    smoothie = SmoothingFunction(epsilon=1e-12).method1
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothie,
                         weights=weights)


def calc_word_error_rate(s1,s2):
    """
    Computes the Word Error Rate, defined as the edit distance between the
    two provided sentences (list of words should be given).
    s1:reference
    s2:hypothesis
    """

    # build mapping of words to integers
    ba = set(s1+ s2)
    word2idx = dict(zip(ba, range(len(ba))))

    # map the words to a char array (Levenshtein packages only accepts
    # strings)
    w1 = [chr(word2idx[w]) for w in s1]
    w2 = [chr(word2idx[w]) for w in s2]
    return Lev.distance(''.join(w1), ''.join(w2)) / len(s1)


def distinct_1(lines):
  '''Computes the number of distinct words divided by the total number of words.

  Input:
  lines: List of strings.
  '''
  words = ' '.join(lines).split(' ')
  num_distinct_words = len(set(words))
  return float(num_distinct_words) / len(words)


def distinct_2(lines):
  '''Computes the number of distinct bigrams divided by the total number of words.

  Input:
  lines: List of strings.
  '''
  all_bigrams = []
  num_words = 0

  for line in lines:
    line_list = line.split(' ')
    num_words += len(line_list)
    bigrams = zip(line_list, line_list[1:])
    all_bigrams.extend(list(bigrams))

  return len(set(all_bigrams)) / float(num_words)

def avg_len(lines):
  '''Computes the average line length.

  Input:
  lines: List of strings.
  '''
  return(len([w for s in lines for w in s.strip().split()])/len(lines))

