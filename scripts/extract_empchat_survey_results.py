import argparse
import statistics
from scipy import stats
import pingouin
from statsmodels.stats import weightstats,proportion
import numpy as np
import matplotlib.pyplot as plt



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--attempt",
        type=int,
        required=True,
        help="chose attempt number!",
    )
    options = parser.parse_args()
    attempt = options.attempt

    infile = open('/home/manzar/Desktop/attempt{}/attempt{}/used_ips'
                  '.csv'.format(attempt,attempt), "r")
    lines = infile.readlines()

    users = []
    for index, line in enumerate(lines):
        if index == 0:
            # ignore header
            continue
        id, address, eng_lvl = line.split(",")
        users.append([id,address,eng_lvl])
    infile.close()

    infile = open('/home/manzar/Desktop/attempt{}'
                  '/attempt{}/pairwise_comparison.csv'.format(attempt,attempt), "r")
    lines = infile.readlines()

    pairwise_comp = []
    mymodel_comp_selections = 0
    dodeca_comp_selections = 0
    for index, line in enumerate(lines):
        if index == 0:
            # ignore header
            continue
        id, selection, address_id = line.split(",")
        pairwise_comp.append(selection)
        if selection == '1':
            mymodel_comp_selections += 1
        else:
            dodeca_comp_selections += 1
    infile.close()

    infile = open('/home/manzar/Desktop/attempt{}/attempt{}'
                  '/pairwise_emotions.csv'.format(attempt,attempt), "r")
    lines = infile.readlines()

    pairwise_emo = []
    mymodel_emo_selections = 0
    dodeca_emo_selections = 0
    for index, line in enumerate(lines):
        if index == 0:
            # ignore header
            continue
        id, selection, address_id = line.split(",")
        pairwise_emo.append(selection)
        if selection == '1':
            mymodel_emo_selections += 1
        else:
            dodeca_emo_selections += 1
    infile.close()

    infile = open('/home/manzar/Desktop/attempt{}'
                  '/attempt{}/pairwise_sentiments.csv'.format(attempt,attempt), "r")
    lines = infile.readlines()

    pairwise_sents = []
    mymodel_sent_selections = 0
    dodeca_sent_selections = 0
    for index, line in enumerate(lines):
        if index == 0:
            # ignore header
            continue
        id, selection, address_id = line.split(",")
        pairwise_sents.append(selection)
        if selection == '1':
            mymodel_sent_selections += 1
        else:
            dodeca_sent_selections += 1
    infile.close()

    infile = open('/home/manzar/Desktop/attempt{}'
                  '/attempt{}/ratings_mymodel.csv'.format(attempt,attempt), "r")
    lines = infile.readlines()

    mymodel_emp = []
    mymodel_rel = []
    mymodel_flu = []
    for index, line in enumerate(lines):
        if index == 0:
            # ignore header
            continue
        id, emp,rel,flu, address_id = line.split(",")
        mymodel_emp.append(int(emp))
        mymodel_rel.append(int(rel))
        mymodel_flu.append(int(flu))
    infile.close()

    infile = open('/home/manzar/Desktop/attempt{}'
                  '/attempt{}/ratings_dodeca.csv'.format(attempt,attempt), "r")
    lines = infile.readlines()

    dodeca_emp = []
    dodeca_rel = []
    dodeca_flu = []
    for index, line in enumerate(lines):
        if index == 0:
            # ignore header
            continue
        id, emp,rel,flu, address_id = line.split(",")
        dodeca_emp.append(int(emp))
        dodeca_rel.append(int(rel))
        dodeca_flu.append(int(flu))
    infile.close()


    print("=== Empchat Survey Statistic Results ===")
    print("Number of Participants:  ", len(users))
    print("Number of Samples:  ", len(pairwise_comp))
    print("")
    print("+++ Pairwise Comparison - Evaluating Relevance and Fluency +++")
    print("MyModel Win Percentage:  ", mymodel_comp_selections/len(
        pairwise_comp))
    print("Dodeca Win Percentage:  ", dodeca_comp_selections/len(
        pairwise_comp))
    print("")
    print("+++ Pairwise Comparison - Evaluating Empathy according to "
          "Sentiment "
          "+++")
    print("MyModel Win Percentage:  ", mymodel_sent_selections/len(
        pairwise_sents))
    print("Dodeca Win Percentage:  ", dodeca_sent_selections/len(
        pairwise_sents))
    print("")
    print("+++ Pairwise Comparison - Evaluating Empathy according to Emotion "
          "+++")
    print("MyModel Win Percentage:  ", mymodel_emo_selections/len(
        pairwise_emo))
    print("Dodeca Win Percentage:  ", dodeca_emo_selections/len(
        pairwise_emo))
    print("")
    
    print("+++ Ratings of MyModel +++")
    print("Avg Empathy Rating:  ", statistics.mean(mymodel_emp))
    print("Avg Relevance Rating:  ", statistics.mean(mymodel_rel))
    print("Avg Fluency Rating:  ", statistics.mean(mymodel_flu))
    print("Std Empathy Rating:  ", statistics.stdev(mymodel_emp))
    print("Std Relevance Rating:  ", statistics.stdev(mymodel_rel))
    print("Std Fluency Rating:  ", statistics.stdev(mymodel_flu))
    print("")
    
    print("+++ Ratings of Dodeca Model +++")
    print("Avg Empathy Rating:  ", statistics.mean(dodeca_emp))
    print("Avg Relevance Rating:  ", statistics.mean(dodeca_rel))
    print("Avg Fluency Rating:  ", statistics.mean(dodeca_flu))
    print("Std Empathy Rating:  ", statistics.stdev(dodeca_emp))
    print("Std Relevance Rating:  ", statistics.stdev(dodeca_rel))
    print("Std Fluency Rating:  ", statistics.stdev(dodeca_flu))
    print("")

    # print("+++ Statistical Significance Testing +++")
    # z_emp = scipy.stats.zscore(mymodel_emp,)
    # print("Statistical Significance for Empathy Rating ")
    # print("Null Hypothesis: mean empathy dodeca >= mean empathy mymodel")
    # print("Alternative Hypothesis: mean empathy dodeca < mean empathy mymodel")
    # print(stats.normaltest(mymodel_emp))
    # print(stats.shapiro(dodeca_emp))
    # print(np.histogram(mymodel_emp))
    # print("Since the normality criterion is violated (samples do not come "
    #       "from normal distributions) we should use the Mann Whitney U test!")
    #
    # print("Statistical Significance for Empathy Rating ")
    #
    # print(stats.mannwhitneyu(dodeca_emp,mymodel_emp,alternative='less'))
    # print(stats.mannwhitneyu(dodeca_rel,mymodel_rel,alternative='two-sided'))
    # print(stats.mannwhitneyu(dodeca_flu,mymodel_flu,alternative='less'))

    # print("add ranks:")
    #
    # sum_dodeca = sum(dodeca_emp)
    # sum_mymodel = sum(mymodel_emp)
    # n1 = 308
    # n2 = 308
    # nx=308
    # print(sum_dodeca)
    # print(sum_mymodel)
    # U = n1*n2 + nx*(nx+1)/2 - sum_mymodel
    # print("U: ",U)

    # print(stats.wilcoxon(mymodel_emp,dodeca_emp))
    # print(stats.kruskal(mymodel_emp,dodeca_emp))
    # print(stats.ttest_ind(mymodel_emp,dodeca_emp,equal_var=False,
    #                       alternative='greater'))
    # print(pairwise_comp)
    # weightstats.ztest(pairwise_comp)

    # print(statistics.variance(mymodel_emp))
    # print(statistics.variance(dodeca_emp))
    # f1 = plt.figure(1)
    # plt.hist(mymodel_emp)
    # f1.show()
    # f2 = plt.figure(2)
    # plt.hist(dodeca_emp)
    # f2.show()
    # input()
    # print("z-test for pairwise comparison two-tailed test")
    # n1 = 308
    # n2 = 308
    # p1 = mymodel_comp_selections/n1
    # p2 = dodeca_comp_selections/n2
    # p = (n1*p1+n2*p2)/(n1+n2)
    # temp = p*(1-p)*(1/n1+1/n2)
    # z = (p1-p2)/np.sqrt(temp)
    # print("p1: ",p1)
    # print("p2: ",p2)
    # print("p: ",p)
    # print("temp: ",temp)
    # print("z: ",z)
    # print("for alpha=0.05 the Z_a/2 is 1.96")
    # print("|z|>Z_a/2 as "+str(z)+"is greater than 1.96")
    # print(statistics.mean(mymodel_comp_selections))
    # print(statistics.mean(dodeca_comp_selections))

    print()
    print("+++Binomial testing for pairwise comparison relevance and "
          "fluency+++")
    print(stats.binom_test(mymodel_comp_selections,len(pairwise_comp),p=0.5,
                           alternative='greater'))
    print(stats.binom_test(mymodel_comp_selections, len(pairwise_comp), p=0.5,
                           alternative='two-sided'))

    print("+++Binomial testing for pairwise empathy with emotions+++")
    print(stats.binom_test(mymodel_emo_selections,len(pairwise_emo),p=0.5,
                           alternative='greater'))
    print(stats.binom_test(mymodel_emo_selections,len(pairwise_emo),p=0.5,
                           alternative='two-sided'))

    print("+++Binomial testing for pairwise empathy with sentiments+++")
    print(stats.binom_test(mymodel_sent_selections,len(pairwise_sents),p=0.5,
                           alternative='greater'))
    print(stats.binom_test(mymodel_sent_selections,len(pairwise_sents),p=0.5,
                           alternative='two-sided'))
    print()
    print("+++Mann-Witney U test for empathy rating+++")
    print("one-tailed")
    print(stats.mannwhitneyu(dodeca_emp,mymodel_emp,alternative='less'))
    print("two-tailed")
    print(stats.mannwhitneyu(dodeca_emp,mymodel_emp,alternative='two-sided'))

    print("+++Mann-Witney U test for relevance rating+++")
    print("one-tailed")
    print(stats.mannwhitneyu(dodeca_rel,mymodel_rel,alternative='less'))
    print("two-tailed")
    print(stats.mannwhitneyu(dodeca_rel,mymodel_rel,alternative='two-sided'))
    print("+++Mann-Witney U test for fluency rating+++")
    print("one-tailed")
    print(stats.mannwhitneyu(dodeca_flu,mymodel_flu,alternative='less'))
    print("two-tailed")
    print(stats.mannwhitneyu(dodeca_flu,mymodel_flu,alternative='two-sided'))

    print(mymodel_comp_selections)
    print(mymodel_emo_selections)
    print(mymodel_sent_selections)

    print()
    print("Phase 2")
    print("Number of Participants:  ", 15)
    print("Number of Samples:  ", 15*7)
    print("")
    print("MyModel - Baseline")
    print("+++ Pairwise Comparison - Evaluating Relevance and Fluency +++")
    print("MyModel Win Percentage:  ", 67/105)
    print("Baseline Win Percentage:  ", 38/105)
    print(stats.binom_test(67,105,p=0.5,
                           alternative='greater'))
    print(stats.binom_test(67, 105, p=0.5,
                           alternative='two-sided'))
    print("+++ Pairwise Comparison - Evaluating Empathy accordint to Sentiment "
          "+++")
    print("MyModel Win Percentage:  ", 69/105)
    print("Baseline Win Percentage:  ", 36/105)
    print(stats.binom_test(69,105,p=0.5,
                           alternative='greater'))
    print(stats.binom_test(69, 105, p=0.5,
                           alternative='two-sided'))
    print("+++ Pairwise Comparison - Evaluating Empathy according to Emotion "
          "+++")
    print("MyModel Win Percentage:  ", 62/105)
    print("Baseline Win Percentage:  ", 43/105)
    print(stats.binom_test(62,105,p=0.5,
                           alternative='greater'))
    print(stats.binom_test(62, 105, p=0.5,
                           alternative='two-sided'))


    print("")
    print("Dodeca - Baseline")
    print("+++ Pairwise Comparison - Evaluating Relevance and Fluency +++")
    print("Dodeca Win Percentage:  ", 62/105)
    print("Baseline Win Percentage:  ", 43/105)
    print(stats.binom_test(62,105,p=0.5,
                           alternative='greater'))
    print(stats.binom_test(62, 105, p=0.5,
                           alternative='two-sided'))
    print("+++ Pairwise Comparison - Evaluating Empathy accordint to Sentiment "
          "+++")
    print("Dodeca Win Percentage:  ", 64/105)
    print("Baseline Win Percentage:  ", 41/105)
    print(stats.binom_test(63,105,p=0.5,
                           alternative='greater'))
    print(stats.binom_test(63, 105, p=0.5,
                           alternative='two-sided'))
    print("+++ Pairwise Comparison - Evaluating Empathy according to Emotion "
          "+++")
    print("Dodeca Win Percentage:  ", 63/105)
    print("Baseline Win Percentage:  ", 42/105)
    print(stats.binom_test(63,105,p=0.5,
                           alternative='greater'))
    print(stats.binom_test(63, 105, p=0.5,
                           alternative='two-sided'))