import copy
import csv
import math
import cmath
import time
import itertools
rows = []

with open("bigger_list.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)


def find_combos(whole_list, customer):
    products = []
    for row_prod in whole_list[customer*500:(customer+1)*500]:
        if len(row_prod) > 1 and row_prod[2]:
            product = row_prod[1]
            products.append(product)
    combos = list(itertools.product(products, products))
    return combos


def find_complements(combinations, customer):
    all_prods = []
    combination_list = []
    for combination in combinations:
        if combination[0] != combination[1] and combination not in combination_list:
            combination_list.append(combination)
            deactivate = 0
            for rowe in rows[customer*500:(customer+1)*500]:
                if len(rowe) > 1 and rowe[2]:
                    if int(rowe[1]) == int(combination[0]):
                        deactivate += 1
                        compare1 = list(copy.deepcopy(rowe[2:]))
                    elif int(rowe[1]) == int(combination[1]):
                        deactivate += 1
                        compare2 = list(copy.deepcopy(rowe[2:]))
            if compare1.count("2") > 5 and compare2.count("2") > 5 and deactivate == 2:
                deactivate = 0
                subtracted = [int(a_i) - int(b_i) for a_i, b_i in zip(compare1, compare2)]
                product = abs(sum(subtracted))/(((compare1.count("2")+compare2.count("2"))/2))
                all_prods.append([combination[0], combination[1], product])
    return all_prods


def GetSequences(guesses_in): # This function fetches all sequences, sequences are then matched and the next item is added to an index in a new list
    list_length = len(guesses_in)
    next_guesses = []
    time_lag = range(0, 5) #modify this to be a function input later
    for lag in time_lag:
        sequence_length = 0
        while list_length > lag:
            found_match, start_range = 0, 0
            end_range = sequence_length  # note that sequence length will increase 1 at the end of each loop
            if sequence_length > 0:
                sequence = guesses_in[(list_length-sequence_length-lag-1):list_length-lag]
            else:
                sequence = [guesses_in[-1-lag]]
            for x in range(list_length-sequence_length):
                if guesses_in[start_range:end_range+1] == sequence and len(guesses_in) > end_range + 1 + lag:  # we go over the list of the guesses and look for matches, then add any matches to a new list
                    distance_from_end = list_length - end_range
                    next_guesses.append([sequence, guesses_in[end_range+1+lag], distance_from_end, lag])
                    found_match += 1
                start_range += 1
                end_range += 1
            if found_match < 2:
                break
            sequence_length += 1
    return next_guesses


def division_by_zero(n, d):
    return n / d if d else 0


def log_loss(predicted, actual):
    p = predicted
    y = int(actual) - 1
    if p == 1:
        p = p - 0.02
    if p == 0:
        p = p + 0.02
    loss_calc = abs((y*(math.log(p))) + (1-y) * math.log(1-p))
    return loss_calc

"""def backwards_propagation(weight_list):
    weight_matrix_1 = weight_list[0]
    weight_matrix_2 = weight_list[1]
    weight_matrix_3 = weight_list[2]
    for x in range(len(weight_matrix_1)):
        weight_matrix_1[x] ="""


def Get_Probabilities(input_list, weightlist, function_number, weights_ngram):
    guesses_in = input_list
    probability_list = []
    processed_data = GetSequences(guesses_in)
    windex = weightlist[0]
    weight_length1, weight_lag1, weight_frequency1, weight_recency1, meta_weight1 =\
        windex[0],  windex[1],  windex[2],  windex[3],  windex[4]
    windex2 = weightlist[1]
    weight_length2, weight_lag2, weight_frequency2, weight_recency2, meta_weight2 =\
        windex2[0],  windex2[1],  windex2[2],  windex2[3],  windex2[4]
    windex3 = weightlist[2]
    weight_length3, weight_lag3, weight_frequency3, weight_recency3, meta_weight3 =\
        windex3[0],  windex3[1],  windex3[2],  windex3[3],  windex3[4]

    setz = sorted(set(guesses_in))
    setz = list(setz)
    for z in setz:
        probability_list.append([z])
    probability_list = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    probability_list2 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    probability_list3 = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    for sub_list in processed_data:
        counter = 0
        if len(sub_list) > 1:  # Generates a likelyhood predictions score for the next number
            next_number = int(sub_list[-3])
            lag = int(sub_list[-1])
            length_multiplier = len(sub_list[0]) * weight_length1
            lag_multiplier = weight_lag1 * (lag+1)
            frequency_multiplier = int(guesses_in.count(str(next_number))) * weight_frequency1
            recency_multiplier = int(sub_list[-2]) * weight_recency1
            ngram_index = (len(guesses_in)) - int(sub_list[-2]) + 1 + lag
            f1 = ((frequency_multiplier) + (recency_multiplier) + (lag_multiplier) + (length_multiplier)) \
                             * ((weights_ngram[0][ngram_index]))
            probability_list[next_number-1].append(f1)

            length_multiplier2 = len(sub_list[0]) * weight_length2
            lag_multiplier2 = weight_lag2 * (lag+1)
            frequency_multiplier2 = int(guesses_in.count(str(next_number))) * weight_frequency2
            recency_multiplier2 = int(sub_list[-2]) * weight_recency2
            f2 = ((frequency_multiplier2) + (recency_multiplier2) + (lag_multiplier2) + (length_multiplier2)) \
                 * ((weights_ngram[1][ngram_index]))
            probability_list2[next_number-1].append(f2)

            length_multiplier3 = len(sub_list[0]) * weight_length3
            lag_multiplier3 = weight_lag3 * (lag+1)
            frequency_multiplier3 = int(guesses_in.count(str(next_number))) * weight_frequency3
            recency_multiplier3 = int(sub_list[-2]) * weight_recency3
            f3 = ((frequency_multiplier3) + (recency_multiplier3) + (lag_multiplier3) + (length_multiplier3)) \
                 + ((weights_ngram[2][ngram_index]))
            probability_list3[next_number-1].append(f3)
            counter += 1
    prob_outcomes = []
    prob_outcomes2 = []
    prob_outcomes3 = []
    if len(guesses_in) > 0 and len(processed_data) > 0:
        for h in probability_list:
            countr = 0
            if len(h) > 1:
                prob_outcomes.append([countr+1,sum(h[1:])])
            countr += 1
        for h in probability_list2:
            countr = 0
            if len(h) > 1:
                prob_outcomes2.append([countr+1,sum(h[1:])])
            countr += 1
        for h in probability_list3:
            countr = 0
            if len(h) > 1:
                prob_outcomes3.append([countr+1,sum(h[1:])])
            countr += 1
        #idx, out1 = max(prob_outcomes, key=lambda item: item[-1])
        #idx2, out2 = min(prob_outcomes, key=lambda item: item[-1])
        if len(prob_outcomes) > 1:
            out1 = abs(prob_outcomes[0][-1]*meta_weight1) + 0.01
            out2 = abs(prob_outcomes[1][-1]) + 0.01
            out3 = abs(prob_outcomes2[0][-1]*meta_weight2) + 0.01
            out4 = abs(prob_outcomes2[1][-1]) + 0.01
            out5 = abs(prob_outcomes3[0][-1]*meta_weight3) + 0.01
            out6 = abs(prob_outcomes3[1][-1]) + 0.01
            sigmoid1 = 1/(1+math.e**-out1)
            sigmoid2 = 1/(1+math.e**-out2)
        counts = 0
        weights_secondlayer = weightlist[3:]
        for y in weights_secondlayer:
            counts += 1
            if counts == 1:
                percept_1 = abs(out1*y[0] + out2*y[1] + out3*y[2] + out4*y[3] + out5*y[4] + out6*y[5])
            if counts == 2:
                percept_2 = abs(out1*y[0] + out2*y[1] + out3*y[2] + out4*y[3] + out5*y[4] + out6*y[5])
            if counts == 3:
                percept_3 = abs(out1*y[0] + out2*y[1] + out3*y[2] + out4*y[3] + out5*y[4] + out6*y[5])
        percept_new1 = percept_1 / (percept_3 + percept_2 + percept_1)
        percept_new2 = percept_2 / (percept_3 + percept_2 + percept_1)
        percept_new3 = percept_3 / (percept_3 + percept_2 + percept_1)
        percept_1 = percept_new1
        percept_2 = percept_new2
        percept_3 = percept_new3
        #print(percept_1,percept_2,percept_3)
        if percept_1 > 0.333:
            prediction = ((percept_2 ** 1.4) + (percept_3 ** 1.7) + (percept_1**1.6)) / percept_1**1.5
            prediction = 1/(prediction+1)
        elif percept_2 > 0.333:
            prediction = ((percept_2 ** 1.4) + (percept_3 ** 1.1) + (percept_1**1.4)) / percept_2**2.2
            prediction = 1/(1+prediction)
        elif percept_3 > 0.333:
            prediction = ((percept_2 ** 1.8) + (percept_3 ** 1.3) + (percept_1**1.7)) / (0.01 + percept_3**2)
            prediction = 1 / (1 + prediction)
        return prediction

def learn(sequence, theshold_accuracy):
    learned_list = []  # this list stores the tested weights and their accuracy
    weight_length, weight_frequency, weight_lag, weight_recency, meta_w = 1,1,1,1,1
    correct_ratio = 3000
    weight_number = 0
    step_size = 0.1
    swing_limit = 5
    counts = 0
    break_count = 0
    no_improvement_limit = 50
    iteration_limit = 50
    weight_limit = 1000000
    result_matches = 500
    accuracy_list = [1, 1]
    gradients = [1, 1]
    loss_list = []
    function_number = 0
    trial_weights = copy.deepcopy(all_weights)
    trial_purchase = copy.deepcopy(purchase_weights)
    while abs(weight_length) + abs(weight_lag) + abs(weight_frequency) + abs(weight_recency) + abs(meta_w) \
            < weight_limit \
            and counts < iteration_limit\
            and break_count < no_improvement_limit\
            and result_matches > theshold_accuracy\
            and result_matches > 0.04:
        auto_count = 0
        last_prob = 100
        while 0 <= auto_count <= swing_limit and correct_ratio > 0.05:
            counts += 1
            odds_improved = 0
            if counts > iteration_limit or correct_ratio < 0.03:
                break
            if function_number < 6:
                boosting = 0
                weight = all_weights[function_number][weight_number]
                step_size = step_sizes[function_number][weight_number]
                trial_weights[function_number][weight_number] = weight + step_size
            else:
                boosting = 1
                weight = purchase_weights[function_number-5][weight_number]
                step_size = purchase_step_sizes[function_number-5][weight_number-5]
                trial_purchase[function_number-5][weight_number] = weight + step_size

            cnt_loop = 0
            for n in range(30, len(sequence) - 1): # Compare the forecast based on guesses 0:n to the actual outcome
                if function_number < 6:
                    predicted_number = Get_Probabilities(sequence[0:n], trial_weights, function_number, purchase_weights)
                else:
                    predicted_number = Get_Probabilities(sequence[0:n], trial_weights, function_number, trial_purchase)

                if predicted_number is not None:  # Checks if simulation predicted the next value
                    loss_list.append(log_loss(predicted_number, sequence[n+1]))
                    correct_ratio = sum(loss_list) / len(loss_list)
            accuracy_list.append(correct_ratio)
            if len(gradients) > 1:
                gradient = (accuracy_list[-2] - accuracy_list[-1]) / accuracy_list[-1]
                gradients.append(gradient)
            else:
                gradient = -1
            #print(auto_count, function_number, weight_number,  step_size, gradient, correct_ratio, all_weights, purchase_weights)
            if last_prob > correct_ratio:
                last_prob = correct_ratio
                break_count = 0
                odds_improved = 1
                if gradients[-1] < gradients[-2]:
                    if function_number < 6:
                        step_sizes[function_number][weight_number] = step_sizes[function_number][weight_number] * (gradients[-1]/(gradients[-2]+0.0000000000000001))
                    else:
                        purchase_step_sizes[function_number-5][weight_number] = purchase_step_sizes[function_number-5][weight_number] * (gradients[-2]+0.0000000000000001)
                    if gradient < 0.0000001: #min improvement rate
                        auto_count += 1
           # print(correct_ratio, last_prob)
            if correct_ratio > last_prob or 0 > gradients[-1]:
                break_count += 1
                odds_improved = 0
                auto_count = swing_limit
                break
            else:
                auto_count += 1
            if odds_improved == 1 and function_number < 6:
                all_weights[function_number][weight_number] = weight + step_size
                trial_weights = copy.deepcopy(all_weights)
            if odds_improved == 1 and function_number > 5:
                purchase_weights[function_number-5][weight_number] = weight + step_size
                trial_purchase= copy.deepcopy(purchase_weights)
            #  find a cleaner solution to the below
        if auto_count >= swing_limit and odds_improved == 0:
            swing_limit += 0.1
            if function_number < 6:
                step_sizes[function_number][weight_number] = step_sizes[function_number][weight_number] * -1.1
            elif function_number > 6:
                purchase_step_sizes[function_number-5][weight_number] = purchase_step_sizes[function_number-5][weight_number] * -1.1
            #print("Step size increased to " + str(step_size))
        weight_number += 1
        if weight_number > 4 and boosting == 0:
            weight_number = 0
            function_number += 1
        if weight_number > len(sequence)-1 and boosting == 1:
            weight_number = 0
            function_number += 1
        if function_number > 7:
            function_number = 0

    return all_weights, purchase_weights


total_right, total_wrong = 1, 1
one_count, two_count = 0, 0
training_lim = 60
testing_lim = 88
dyn_length, dyn_frq, dyn_lag, dyn_recency, dyn_meta = 1, 1, 1, 1, 1
total_wrong = 0
total_right = 1
losses_testing = []
rows = []
total_thresholds = []
step_sizes = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
              [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
all_weights = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
               [1, 1, 1, 1, 1, 1]]
purchase_weights = [[], [], []]
purchase_step_sizes = [[], [], []]
for n in range(3):
    for k in range(900):
        purchase_weights[n].append(1)
        purchase_step_sizes[n].append(1)
with open("bigger_list.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)

customer_count = 0

for row in rows:
    step_sizes = [[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    all_weights = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1, 1]]
    print(row)
    found_complements = []
    weight_complements= []
    if len(row) > 1 and row[2] and int(row[0]) > 38:
        print("Customer: ", row[0], "Product number: ", row[1])
        customer = row[0]
        product = row[1]

        """ combs = find_combos(rows, customer)
        produce = find_complements(combs, customer)
        print(produce)
        for pro in produce:
            if str(row[1]) == str(pro[0]):
                found_complements.append([pro[-2], pro[-1]])
        for com in found_complements:
            index = (int(customer) * 500) + ((int(com[0]*2)) - 1)
            weight_complements.append([rows[index][2:], pro[-1]])
        print(weight_complements)
        print("found")"""

        row = row[2:]
        timeseries, live_list = [], [],
        probability_list = []
        losses_product = []
        training_data = row[0:training_lim]
        testing_data = row[training_lim:testing_lim]
        if (row[0:30].count("2") / (row[0:30].count("2") + row[0:30].count("1"))) > 0.05\
                and (row[training_lim:testing_lim].count("2") / (row[training_lim:testing_lim].count("2") +
                                                                 row[training_lim:testing_lim].count("1"))) > 0.05:
            print("Proportion of purchases in training data:",
                  training_data.count("1") / (training_data.count("2") + training_data.count("1")))
            print("Proportion of purchases in testing data:",
                  testing_data.count("1") / (testing_data.count("2") + testing_data.count("1")))
            tester = []
            for num in training_data:
                num = int(num)
                timeseries.extend(str(num))
            for num in testing_data:
                num = int(num)
                tester.extend(str(num))
                timeseries.extend(str(num))
            p = testing_data.count("1") / (testing_data.count("2") + testing_data.count("1"))
            threshold_accuracy = -1 * ((p*(math.log(p))) +((1-p)*math.log(1-p)))
            """threshold_accuracy = log_loss(max(timeseries.count("1") / (timeseries.count("1") + timeseries.count("2")),
                                              timeseries.count("2") / (timeseries.count("1") + timeseries.count("2"))), 2)"""
            total_thresholds.append(threshold_accuracy)

            correct_guesses, incorrect_guesses, newcount = 0, 0, 0
            simul_right = 0.001
            simul_wrong = 0
            learned = learn(timeseries[0:training_lim], threshold_accuracy)
            for h in range(training_lim, testing_lim-1):
                predicted_number = Get_Probabilities(timeseries[0:h], learned[0], 0, learned[1])
                if predicted_number is not None:  # Checks if simulation predicted the next value
                    testing_loss = log_loss(predicted_number, int(timeseries[h + 1]))
                    losses_testing.append(testing_loss)
                    losses_product.append(testing_loss)
                    print(sum(losses_testing)/len(losses_testing),predicted_number, int(timeseries[h + 1]) - 1, h)
                    if int(timeseries[h + 1]) == 1:
                        one_count += 1
                    else:
                        two_count += 1
                loss_product = sum(losses_product)/len(losses_product)
                loss_testing = sum(losses_testing)/len(losses_testing)
            final_prediction = Get_Probabilities(timeseries[0:89], learned[0], 0, learned[1])
            print(loss_product, "log_loss for weeks: ", training_lim, " to ", testing_lim)
            print(loss_testing, "log_loss correctly for ", one_count+two_count,
                  " predictions in testing so far, baseline ratio:", sum(total_thresholds)/len(total_thresholds))
            with open('final_predictions4.csv', 'a') as file2:
                writer = csv.writer(file2)
                writer.writerow([customer, product, final_prediction, threshold_accuracy, sum(losses_product)/len(losses_product)])
                print(customer, product, final_prediction, threshold_accuracy, testing_loss)
        else:
            with open('final_predictions4.csv', 'a') as file2:
                writer = csv.writer(file2)
                writer.writerow([customer, product, 0.02, 0, 0])


