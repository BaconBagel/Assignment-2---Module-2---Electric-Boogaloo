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


def GetSequences(guesses_in): # This function fetches all guesses, the last x guesses are then matched and the next item is added to an index in a new list
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
            f1 = ((frequency_multiplier) + (recency_multiplier) + (lag_multiplier) + (length_multiplier)) \
                             + ((weights_ngram[-next_number-1]))
            probability_list[next_number-1].append(f1)

            length_multiplier = len(sub_list[0]) * weight_length2
            lag_multiplier = weight_lag2 * (lag+1)
            frequency_multiplier = int(guesses_in.count(str(next_number))) * weight_frequency2
            recency_multiplier = int(sub_list[-2]) * weight_recency2
            f2 = ((frequency_multiplier) + (recency_multiplier) + (lag_multiplier) + (length_multiplier)) \
                             + ((weights_ngram[-next_number-1]))
            probability_list2[next_number-1].append(f1)

            length_multiplier = len(sub_list[0]) * weight_length3
            lag_multiplier = weight_lag3 * (lag+1)
            frequency_multiplier = int(guesses_in.count(str(next_number))) * weight_frequency3
            recency_multiplier = int(sub_list[-2]) * weight_recency3
            f3 = ((frequency_multiplier) + (recency_multiplier) + (lag_multiplier) + (length_multiplier)) \
                             + ((weights_ngram[-next_number-1]))
            probability_list3[next_number-1].append(f1)
            counter += 1
    prob_outcomes = []
    prob_outcomes2 = []
    prob_outcomes3 = []
    countr = 0
    if len(guesses_in) > 0 and len(processed_data) > 0:
        for h in probability_list:
            if len(h) > 1:
                prob_outcomes.append([countr+1,sum(h[1:])])
            countr += 1
        for h in probability_list2:
            if len(h) > 1:
                prob_outcomes2.append([countr+1,sum(h[1:])])
            countr += 1
        for h in probability_list3:
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

        percept_1 = 1 - abs(out2 / (out1 + out2))
        percept_2 = abs(out4 / ((out3 + out4)**0.52))
        percept_2 = 1/math.log(percept_2)
        percept_3 = 1 - abs(out6**2 / (out6**2 + out5**2))

        if math.log(percept_2) < 3:
            prediction = percept_1
        elif percept_1 < 0.5 and math.log(percept_2) < 0.22:
            prediction = (percept_2 ** 2 + percept_3 ** 2) /2
        else:
            prediction = (percept_3 ** 2 + percept_3 ** 2) / 2
        return prediction


def learn(sequence, init_length, init_frq, iinit_lag, init_recency, init_meta, threshold_accuracy):
    learned_list = []  # this list stores the tested weights and their accuracy
    purchase_weights = []
    for k in range(89):
        purchase_weights.append(1)
    weight_length, weight_frequency, weight_lag, weight_recency, meta_w = init_length, init_frq, iinit_lag, \
                                                                          init_recency, init_meta # starting values of weights
    correct_ratio = 1
    weight_number = 0
    step_size = 1
    swing_limit = 2000
    counts = 0
    break_count = 0
    no_improvement_limit = 5000000
    iteration_limit = 5000000
    weight_limit = 1000000000
    accuracy_list = [1, 1]
    gradients = [1, 1]
    step_sizes = [[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]]
    loss_list = []
    function_number = 0
    all_weights = [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]
    while abs(weight_length) + abs(weight_lag) + abs(weight_frequency) + abs(weight_recency) + abs(meta_w) \
            < weight_limit \
            and counts < iteration_limit\
            and break_count < no_improvement_limit\
            :
        auto_count = 0
        switch = 0
        last_prob = 1
        weight_adder = 0
        weight_list = all_weights[function_number]
        while 0 <= auto_count <= swing_limit and correct_ratio > 0.05:
            counts += 1
            odds_improved = 0
            if counts > iteration_limit or correct_ratio < 0.03:
                break
            if weight_number < 5:
                weight = all_weights[function_number][weight_number]
                trial_weights = copy.deepcopy(all_weights)
                trial_weights[function_number][weight_number] = trial_weights[function_number][weight_number]
            step_size = step_sizes[function_number][weight_number]

            cnt_loop = 0
            for n in range(20, len(sequence) - 1): # Compare the forecast based on guesses 0:n to the actual outcome
                if weight_number == 0:
                    predicted_number = Get_Probabilities(sequence[0:n], trial_weights, function_number, purchase_weights)

                if predicted_number is not None:  # Checks if simulation predicted the next value
                    loss_list.append(log_loss(predicted_number, sequence[n+1]))
                    correct_ratio = sum(loss_list) / len(loss_list)
                    if log_loss(int(predicted_number), sequence[n+1]) > correct_ratio and weight_number == 5:
                        purchase_weights[cnt_loop] = abs(purchase_weights[cnt_loop]) + step_size
                    elif log_loss(int(predicted_number), sequence[n+1]) < correct_ratio and weight_number == 5:
                        purchase_weights[cnt_loop] = abs(purchase_weights[cnt_loop]) - step_size
                cnt_loop += 1
            accuracy_list.append(correct_ratio)
            if len(gradients) > 1:
                gradient = (accuracy_list[-2] - accuracy_list[-1]) / accuracy_list[-1]
                gradients.append(gradient)
            else:
                gradient = -1
            print(auto_count, function_number, weight_number, step_size, gradient, correct_ratio, all_weights, purchase_weights)
            if last_prob > correct_ratio:
                last_prob = correct_ratio
                break_count = 0
                odds_improved = 1
                if gradients[-1] < gradients[-2]:
                    step_sizes[function_number][weight_number] = step_sizes[function_number][weight_number]
                    auto_count += 0.01
                else:
                    auto_count -= 0.2
            print(correct_ratio, last_prob)
            if correct_ratio > last_prob or 0 > gradients[-1]:
                break_count += 1
                odds_improved = 0
                auto_count = swing_limit
                break
            else:
                auto_count += 1
            if odds_improved == 1 and weight_number != 5:
                all_weights[function_number][weight_number] = weight + step_size
            #  find a cleaner solution to the below
        if auto_count >= swing_limit:
            swing_limit += 1
            step_sizes[function_number][weight_number] = step_sizes[function_number][weight_number] * -1.1
            #print("Step size increased to " + str(step_size))
        weight_number += 1
        if weight_number > 5:
            weight_number = 0
            function_number += 1
        if function_number > 2:
            function_number = 0

        #print("current correct guesses in training (overfitted):", str(correct_ratio), last_prob, "weights: ",
              #str(weight_frequency), weight_lag, weight_length, weight_recency, meta_w)
    return weight_length, weight_lag, weight_frequency, weight_recency, purchase_weights, meta_w


total_right, total_wrong = 1, 1
one_count, two_count = 0, 0
training_lim = 40
testing_lim = 88
dyn_length, dyn_frq, dyn_lag, dyn_recency, dyn_meta = 1, 1, 1, 1, 1
total_wrong = 0
total_right = 1
losses_testing = []
rows = []
total_thresholds = []
with open("bigger_list.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)

customer_count = 0
for row in rows:
    found_complements = []
    weight_complements= []
    if len(row) > 1 and row[2]:
        print("Customer: ", row[0], "Product number: ", row[1])
        customer = int(row[0])
        """
        combs = find_combos(rows, customer)
        produce = find_complements(combs, customer)
        for pro in produce:
            if str(row[1]) == str(pro[0]):
                found_complements.append([pro[-2], pro[-1]])
        for com in found_complements:
            index = (int(customer) * 500) + (com[0]*2) - 1
            weight_complements.append([rows[index][2:], pro[-1]])
        print(weight_complements)
        print("found")"""

        row = row[2:]
        guesses, live_list = [], [],
        probability_list = []
        losses_product = []
        training_data = row[0:training_lim]
        testing_data = row[training_lim:testing_lim]
        if (row.count("2") / (row.count("2") + row.count("1"))) > 0.1:
            print("Proportion of purchases in training data:",
                  training_data.count("1") / (training_data.count("2") + training_data.count("1")))
            print("Proportion of purchases in testing data:",
                  testing_data.count("1") / (testing_data.count("2") + testing_data.count("1")))
            tester = []
            for num in training_data:
                num = int(num)
                guesses.extend(str(num))
            for num in testing_data:
                num = int(num)
                tester.extend(str(num))
                guesses.extend(str(num))

            threshold_accuracy = log_loss(max(guesses.count("1")/(guesses.count("1") + guesses.count("2")),
                                               guesses.count("2")/(guesses.count("1") + guesses.count("2"))), 2)
            total_thresholds.append(threshold_accuracy)

            correct_guesses, incorrect_guesses, newcount = 0, 0, 0
            simul_right = 0.001
            simul_wrong = 0
            for h in range(training_lim, testing_lim-1):
                learned = learn(guesses[0:h], dyn_length, dyn_lag, dyn_frq, dyn_recency, dyn_meta, threshold_accuracy)
                predicted_number = Get_Probabilities(guesses[0:h], learned[0], learned[1], learned[2],
                                                     learned[3], learned[4], learned[5])
                dyn_length, dyn_frq, dyn_lag, dyn_recency, dyn_meta = learned[0], learned[1], learned[2], \
                                                                      learned[3], learned[5]

                if predicted_number is not None:  # Checks if simulation predicted the next value
                    print(predicted_number, int(guesses[h+1])-1, h)
                    testing_loss = log_loss(predicted_number, int(guesses[h+1]))
                    losses_testing.append(testing_loss)
                    losses_product.append(testing_loss)
                    if int(guesses[h + 1]) == 1:
                        one_count += 1
                    else:
                        two_count += 1
                loss_product = sum(losses_product)/len(losses_product)
                loss_testing = sum(losses_testing)/len(losses_testing)
            print(loss_product, "log_loss for weeks: ", training_lim, " to ", testing_lim)
            print(loss_testing, "log_loss correctly for ", one_count+two_count,
                  " predictions in testing so far, baseline ratio:", sum(total_thresholds)/len(total_thresholds))

