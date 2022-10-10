import csv
import math


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


def Get_Probabilities(input_list, weight_length, weight_lag, weight_frequency,
                      weight_recency, weights_ngram, meta_weight):
    guesses_in = input_list
    probability_list = []
    processed_data = GetSequences(guesses_in)
    setz = sorted(set(guesses_in))
    setz = list(setz)
    for z in setz:
        probability_list.append([z])
    probability_list = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    for sub_list in processed_data:
        counter = 0
        if len(sub_list) > 1:  # Generates a likelyhood predictions score for the next number
            length_multiplier = len(sub_list[0]) * weight_length
            lag = int(sub_list[-1])
            lag_weight = weight_lag * (lag+1)
            next_number = int(sub_list[-3])
            frequency_bias = int(guesses_in.count(str(next_number))) * weight_frequency
            recencey_bias = int(sub_list[-2]) * weight_recency
            result_matches = ((frequency_bias) + (recencey_bias) + (lag_weight) + (length_multiplier)) \
                             * (weights_ngram[-sub_list[-2]-1])
            if probability_list[next_number-1] == 2:
                result_matches = result_matches * meta_weight
            probability_list[next_number-1].append(result_matches)
            counter += 1
    prob_outcomes = []
    countr = 0
    if len(guesses_in) > 0 and len(processed_data) > 0:
        for h in probability_list:
            if len(h) > 1:
                prob_outcomes.append([countr+1,sum(h[1:])])
            countr += 1

        idx, max_value = max(prob_outcomes, key=lambda item: item[-1])
        idx2, min_value = min(prob_outcomes, key=lambda item: item[-1])
        return int(idx)


def learn(sequence, init_length, init_frq, iinit_lag, init_recency, init_meta, threshold_accuracy):
    learned_list = []  # this list stores the tested weights and their accuracy
    purchase_weights = []
    for k in range(89):
        purchase_weights.append(1)
    weight_length, weight_frequency, weight_lag, weight_recency, meta_w = init_length, init_frq, iinit_lag, \
                                                                          init_recency, init_meta # starting values of weights
    accuracy_list = []
    correct_ratio = 0
    weight_number = 0
    step_size = 1
    swing_limit = 10
    counts = 0
    break_count = 0
    no_improvement_limit = 500
    iteration_limit = 5000
    weight_limit = 1400
    while abs(weight_length) + abs(weight_lag) + abs(weight_frequency) + abs(weight_recency) + abs(meta_w) \
            < weight_limit \
            and counts < iteration_limit\
            and break_count < no_improvement_limit\
            and correct_ratio <= 0.95:
        auto_count = 0
        switch = 0
        last_prob = 0
        weight_adder = 0
        weight_list = [weight_length, weight_lag, weight_frequency, weight_recency, meta_w]
        while 0 <= auto_count <= swing_limit:
            counts += 1
            odds_improved = 0
            weight = weight_list[weight_number]
            if auto_count > 0 or switch == 1:
                weight_adder += step_size
            simul_right = 1
            simul_wrong = 1
            cnt_loop = 0
            for n in range(5, len(sequence) - 1): # Compare the forecast based on guesses 0:n to the actual outcome
                if weight_number == 0:
                    predicted_number = Get_Probabilities(sequence[0:n], weight + weight_adder, weight_lag, weight_frequency,
                                                         weight_recency, purchase_weights, meta_w)
                if weight_number == 1:
                    predicted_number = Get_Probabilities(sequence[0:n], weight_length, weight + weight_adder, weight_frequency,
                                                         weight_recency, purchase_weights, meta_w)
                if weight_number == 2:
                    predicted_number = Get_Probabilities(sequence[0:n], weight_length, weight_lag, weight + weight_adder,
                                                         weight_recency, purchase_weights, meta_w)
                if weight_number == 3:
                    predicted_number = Get_Probabilities(sequence[0:n], weight_length, weight_lag, weight_frequency,
                                                         weight + weight_adder, purchase_weights, meta_w)
                if weight_number == 4:
                    predicted_number = Get_Probabilities(sequence[0:n], weight_length, weight_lag, weight_frequency,
                                                         weight + weight_adder, purchase_weights, meta_w + weight_adder)
                if predicted_number is not None:  # Checks if simulation predicted the next value
                    if int(predicted_number) == int(sequence[n+1]):
                        simul_right += 1
                        purchase_weights[cnt_loop] = abs(purchase_weights[cnt_loop]) * 0.99
                    else:
                        simul_wrong += 1
                        purchase_weights[cnt_loop] = abs(purchase_weights[cnt_loop]) * 1.01
                correct_ratio = simul_right / (simul_wrong + simul_right)
                accuracy_list.append(correct_ratio)
                cnt_loop += 1
            learned_list.append([correct_ratio, weight])

            if max(accuracy_list) < correct_ratio:
                break_count = 0
            else:
                break_count += 1

            if last_prob < correct_ratio:
                last_prob = float(learned_list[-1][0])
                auto_count = 1
                odds_improved = 1
            elif correct_ratio < last_prob:
                switch += 1
                step_size = -1*step_size
                if switch > 1:
                    switch = 0
                    auto_count = -1
            else:
                break_count += 1
                auto_count += 1
            if odds_improved == 1 or auto_count == 9:
                if weight_number == 0:
                    weight_length = weight + weight_adder
                if weight_number == 1:
                    weight_lag = weight + weight_adder
                if weight_number == 2:
                    weight_frequency = weight + weight_adder
                if weight_number == 3:
                    weight_recency = weight + weight_adder
                elif weight_number == 4:
                    meta_w = weight + weight_adder
                weight_list = [weight_length, weight_lag, weight_frequency, weight_recency, meta_w]

            # print(learned_list[-1], last_prob, weight_numberweight)

            #  find a cleaner solution to the below
        weight_number += 1
        if auto_count > swing_limit:
            swing_limit += 1
            step_size = step_size * -1.1
            #print("Step size increased to " + str(step_size))
        if weight_number > 4:
            weight_number = 0

        #print("current correct guesses in training (overfitted):", str(correct_ratio), last_prob, "weights: ",
              #str(weight_frequency), weight_lag, weight_length, weight_recency)
    return weight_length, weight_lag, weight_frequency, weight_recency, purchase_weights, meta_w


total_right, total_wrong = 1, 1
one_count, two_count = 0, 0
training_lim = 40
testing_lim = 80
dyn_length, dyn_frq, dyn_lag, dyn_recency, dyn_meta = 1, 1, 0.1, 1, 1
total_wrong = 0
total_right = 1
rows = []
with open("customers_long.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        rows.append(row)
for row in rows:
    row = row[1:]
    if len(row) > 1:
        print("Product number: ", row[0])
        guesses, live_list = [], [],
        probability_list = []
        training_data = row[0:training_lim]
        testing_data = row[training_lim:testing_lim]
        if 0.5 > (row.count("1") / (row.count("2") + row.count("1"))) > 0.4:
            print("Proportion of purchases in training data:",
                  training_data.count("1") / (training_data.count("2") + training_data.count("1")))
            print("Proportion of purchases in testing data:",
                  testing_data.count("1") / (testing_data.count("2") + testing_data.count("1")))
            tester = []
            import time
            for num in training_data:
                num = int(num)
                guesses.extend(str(num))
            for num in testing_data:
                num = int(num)
                tester.extend(str(num))
                guesses.extend(str(num))
            threshold_accuracy = max(guesses.count("1")/(guesses.count("1")+guesses.count("2")),
                                      (guesses.count("2")/(guesses.count("1")+guesses.count("2"))))

            correct_guesses, incorrect_guesses, newcount = 0, 0, 0
            simul_right = 0.001
            simul_wrong = 0
            for h in range(training_lim, testing_lim-1):
                learned = learn(guesses[0:h], dyn_length, dyn_frq, dyn_lag, dyn_recency, dyn_meta, threshold_accuracy)
                predicted_number = Get_Probabilities(guesses[0:h], learned[0], learned[1], learned[2],
                                                     learned[3], learned[4], learned[5])
                dyn_length, dyn_frq, dyn_lag, dyn_recency, dyn_meta = learned[0], learned[1], learned[2], \
                                                                      learned[3], learned[5]

                if predicted_number is not None:  # Checks if simulation predicted the next value
                    if int(predicted_number) == int(guesses[h + 1]):
                        simul_right += 1
                        total_right += 1
                    else:
                        simul_wrong += 1
                        total_wrong += 1
                    if int(guesses[h + 1]) == 1:
                        one_count += 1
                    else:
                        two_count += 1
                print(h, predicted_number, guesses[h+1])
                correct_ratio = simul_right / (simul_wrong + simul_right)
                total_correct_ratio = total_right / (total_wrong + total_right)
            print(correct_ratio * 100, "% guessed correctly for week 70-89")
            print(total_correct_ratio * 100, "% guessed correctly for ", total_wrong+total_right,
                  " predictions in testing so far, baseline ratio:", one_count/(one_count+two_count))

