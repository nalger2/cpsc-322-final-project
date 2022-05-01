#downsizing data
import importlib
import numpy as np
import mysklearn.mypytable
import os
import copy

from mysklearn import myutils
from mysklearn.mypytable import MyPyTable



def clean_data(random_seed_val, original_filename):
    np.random.seed(random_seed_val)

    filename = original_filename
    table = MyPyTable().load_from_file(filename)
    print("length before 'NA' values removed:", len(table.data))

    table.remove_rows_with_missing_values()
    print("length after 'NA' values removed:", len(table.data))

    # creating downsample data
    table_deep_copy = copy.deepcopy(table)


    non_stroke = []
    stroke_data = []
    for row in table.data:
        if row[-1] == 0.0:
            non_stroke.append(row)
        if row[-1] == 1.0:
            stroke_data.append(row)
    print("amount of non-strokes with no 'NA' rows:", len(non_stroke))
    print("amount of strokes with no 'NA' rows:", len(stroke_data))

    unknown_count = 0
    for row in stroke_data:
        if row[10] == "Unknown" or row[10] == "unknown":
            unknown_count += 1
    #print("num of strokes with unknown smoking status:", unknown_count)

    downsized_non_stroke_data = []
    for i in range(0, 1000 - len(stroke_data)):
        index = np.random.randint(0, len(non_stroke))
        row = non_stroke[index]
        downsized_non_stroke_data.append(row)
        non_stroke.remove(row)

    print("length of downsized non-stroke data:", len(downsized_non_stroke_data))

    data_downsized = stroke_data + downsized_non_stroke_data # adding 1.0 class label and downsized 0.0 sample

    print("length of all downsized data:", len(data_downsized)) # 1000 0 class labels, 209 1 class labels (after removal of 'NA' rows)

    final_data = table_deep_copy
    final_data.data = data_downsized
    #print("length of final_data", len(final_data.data))

    final_data.save_to_file("input_data/stroke-data-downsized.csv")

    print("-----SAVED DOWNSIZED DATA-----")
    stroke_count = 0
    non_stroke_count = 0
    for row in final_data.data:
        if row[-1] == 0.0:
            non_stroke_count += 1
        elif row[-1] == 1.0:
            stroke_count += 1

    #print("non-stroke amount:", non_stroke_count)
    #print("stroke amount:", stroke_count)

    # checking if theres any duplicate rows # can delete loop
    for i in range(len(final_data.data)): 
        row = final_data.data[i]
        for j in range(len(final_data.data)):
            if row == final_data.data[j] and j != i:
                print("same")

    #FINISH CLEANING IN MYPYTABLE FORM
    stroke_data = MyPyTable()
    stroke_data.load_from_file("input_data/stroke-data-downsized.csv")

    #clean stroke data for classification- discretize, convert nominal to numeric
    stroke_data_discretized = myutils.discretize_attributes_for_stroke_classification(stroke_data)
    stroke_data_discretized.save_to_file("input_data/stroke-data-discretized.csv")
    print("-----saved discretized columns-----")
    #strings to numeric
    stroke_data_cleaned_numeric = myutils.numerize_all_strings(stroke_data_discretized)
    stroke_data_cleaned_numeric.save_to_file("input_data/stroke-data-all-attributes-cleaned.csv")
    print("-----saved numerical final data-----")

    #print("len before removing missing vals", len(stroke_data.data))
    #code to remove rows w/missing values (use only once relevant attributes are decided)
    #stroke_data.remove_rows_with_missing_values()
    #stroke_data.remove_rows_with_missing_values_by_col()
    #print("len after removing missing vals", len(stroke_data.data))

    #ATTRIBUTE SELECTION
    data_for_attribute_selection = copy.deepcopy(stroke_data_cleaned_numeric)
    #0gender,1age,2hypertension,3heart_disease,4ever_married,5work_type,6Residence_type,7avg_glucose_level,8bmi,9smoking_status,10stroke
    #KEEP: 0, 1, 3, 8, 9, 10 REMOVE: 2, 4, 5, 6, 7
    data_for_attribute_selection.remove_columns([2, 4, 5, 6, 7])
    data_for_attribute_selection.save_to_file("input_data/stroke_data_atts_selected.csv")
    print("-----attribute selection complete & saved-----")