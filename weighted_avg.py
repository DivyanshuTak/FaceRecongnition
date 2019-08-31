import correlation_general

#==============================================================
#                 THIS FILE TAKES THE GIVEN
#                 TWO LISTS AND THEN CALLS
#                 CORRELATION FUNCTION AND THE CALCULATES
#                 THE WEIGHTED AVERAGE OF THE RESULT MATRIX
#=                THIS WEIGHTED AVE. IS THEN USED IN CLASSIFIER
#===============================================================

def weight_avg(list_1,list_2):
    a=1
    average=0
    result_array = correlation_general.match_func(list_1, list_2)
    while a < len(result_array):
        temp_avg = 1*result_array[a]                                                # weight is removed
        average = average + temp_avg
        a+=1

    return average



