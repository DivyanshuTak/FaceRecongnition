EQUAL=1
convol_sum=0


#===================================================================
#                  THIS FILE USES CONVOLUTION
#            TO FIND SIMILARITY BETWEEN TWO FEATURES
#====================================================================
#    FIRST COMPARE THE TWO FEATURES ARE EQUAL OR NOT
#====================================================================

def match_func(ip_list_1,ip_list_2):
    convol_result = []
    if len(ip_list_1) == len(ip_list_2):
        EQUAL = 1
        a = 0
        while a < len(ip_list_2):
            temp_mul = ip_list_1[a] * ip_list_2[a]
            convol_result.append(temp_mul)
            a += 1
    else:
        raise print("length of features dont match")
        EQUAL = 0
    print("fuck this shit")
    return convol_result
