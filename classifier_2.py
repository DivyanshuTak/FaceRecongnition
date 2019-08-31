import database
import weighted_avg
import classifier_1
import _thread



#=============================================================
#           THIS FILE RUNS PARALLEL THREADS AND
#           CALCULATES THE CORELATION COEFF AND
#           COMPANRES IT WITH AUTO CORELATION COEF
#           THEN DECIDES THE CLOEST FEATURE
#=============================================================

target_feature = [1,1,1,1]
auto_corelation = weighted_avg.weight_avg(target_feature, target_feature)
print("autocorelation coeff if ",auto_corelation)
#===============================================================

coeff_list = []
coeff_list.append(classifier_1.match_relation())
coeff_list.append(classifier_1.match_relation())
coeff_list.append(classifier_1.match_relation())

#try:
 #   _thread.start_new_thread(coeff_list.append(classifier_1.match_relation()))
 #   _thread.start_new_thread(coeff_list.append(classifier_1.match_relation()))
 #   _thread.start_new_thread(coeff_list.append(classifier_1.match_relation()))
#except:
 #   print("problem in starting threads")

print("the cross coeff list is::",coeff_list)

#========================================================
#               COMPARE THE CROSS COEFF
#               WITH AUTO CORELATION COEFF
#               THE NEAREST ONE IS THE ANSWER
#=========================================================

diff_list = []
for a in range(len(coeff_list)):
    diff_list.append(coeff_list[a] - auto_corelation)

matched_coeff=0
a=0
b=0
temp=0
for a in range(len(diff_list)):
    for b in range(len(diff_list)):
        if diff_list[a] < diff_list[b]:
            temp+=1
    if(temp == len(diff_list) - 1):
        matched_coeff = coeff_list[a]
        a=10
    temp=0

print("closest coeff is ",matched_coeff)























