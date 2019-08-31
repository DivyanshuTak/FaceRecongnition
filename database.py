#==================================================
#        THIS IS THE DATABASE FILE
#        IMPORT THIS FILE IN THE MAIN
#        RUNNING FILE
#        TAKE THE FEATURES FROM IT
#        THEN PASS THEM ASS ARGUMENTS IN DIFFERENT
#        FUNCTION
#====================================================

#====================================================
#         FOR DYNAMIC DISPATCHING
#         OF THE DATA
#         USE FILE I/O FILE MODULE
#         TO TAKE THE SAMPLE ANS
#         STORE IT IN AN ARRAY
#=====================================================

set_1 = {'feature_1':[1,1,1,1],'feature_2':[1,1,2,1],'feature_3':[3,2,4,6]}
#----------------------------------------------------------
#           FILE OPEN
#           FOR FEATURE DATABASE
#
#===========================================================

file_1 = open('database.txt',"r+")
for a in range(len(set_1['feature_1'])):
    data = set_1['feature_1']
    file_1.write(str(data[a]))



file_1.write(":")

for a in range(len(set_1['feature_2'])):
    data = set_1['feature_2']
    file_1.write(str(data[a]))

file_1.write(":")

for a in range(len(set_1['feature_3'])):
    data = set_1['feature_3']
    file_1.write(str(data[a]))









