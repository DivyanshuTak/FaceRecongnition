import database
import weighted_avg

#==============================================================                                                         # for reading from a file first
#    ALWAYS INCREASE THE POINTER BY NUMBER OF                                                                           # make a open read object of that file
#    FEATURES IN THE ARRAY example:4                                                                                    # then call object.read function
#
#
#
#================================================================


#================================================================
file_1_obj = open('database.txt','r')
target_feature = [1,1,1,1]
file_1_obj.seek(0, 0)
def match_relation():
    a = 0
    dummy_list = []
    for a in range(len(target_feature)):
        dummy_list.append(int(file_1_obj.read(1)))
    val = file_1_obj.tell()
    file_1_obj.seek(val+1,0)
    return weighted_avg.weight_avg(target_feature, dummy_list)                                                          # explicit type convesion is
                                                                                                                        # necessary for further process
#print(result)