import itertools
import pandas as pd

from Client.first_perturbation import first_perturbation
from Client.second_perturbation import second_perturbation
from Server.PrivBayes import PrivBayes
from Server.clustering import clustering
from Server.PRAM import PRAM
from Experimental_Evzluations.correlation import correlation
from Experimental_Evzluations.TVD import TVD


import pandas as pd
import itertools



def read_dataset():
        input_domain='dataset/Data14-coarse.domain'
        input_data='dataset/Data14-coarse.dat'

        fd=open(input_domain,'r')
        head_line=fd.readline()
        readrow=head_line.split(" ")
        att_num=int(readrow[0])
        fd.close()

        ##########################################################################################
        #Get the attributes info in the domain file
        multilist = []
        fp=open(input_data,"r")
        fp.readline();  ###just for skip the header line


        while 1:
            line = fp.readline()
            if not line:
                break

            line=line.strip("\n")
            temp=line.split(",")
            temp2 = [str(i)+'a' for i in temp]
            multilist.append(temp2)
        fp.close()

        colomn = [str(i)+'a' for i in range(att_num)]
        df = pd.DataFrame(multilist, columns = colomn)
        return df



def attributes_domain():
        input_domain='dataset/Data14-coarse.domain'
        fd=open(input_domain,'r')
        head_line=fd.readline()
        readrow=head_line.split(" ")

        domain = dict()
        i=0
        while 1:
            line = fd.readline()
            if not line:
                break

            line=line.strip("\n")
            readrow=line.split(" ")
            start_x=0
            dom = []
            for eachit in readrow:
                start_x=start_x+1
                eachit.rstrip()
                if start_x>3:
                    dom.append(str(eachit)+'a')
            domain.update({str(i)+'a':dom})
            i=i+1
        fd.close()
        return domain



if __name__ == "__main__":
        dataset = read_dataset()
        domains = attributes_domain()
        _total_tvd_alpha_3_SP = 0
        _total_tvd_alpha_3_PRAM = 0
        _total_tvd_alpha_3_FE = 0
        _total_tvd_alpha_3_PRAM_FE = 0
        _total_tvd_alpha_3_MLE1 = 0

        _total_tvd_alpha_4_SP = 0
        _total_tvd_alpha_4_PRAM = 0
        _total_tvd_alpha_4_FE = 0
        _total_tvd_alpha_4_PRAM_FE = 0
        _total_tvd_alpha_4_MLE1 = 0

        _total_SP_correlation = 0
        _total_PRAM_correlation = 0
        _total_FE_correlation = 0
        _total_PRAM_FE_correlation = 0
        _total_MLE1_correlation = 0

        ## epsilon for each level is 1 but the total is 2
        total_epsilon = 2
        epsilon = total_epsilon/2
        print("epsilon : "+str(total_epsilon))
        
        for j in range(20):
            print("Loop number : "+str(j+1))

            datasett = dataset.copy(deep=True)
            clients_dataset = dataset.copy(deep=True)
            clients_dataset2 = dataset.copy(deep=True)


            clients_dataset_list = []
            for index, row in clients_dataset.iterrows():
                    clients_dataset_list.append(row.to_dict())

            clients = []
            for data in clients_dataset_list:
                client = first_perturbation(epsilon, domains, data)
                clients.append(client.randomized_data)
            FP = pd.DataFrame(clients, columns = dataset.columns.to_list())
            FP.to_csv("firstPerturbation_{0}_{1}.csv".format(total_epsilon,j),index=False)
            

            # constructing bayesian network  (k = 2)
            privBayes = PrivBayes(FP)

            # constructing clusters of size 3
            # len3 = False
            # while not len3:
            clu = clustering(privBayes.BN, domains, FP)
                # len3 = all(len(sublist) < 4 for sublist in clu.clusters)
            clients_dataset_list2 = []

            clients2 = []

            client = second_perturbation(epsilon, domains, clients_dataset2, clu.clusters, clu.PBC)
            clients2 = client.randomized_data

            corr = correlation(datasett,clients2)
            print(corr.MAE)
            print("correlation SP : "+ str(corr.MAE))
            _total_SP_correlation += corr.MAE

            ### alpha = 3
            _tvd_alpha_3_1 = 0
            subsets_of_size_3_1 = list(itertools.combinations(domains.keys(), 3))
            for subset in subsets_of_size_3_1:
                _tvd_alpha_3_1 += TVD(dataset[list(subset)], clients2[list(subset)] , domains).tvd

            _tvd_alpha_3_average_1 = _tvd_alpha_3_1 / len(subsets_of_size_3_1)
            _total_tvd_alpha_3_SP  += _tvd_alpha_3_average_1
            #######################


            #### alpha = 4
            _tvd_alpha_4_1 = 0
            subsets_of_size_4_1 = list(itertools.combinations(domains.keys(), 4))
            for subset in subsets_of_size_4_1:
                _tvd_alpha_4_1 += TVD(dataset[list(subset)], clients2[list(subset)] , domains).tvd

            _tvd_alpha_4_average_1 = _tvd_alpha_4_1 / len(subsets_of_size_4_1)
            _total_tvd_alpha_4_SP += _tvd_alpha_4_average_1
            ########################

            print("alpah = 3 SP .... total variation distance1 : "+ str(_tvd_alpha_3_average_1))

            print("alpah = 4 SP .... total variation distance1 : "+ str(_tvd_alpha_4_average_1))
            print()
            # invariant PRAM
            P = PRAM(epsilon, clu.clusters, clu.PBC, domains, clients2)
          
            corr1 = correlation(datasett,P.randomized_data)
            print("correlation PRAM : "+ str(corr1.MAE))
            _total_PRAM_correlation += corr1.MAE

            corr2 = correlation(datasett,P.randomized_data2)
            print("correlation FE : "+ str(corr2.MAE))
            _total_FE_correlation += corr2.MAE

            corr3 = correlation(datasett,P.randomized_data3)
            print("correlation PRAM & FE : "+ str(corr3.MAE))
            _total_PRAM_FE_correlation += corr3.MAE

            corr4 = correlation(datasett,P.randomized_data4)
            print("correlation MLE1 : "+ str(corr4.MAE))
            _total_MLE1_correlation += corr4.MAE


            #### alpha = 3
            _tvd_alpha_3_PRAM = 0
            _tvd_alpha_3_FE = 0
            _tvd_alpha_3_PRAM_FE = 0
            _tvd_alpha_3_MLE1 = 0
            subsets_of_size_3_2 = list(itertools.combinations(domains.keys(), 3))
            for subset in subsets_of_size_3_2:
                _tvd_alpha_3_PRAM += TVD(dataset[list(subset)], P.randomized_data[list(subset)] , domains).tvd
                _tvd_alpha_3_FE += TVD(dataset[list(subset)], P.randomized_data2[list(subset)] , domains).tvd
                _tvd_alpha_3_PRAM_FE += TVD(dataset[list(subset)], P.randomized_data3[list(subset)] , domains).tvd
                _tvd_alpha_3_MLE1 += TVD(dataset[list(subset)], P.randomized_data4[list(subset)] , domains).tvd


            _tvd_alpha_3_average_PRAM = _tvd_alpha_3_PRAM / len(subsets_of_size_3_2)
            _tvd_alpha_3_average_FE = _tvd_alpha_3_FE / len(subsets_of_size_3_2)
            _tvd_alpha_3_average_PRAM_FE = _tvd_alpha_3_PRAM_FE / len(subsets_of_size_3_2)
            _tvd_alpha_3_average_MLE1 = _tvd_alpha_3_MLE1 / len(subsets_of_size_3_2)
            _total_tvd_alpha_3_PRAM +=_tvd_alpha_3_average_PRAM
            _total_tvd_alpha_3_FE  += _tvd_alpha_3_average_FE
            _total_tvd_alpha_3_PRAM_FE += _tvd_alpha_3_average_PRAM_FE
            _total_tvd_alpha_3_MLE1 += _tvd_alpha_3_average_MLE1
            #######################


            #### alpha = 4
            _tvd_alpha_4_PRAM = 0
            _tvd_alpha_4_FE = 0
            _tvd_alpha_4_PRAM_FE = 0
            _tvd_alpha_4_MLE1 = 0
            subsets_of_size_4_2 = list(itertools.combinations(domains.keys(), 4))
            for subset in subsets_of_size_4_2:
                _tvd_alpha_4_PRAM += TVD(dataset[list(subset)], P.randomized_data[list(subset)] , domains).tvd
                _tvd_alpha_4_FE += TVD(dataset[list(subset)], P.randomized_data2[list(subset)] , domains).tvd
                _tvd_alpha_4_PRAM_FE += TVD(dataset[list(subset)], P.randomized_data3[list(subset)] , domains).tvd
                _tvd_alpha_4_MLE1 += TVD(dataset[list(subset)], P.randomized_data4[list(subset)] , domains).tvd


            _tvd_alpha_4_average_PRAM = _tvd_alpha_4_PRAM / len(subsets_of_size_4_2)
            _tvd_alpha_4_average_FE = _tvd_alpha_4_FE / len(subsets_of_size_4_2)
            _tvd_alpha_4_average_PRAM_FE = _tvd_alpha_4_PRAM_FE / len(subsets_of_size_4_2)
            _tvd_alpha_4_average_MLE1 = _tvd_alpha_4_MLE1 / len(subsets_of_size_4_2)
            _total_tvd_alpha_4_PRAM +=_tvd_alpha_4_average_PRAM
            _total_tvd_alpha_4_FE  += _tvd_alpha_4_average_FE
            _total_tvd_alpha_4_PRAM_FE += _tvd_alpha_4_average_PRAM_FE
            _total_tvd_alpha_4_MLE1 += _tvd_alpha_4_average_MLE1
            #######################


            print("alpah = 3 PRAM.... total variation distance : "+ str(_tvd_alpha_3_average_PRAM))
            print("alpah = 3 FE .... total variation distance : "+ str(_tvd_alpha_3_average_FE))
            print("alpah = 3 PRAM & FE .... total variation distance : "+ str(_tvd_alpha_3_average_PRAM_FE))
            print("alpah = 3 MLE1.... total variation distance : "+ str(_tvd_alpha_3_average_MLE1))
            print()
            print("alpah = 4 PRAM.... total variation distance : "+ str(_tvd_alpha_4_average_PRAM))
            print("alpah = 4 FE .... total variation distance : "+ str(_tvd_alpha_4_average_FE))
            print("alpah = 4 PRAM & FE .... total variation distance : "+ str(_tvd_alpha_4_average_PRAM_FE))
            print("alpah = 4 MLE1.... total variation distance : "+ str(_tvd_alpha_4_average_MLE1))
            print()
            


        _tvdd_3_SP = _total_tvd_alpha_3_SP / 20
        _tvdd_3_PRAM = _total_tvd_alpha_3_PRAM / 20
        _tvdd_3_FE = _total_tvd_alpha_3_FE / 20
        _tvdd_3_PRAM_FE = _total_tvd_alpha_3_PRAM_FE / 20
        _tvdd_3_MLE1 = _total_tvd_alpha_3_MLE1 / 20


        _tvdd_4_SP = _total_tvd_alpha_4_SP / 20
        _tvdd_4_PRAM = _total_tvd_alpha_4_PRAM / 20
        _tvdd_4_FE = _total_tvd_alpha_4_FE / 20
        _tvdd_4_PRAM_FE = _total_tvd_alpha_4_PRAM_FE / 20
        _tvdd_4_MLE1 = _total_tvd_alpha_4_MLE1 / 20

        total_correlation_SP = _total_SP_correlation/20
        total_correlation_PRAM = _total_PRAM_correlation/20
        total_correlation_FE = _total_FE_correlation/20
        total_correlation_PRAM_FE = _total_PRAM_FE_correlation/20
        total_correlation_MLE1 = _total_MLE1_correlation/20


        print("SP.... average correlation : "+ str(total_correlation_SP))
        print("PRAM.... average correlation : "+ str(total_correlation_PRAM))
        print("FE.... average correlation : "+ str(total_correlation_FE))
        print("PRAM & FE.... average correlation : "+ str(total_correlation_PRAM_FE))
        print("MLE1.... average correlation : "+ str(total_correlation_MLE1))




        print("alpha = 3 SP.... average total variation distance : "+ str(_tvdd_3_SP))
        print("alpha = 3 PRAM.... average total variation distance : "+ str(_tvdd_3_PRAM))
        print("alpha = 3 FE.... average total variation distance : "+ str(_tvdd_3_FE))
        print("alpha = 3 PRAM & FE .... average total variation distance : "+ str(_tvdd_3_PRAM_FE))
        print("alpha = 3 MLE1 .... average total variation distance : "+ str(_tvdd_3_MLE1)) 




        print("alpha = 4 SP.... average total variation distance : "+ str(_tvdd_4_SP))
        print("alpha = 4 PRAM.... average total variation distance : "+ str(_tvdd_4_PRAM))
        print("alpha = 4 FE.... average total variation distance : "+ str(_tvdd_4_FE))
        print("alpha = 4 PRAM & FE .... average total variation distance : "+ str(_tvdd_4_PRAM_FE))
        print("alpha = 4 MLE1 .... average total variation distance : "+ str(_tvdd_4_MLE1))


        

