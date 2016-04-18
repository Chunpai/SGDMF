import numpy as np
import math

def readR():
    """
    read training data, and store in dictionray
    """       
    infile =open("ratings_train.txt","r")
    R_dict = {}
    user_list = []
    movie_list = []
    count = 0
    for line in infile:
        fields = line.strip().split("\t")
        user_id = int(fields[0])
        movie_id = int(fields[1])
        if user_id not in R_dict:
            R_dict[user_id] = {}
            R_dict[user_id][movie_id] = float(fields[2])
        else:
            R_dict[user_id][movie_id] = float(fields[2])
        
        if user_id not in user_list:
            user_list.append(user_id)
        if movie_id not in movie_list:
            movie_list.append(movie_id)
    #m = len(movie_list)
    #n = len(user_list)
    #print m,n
    infile.close()
    user_list.sort()
    movie_list.sort()
    m = movie_list[-1] 
    n = user_list[-1]
    print m,n
    return R_dict, m, n


def initialization(m,n,k):
    fraction = 1.0 / math.sqrt(5.0/k)
    Q = np.random.rand(m,k) /fraction
    P = np.random.rand(n,k) /fraction
    #print Q,P
    return Q, P

def sgdmf(R_dict, Q,P, k, reg=0.2, rate=0.03, iterations = 40):
    P = P.T
    E_list = []
    old_E = 10000000
    for step in range(iterations):
        for user_id in R_dict:
            for movie_id in R_dict[user_id]:
                rating = R_dict[user_id][movie_id]
                error = rating - np.dot(Q[movie_id-1,:],P[:,user_id-1])
                for index in range(k):
                    Q[movie_id-1][index] =  Q[movie_id-1][index] + rate * (2*error*P[index][user_id-1]- reg*Q[movie_id-1][index])
                    P[index][user_id-1] = P[index][user_id-1] + rate * (2*error*Q[movie_id-1][index] - reg*P[index][user_id-1])
        eR = np.dot(Q,P)  #recovered R

        E = 0
        for user_id in R_dict:
            for movie_id in R_dict[user_id]:
                E = E + pow(R_dict[user_id][movie_id] - eR[movie_id-1][user_id-1],2)
                for index in range(k):
                    E = E + reg * (pow(Q[movie_id-1][index],2)+ pow(P[index][user_id-1],2))
        if abs(old_E - E) < 1 or E < 1:
            break
        else:
            #print "E", E
            E_list.append(E)
            old_E = E
    #print E_list
    return Q,P.T


#R_test is a dictionary
def test(Q,P):
    infile =open("ratings_val.txt","r")
    R_test = {}
    for line in infile:
        fields = line.strip().split("\t")
        user_id = int(fields[0])
        movie_id = int(fields[1])
        if user_id not in R_test:
            R_test[user_id] = {}
            R_test[user_id][movie_id] = float(fields[2])
        else:
            R_test[user_id][movie_id] = float(fields[2])
    
    infile.close()
    
    E = 0
    P =  P.T
    eR = np.dot(Q,P)
    for user_id in R_test:
        for movie_id in R_test[user_id]:
            E = E + pow(R_test[user_id][movie_id] - eR[movie_id-1][user_id-1],2) 
    print "test error", E



if __name__ == "__main__":
    R_dict, m, n = readR()
    k = 30
    Q, P = initialization(m,n,k)
    Q, P = sgdmf(R_dict,Q,P,k)
    #print Q,P
    test(Q,P)
