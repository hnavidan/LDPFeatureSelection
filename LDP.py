import numpy as np
import math
from pure_ldp.frequency_oracles import *

def privatise_olh(epsilon, data, Y):

    olh_estimates = [None]*len(data)
    num_of_classes = len(Y.unique())

    for i in range(len(data)):
        d = len(np.unique(data[i]))

        client_olh = LHClient(epsilon=epsilon, d=d, use_olh=True)
        server_olh = LHServer(epsilon=epsilon, d=d, use_olh=True)

        priv_olh_data = [client_olh.privatise(item) for item in data[i]]

        server_olh.aggregate_all(priv_olh_data)
        olh_estimates[i] = server_olh.estimate_all(range(1, d+1))
        olh_estimates[i] = np.reshape(olh_estimates[i], (-1, num_of_classes))
    return olh_estimates

def privatise_oue(epsilon, data, Y):

    oue_estimates = [None]*len(data)
    num_of_classes = len(Y.unique())

    for i in range(len(data)):
        d = len(np.unique(data[i]))

        client_oue = UEClient(epsilon=epsilon, d=d, use_oue=True)
        server_oue = UEServer(epsilon=epsilon, d=d, use_oue=True)

        priv_oue_data = [client_oue.privatise(item) for item in data[i]]

        server_oue.aggregate_all(priv_oue_data)
        oue_estimates[i] = server_oue.estimate_all(range(1, d+1))
        oue_estimates[i] = np.reshape(oue_estimates[i], (-1,num_of_classes))
    return oue_estimates

def privatise_the(epsilon, data, Y):

    the_estimates = [None]*len(data)
    num_of_classes = len(Y.unique())

    for i in range(len(data)):
        d = len(np.unique(data[i]))

        client_the = HEClient(epsilon=epsilon, d=d)
        server_the = HEServer(epsilon=epsilon, d=d, use_the=True)

        priv_the_data = [client_the.privatise(item) for item in data[i]]

        server_the.aggregate_all(priv_the_data)
        the_estimates[i] = server_the.estimate_all(range(1, d+1))
        the_estimates[i] = np.reshape(the_estimates[i], (-1,num_of_classes))
    return the_estimates

def privatise_hr(epsilon, data, Y):

    hr_estimates = [None]*len(data)
    num_of_classes = len(Y.unique())

    for i in range(len(data)):
        d = len(np.unique(data[i]))

        server_hr = HadamardResponseServer(epsilon, d)
        client_hr = HadamardResponseClient(epsilon, d, server_hr.get_hash_funcs())

        priv_hr_data = [client_hr.privatise(item) for item in data[i]]

        server_hr.aggregate_all(priv_hr_data)
        hr_estimates[i] = server_hr.estimate_all(range(1, d+1))
        hr_estimates[i] = np.reshape(hr_estimates[i], (-1,num_of_classes))
    return hr_estimates

def privatise_cms(epsilon, k, m, data, Y):

    cms_estimates = [None]*len(data)
    num_of_classes = len(Y.unique())

    for i in range(len(data)):
        d = len(np.unique(data[i]))

        server_cms = CMSServer(epsilon, k, m)
        client_cms = CMSClient(epsilon, server_cms.get_hash_funcs(), m)

        priv_cms_data = [client_cms.privatise(item) for item in data[i]]

        server_cms.aggregate_all(priv_cms_data)
        cms_estimates[i] = server_cms.estimate_all(range(1, d+1))
        cms_estimates[i] = np.reshape(cms_estimates[i], (-1,num_of_classes))
    return cms_estimates

def privatise_rappor(epsilon, k, m, data, Y):

    f = round(1/(0.5*math.exp(epsilon/2)+0.5), 2)
    rappor_estimates = [None]*len(data)
    num_of_classes = len(Y.unique())

    for i in range(len(data)):
        d = len(np.unique(data[i]))

        server_rappor = RAPPORServer(f, rappor_m, rappor_k, d)
        client_rappor = RAPPORClient(f, rappor_m, server_rappor.get_hash_funcs())

        priv_rappor_data = [client_rappor.privatise(item) for item in data[i]]

        server_rappor.aggregate_all(priv_rappor_data)
        rappor_estimates[i] = server_rappor.estimate_all(range(1, d+1))
        rappor_estimates[i] = np.reshape(rappor_estimates[i], (-1,num_of_classes))
    return rappor_estimates