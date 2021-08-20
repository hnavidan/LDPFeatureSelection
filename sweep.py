import timeit
import random
import numpy as np
import LDP
import utils
from sklearn.metrics import mean_squared_error

def sweep(epsilons, data, Y, seed):
    estimate_olh = [None]*len(epsilons)
    estimate_oue = [None]*len(epsilons)
    estimate_the = [None]*len(epsilons)
    estimate_hr = [None]*len(epsilons)
    estimate_cms = [None]*len(epsilons)
    estimate_rappor = [None]*len(epsilons)

    for i, e in enumerate(epsilons):
        np.random.seed(seed)
        random.seed(seed)
        estimate_olh[i] = privatise_olh(e, data, Y)
        estimate_oue[i] = privatise_oue(e, data, Y)
        estimate_the[i] = privatise_the(e, data, Y)
        estimate_hr[i] = privatise_hr(e, data, Y)
        estimate_cms[i] = privatise_cms(e, cms_k, cms_m, data, Y)
        estimate_rappor[i] = privatise_rappor(e, rappor_k, rappor_m, data, Y)

        fix_zeros(estimate_olh[i], 1)
        fix_zeros(estimate_oue[i], 1)
        fix_zeros(estimate_the[i], 1)
        fix_zeros(estimate_hr[i], 1)
        fix_zeros(estimate_cms[i], 1)
        fix_zeros(estimate_rappor[i], 1)

        estimates = {"OLH": estimate_olh,
                    "OUE": estimate_oue,
                    "THE": estimate_the,
                    "HR": estimate_hr,
                    "CMS": estimate_cms,
                    "RAPPOR": estimate_rappor}

    return estimates


def sweep_time(epsilons, data, Y, seed):
    time_olh = np.zeros(len(epsilons))
    time_oue = np.zeros(len(epsilons))
    time_the = np.zeros(len(epsilons))
    time_hr = np.zeros(len(epsilons))
    time_cms = np.zeros(len(epsilons))
    time_rappor = np.zeros(len(epsilons))

    for i, e in enumerate(epsilons):
        np.random.seed(seed)
        random.seed(seed)
        time_olh[i] = timeit.timeit(lambda: privatise_olh(e, data, Y), globals=globals(), number=1)
        time_oue[i] = timeit.timeit(lambda: privatise_oue(e, data, Y), globals=globals(), number=1)
        time_the[i] = timeit.timeit(lambda: privatise_the(e, data, Y), globals=globals(), number=1)
        time_hr[i] = timeit.timeit(lambda: privatise_hr(e, data, Y), globals=globals(), number=1)
        time_cms[i] = timeit.timeit(lambda: privatise_cms(e, cms_k, cms_m, data, Y), globals=globals(), number=1)
        time_rappor[i] = timeit.timeit(lambda: privatise_rappor(e, rappor_k, rappor_m, data, Y), globals=globals(), number=1)

    times = {"OLH": time_olh,
            "OUE": time_oue,
            "THE": time_the,
            "HR": time_hr,
            "CMS": time_cms,
            "RAPPOR": time_rappor}

    return times

def calc_IG_metrics(epsilons, counts, estimates):
    IG_real = calc_information_gain(counts)
    IG_olh = [None]*len(epsilons)
    IG_oue = [None]*len(epsilons)
    IG_the = [None]*len(epsilons)
    IG_hr = [None]*len(epsilons)
    IG_cms = [None]*len(epsilons)
    IG_rappor = [None]*len(epsilons)

    rmse_olh = np.zeros(len(epsilons))
    rmse_oue = np.zeros(len(epsilons))
    rmse_the = np.zeros(len(epsilons))
    rmse_hr = np.zeros(len(epsilons))
    rmse_cms = np.zeros(len(epsilons))
    rmse_rappor = np.zeros(len(epsilons))

    kendall_olh = np.zeros(len(epsilons))
    kendall_oue = np.zeros(len(epsilons))
    kendall_the = np.zeros(len(epsilons))
    kendall_hr = np.zeros(len(epsilons))
    kendall_cms = np.zeros(len(epsilons))
    kendall_rappor = np.zeros(len(epsilons))
    
    for i, _ in enumerate(epsilons):
        IG_olh[i] = calc_information_gain(np.abs(estimates['OLH'][i]))
        IG_oue[i] = calc_information_gain(np.abs(estimates['OUE'][i]))
        IG_the[i] = calc_information_gain(np.abs(estimates['THE'][i]))
        IG_hr[i] = calc_information_gain(np.abs(estimates['HR'][i]))
        IG_cms[i] = calc_information_gain(np.abs(estimates['CMS'][i]))
        IG_rappor[i] = calc_information_gain(np.abs(estimates['RAPPOR'][i]))

        rmse_olh[i] = mean_squared_error(IG_real, IG_olh[i], squared=False)
        rmse_oue[i] = mean_squared_error(IG_real, IG_oue[i], squared=False)
        rmse_the[i] = mean_squared_error(IG_real, IG_the[i], squared=False)
        rmse_hr[i] = mean_squared_error(IG_real, IG_hr[i], squared=False)
        rmse_cms[i] = mean_squared_error(IG_real, IG_cms[i], squared=False)
        rmse_rappor[i] = mean_squared_error(IG_real, IG_rappor[i], squared=False)

        kendall_olh[i] = calc_kendall_tau(IG_real, IG_olh[i])
        kendall_oue[i] = calc_kendall_tau(IG_real, IG_oue[i])
        kendall_the[i] = calc_kendall_tau(IG_real, IG_the[i])
        kendall_hr[i] = calc_kendall_tau(IG_real, IG_hr[i])
        kendall_cms[i] = calc_kendall_tau(IG_real, IG_cms[i])
        kendall_rappor[i] = calc_kendall_tau(IG_real, IG_rappor[i])
        
        IG = {"OLH": IG_olh,
              "OUE": IG_oue,
              "THE": IG_the,
              "HR": IG_hr,
              "CMS": IG_cms,
              "RAPPOR": IG_rappor}

        RMSE = {"OLH": rmse_olh,
                "OUE": rmse_oue,
                "THE": rmse_the,
                "HR": rmse_hr,
                "CMS": rmse_cms,
                "RAPPOR": rmse_rappor}

        kendall = {"OLH": kendall_olh,
                   "OUE": kendall_oue,
                   "THE": kendall_the,
                   "HR": kendall_hr,
                   "CMS": kendall_cms,
                   "RAPPOR": kendall_rappor}

    return IG, RMSE, kendall

def calc_chi2_metrics(epsilons, counts, estimates):
    real_chi2, real_p = calc_chisquare(counts)
    chi2_olh = [None]*len(epsilons)
    chi2_oue = [None]*len(epsilons)
    chi2_the = [None]*len(epsilons)
    chi2_hr = [None]*len(epsilons)
    chi2_cms = [None]*len(epsilons)
    chi2_rappor = [None]*len(epsilons)

    p_olh = [None]*len(epsilons)
    p_oue = [None]*len(epsilons)
    p_the = [None]*len(epsilons)
    p_hr = [None]*len(epsilons)
    p_cms = [None]*len(epsilons)
    p_rappor = [None]*len(epsilons)

    rmse_olh = np.zeros(len(epsilons))
    rmse_oue = np.zeros(len(epsilons))
    rmse_the = np.zeros(len(epsilons))
    rmse_hr = np.zeros(len(epsilons))
    rmse_cms = np.zeros(len(epsilons))
    rmse_rappor = np.zeros(len(epsilons))
    
    kendall_olh = np.zeros(len(epsilons))
    kendall_oue = np.zeros(len(epsilons))
    kendall_the = np.zeros(len(epsilons))
    kendall_hr = np.zeros(len(epsilons))
    kendall_cms = np.zeros(len(epsilons))
    kendall_rappor = np.zeros(len(epsilons))
    
    for i, _ in enumerate(epsilons):
        chi2_olh[i], p_olh[i] = calc_chisquare(np.abs(estimates['OLH'][i]))
        chi2_oue[i], p_oue[i] = calc_chisquare(np.abs(estimates['OUE'][i]))
        chi2_the[i], p_the[i] = calc_chisquare(np.abs(estimates['THE'][i]))
        chi2_hr[i], p_hr[i] = calc_chisquare(np.abs(estimates['HR'][i]))
        chi2_cms[i], p_cms[i] = calc_chisquare(np.abs(estimates['CMS'][i]))
        chi2_rappor[i], p_rappor[i] = calc_chisquare(np.abs(estimates['RAPPOR'][i]))

        rmse_olh[i] = mean_squared_error(real_chi2, chi2_olh[i], squared=False)
        rmse_oue[i] = mean_squared_error(real_chi2, chi2_oue[i], squared=False)
        rmse_the[i] = mean_squared_error(real_chi2, chi2_the[i], squared=False)
        rmse_hr[i] = mean_squared_error(real_chi2, chi2_hr[i], squared=False)
        rmse_cms[i] = mean_squared_error(real_chi2, chi2_cms[i], squared=False)
        rmse_rappor[i] = mean_squared_error(real_chi2, chi2_rappor[i], squared=False)
        
        kendall_olh[i] = calc_kendall_tau(real_chi2, chi2_olh[i])
        kendall_oue[i] = calc_kendall_tau(real_chi2, chi2_oue[i])
        kendall_the[i] = calc_kendall_tau(real_chi2, chi2_the[i])
        kendall_hr[i] = calc_kendall_tau(real_chi2, chi2_hr[i])
        kendall_cms[i] = calc_kendall_tau(real_chi2, chi2_cms[i])
        kendall_rappor[i] = calc_kendall_tau(real_chi2, chi2_rappor[i])


    RMSE = {"OLH": rmse_olh,
            "OUE": rmse_oue,
            "THE": rmse_the,
            "HR": rmse_hr,
            "CMS": rmse_cms,
            "RAPPOR": rmse_rappor}

    chi2s = {"OLH": chi2_olh,
            "OUE": chi2_oue,
            "THE": chi2_the,
            "HR": chi2_hr,
            "CMS": chi2_cms,
            "RAPPOR": chi2_rappor}

    p_values = {"OLH": p_olh,
                "OUE": p_oue,
                "THE": p_the,
                "HR": p_hr,
                "CMS": p_cms,
                "RAPPOR": p_rappor}

    kendall = {"OLH": kendall_olh,
                   "OUE": kendall_oue,
                   "THE": kendall_the,
                   "HR": kendall_hr,
                   "CMS": kendall_cms,
                   "RAPPOR": kendall_rappor}

    return chi2s, p_values, RMSE, kendall
