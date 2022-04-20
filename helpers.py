import scipy.stats as st

def calculate_normal_distribution(number):
    norm_cdf = st.norm.cdf(number)
    negative_number = -number
    negative_norm_cdf = st.norm.cdf(-number)
    print("N(%f) =  %f" %(number, norm_cdf))
    print("N(%f) =  %f" % (negative_number, negative_norm_cdf))
    print()
    return norm_cdf
