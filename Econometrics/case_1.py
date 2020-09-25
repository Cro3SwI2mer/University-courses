import numpy as np, scipy.stats as stats, matplotlib.pyplot as plt

class Case_LSM:

    def __init__(self, x, y, a):
        self.case_x = x
        self.case_y = y
        self.case_alfa = a

    def reg_eq(self):
        global slope, intercept, rvalue, pvalue, stderr_b1
        slope, intercept, rvalue, pvalue, stderr_b1 = stats.linregress(x, y)
        print('b0 =', intercept, ';', 'b1 =', slope, '\n')

    def corr_coeff(self):
        print('Correlation coefficient =', rvalue, '\n')

    def R_square_calc(self):
        n = len(x)
        R_square = sum([(intercept + slope * x[i] - y.mean())**2 for i in range(n)])/sum([(y[i] - y.mean())**2 for i in range(n)])
        print('Coefficient of determination =', R_square, '\n')

    def error_variance(self):
        n, k = len(x), 2
        b0, b1 = intercept, slope
        s2 = 1. / n * sum([(y[i] - b0 - b1 * x[i]) ** 2 for i in range(n)])
        print('Unbiased estimator of the error variance =', s2 * 10 / 8, '\n')

    def F_test(self):
        n, k = len(x), 2
        R_square = sum([(intercept + slope * x[i] - y.mean()) ** 2 for i in range(n)]) / sum([(y[i] - y.mean()) ** 2 for i in range(n)])
        F_observed = (R_square/(k-1))/((1-R_square)/(n-k))
        F_statistic = stats.f.ppf(q=1-a, dfn=k-1, dfd=n-k)
        p_value = 1-stats.f.cdf(x=F_observed, dfn=k-1, dfd=n-k)
        print('pvalue =', p_value)
        print('F_crit =', F_statistic, ';', 'F_obs =',  F_observed)
        if F_statistic < F_observed:
            print('Regression is statistically significant')
        else:
            print('Model is insignificant')
        print()

    def T_test_for_coef(self):
        b0, b1 = intercept, slope
        n, k = len(x), 2
        std_error = (1. / n * sum([(y[i] - b0 - b1 * x[i]) ** 2 for i in range(n)]) * 10 / 8) ** .5
        stderr_b0 = (std_error ** 2 * (1 / n + (x.mean() ** 2) / (sum([(x[i] - x.mean()) ** 2 for i in range(n)])))) ** .5
        T_observed_b1 = abs(b1 / stderr_b1)
        T_statistics_b1 = stats.t.ppf(q=1 - a / 2., df=n - k)
        print('T_crit =', T_statistics_b1, ';', 'T_obs =', T_observed_b1)
        T_observed_b0 = abs(b0 / stderr_b0)
        T_statistics_b0 = stats.t.ppf(q=1 - a / 2., df=n - k)
        print('T_crit =', T_statistics_b0, ';', 'T_obs =', T_observed_b0)
        if T_statistics_b0 < T_observed_b0 and T_statistics_b1 < T_observed_b1:
            print('Regression coefficients are statistically significant')
        else:
            print('Coefficients are insignificant')
        print()

    def conf_int_num_doct(self):
        n, k = len(x), 2
        b0, b1 = intercept, slope
        new_x = np.c_[np.ones(n), x]
        x_11 = np.array([1., 11.5])
        y_11 = x_11 @ np.array([b0, b1]).T
        value = x_11.T @ (x_11 @ np.linalg.inv(new_x.T @ new_x))
        T_statistics = stats.t.ppf(q=1 - a / 2., df=n - k)
        std_error = (sum([(y[i] - b0 - b1 * x[i]) ** 2 for i in range(n)])/(n-k))**.5
        deviation_av = std_error * value**.5 * T_statistics
        deviation_ind = std_error * (value + 1)**.5 * T_statistics
        print('Interval for the average number of doctors:', [y_11 - deviation_av, y_11 + deviation_av])
        print('Interval for the individual value of the number of doctors:', [y_11 - deviation_ind, y_11 + deviation_ind])
        print()

    def conf_inter_for_reg_param(self):
        n, k = len(x), 2
        b0, b1 = intercept, slope
        s2 = 1. / n * sum([(y[i] - b0 - b1 * x[i]) ** 2 for i in range(n)])
        xx = x ** 2
        c1 = stats.chi2.ppf(a / 2., n - k)
        c2 = stats.chi2.ppf(1 - a / 2., n - k)
        print('The confidence interval of ev is: ', [n * s2 / c2, n * s2 / c1])
        c = -1 * stats.t.ppf(a / 2., n - k)
        bb1 = c * (s2 / ((n - 2) * (xx.mean() - (x.mean()) ** 2))) ** .5
        print('The confidence interval of b1 is: ', [b1 - bb1, b1 + bb1])
        bb0 = c * ((s2 / (n - 2)) * (1 + (x.mean()) ** 2 / (xx.mean() - (x.mean()) ** 2))) ** .5
        print('The confidence interval of b0 is: ', [b0 - bb0, b0 + bb0])
        print()

    def paint(self):
        b0, b1 = intercept, slope
        plt.plot(x, y, 'o', label='Initial data', markersize=10)
        plt.plot(x, b0 + b1*x, label='Linear Regression')
        plt.legend()
        plt.show()

# initial data
x = np.array([10.0, 10.3, 10.4, 10.55, 10.6, 10.7, 10.75, 10.9, 10.9, 11.0])
y = np.array([12.1, 12.6, 13.0, 13.8, 14.9, 16.0, 18.0, 20.0, 21.0, 22.0])
a = 0.05

# output
case_1 = Case_LSM(x, y, a)
print(case_1.reg_eq(), '\n',
      case_1.corr_coeff(), '\n',
      case_1.error_variance(), '\n',
      case_1.F_test(),'\n',
      case_1.T_test_for_coef(), '\n',
      case_1.R_square_calc(), '\n',
      case_1.conf_int_num_doct(), '\n',
      case_1.conf_inter_for_reg_param(), '\n',
      case_1.paint())