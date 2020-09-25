import numpy as np, scipy.stats as stats

class Case:

    def __init__(self, x1, x2, y, a):
        self.x1 = x1
        self.x2 = x2
        self.y = y
        self.a = a

    def req_eq(self):
        global b_matrix, b0, b_x1, b_x2
        n = len(y)
        x_matrix = np.array([np.ones(n), x1, x2]).T
        b_matrix = np.linalg.inv(x_matrix.T @ x_matrix) @ x_matrix.T @ y.T
        b0, b_x1, b_x2 = b_matrix[0], b_matrix[1], b_matrix[2]
        print('Matrix of coefficients =', b_matrix)
        print('b0 =', b_matrix[0], ';', 'b_x1 =', b_matrix[1], ';', 'b_x2 =', b_matrix[2])
        print()

    def error_variance(self):
        n, k = len(y), 3
        s2 = 1. / n * sum([(y[i] - b0 - np.array([b_x1, b_x2]).T @ np.array([x1[i], x2[i]])) ** 2 for i in range(n)])
        print('Unbiased estimator of the error variance =', s2 * 10 / (n - k), '\n')

    def conf_int_pr_ent(self):
        n, k = len(y), 3
        new_x = np.c_[np.ones(n), x1, x2]
        x_11 = np.array([1., 14., 8.])
        y_11 = x_11 @ np.array([b0, b_x1, b_x2]).T
        value = x_11.T @ (x_11 @ np.linalg.inv(new_x.T @ new_x))
        T_statistics = stats.t.ppf(q=1 - a / 2., df=n - k)
        std_error = (sum([(y[i] - b0 - np.array([b_x1, b_x2]).T @ np.array([x1[i], x2[i]])) ** 2 for i in range(n)]) / (n - k)) ** .5
        deviation_av = std_error * value ** .5 * T_statistics
        deviation_ind = std_error * (value + 1) ** .5 * T_statistics
        print('Interval for the average profit for enterprises:', [y_11 - deviation_av, y_11 + deviation_av])
        print('Interval for the individual profit for enterprises:', [y_11 - deviation_ind, y_11 + deviation_ind])
        print()

    def T_test_for_coef(self):
        n, k = len(y), 3
        std_error = (1. / n * sum([(y[i] - b0 - np.array([b_x1, b_x2]).T @ np.array([x1[i], x2[i]])) ** 2 for i in range(n)]) * 10 / (n - k)) ** .5
        stderr_b0 = (1/n + std_error**2 * (x1.mean()**2 * sum([(x2[i] - x2.mean())**2 for i in range(n)]) + x2.mean()**2 * sum([(x1[i] - x1.mean())**2 for i in range(n)]) - 2*x1.mean()*x2.mean()*sum([(x1[i]-x1.mean())*(x2[i]-x2.mean()) for i in range(n)]))/(sum([(x1[i]-x1.mean())**2 for i in range(n)]) * sum([(x2[i]-x2.mean())**2 for i in range(n)]) - (sum([(x1[i]-x1.mean())*(x2[i]-x2.mean()) for i in range(n)]))**2))**.5
        stderr_b1 = (std_error**2 * sum([(x2[i]-x2.mean())**2 for i in range(n)])/(sum([(x1[i]-x1.mean())**2 for i in range(n)])*sum([(x2[i]-x2.mean())**2 for i in range(n)]) - (sum([(x1[i]-x1.mean())*(x2[i]-x2.mean()) for i in range(n)]))**2))**.5
        stderr_b2 = (std_error**2 * sum([(x1[i]-x1.mean())**2 for i in range(n)])/(sum([(x1[i]-x1.mean())**2 for i in range(n)])*sum([(x2[i]-x2.mean())**2 for i in range(n)]) - (sum([(x1[i]-x1.mean())*(x2[i]-x2.mean()) for i in range(n)]))**2))**.5
        T_observed_b0 = abs(b0 / stderr_b0)
        T_statistics_b0 = stats.t.ppf(q=1 - a / 2., df=n - k)
        print('T_crit =', T_statistics_b0, ';', 'T_obs_b0 =', T_observed_b0)
        T_observed_b1 = abs(b_x1 / stderr_b1)
        T_statistics_b1 = stats.t.ppf(q=1 - a / 2., df=n - k)
        print('T_crit =', T_statistics_b1, ';', 'T_obs_b_x1 =', T_observed_b1)
        T_observed_b2 = abs(b_x2 / stderr_b2)
        T_statistics_b2 = stats.t.ppf(q=1 - a / 2., df=n - k)
        print('T_crit =', T_statistics_b2, ';', 'T_obs_b_x2 =', T_observed_b2)
        if T_statistics_b0 < T_observed_b0 and T_statistics_b1 < T_observed_b1 and T_statistics_b2 < T_observed_b2:
            print('Regression coefficients are statistically significant')
        else:
            print('Regression coefficients are statistically insignificant')
        print()

    def conf_inter_for_reg_param(self):
        n, k = len(y), 3
        s2 = 1. / n * sum([(y[i] - b0 - np.array([b_x1, b_x2]).T @ np.array([x1[i], x2[i]])) ** 2 for i in range(n)])
        c1 = stats.chi2.ppf(a / 2., n - k)
        c2 = stats.chi2.ppf(1 - a / 2., n - k)
        print('The confidence interval of ev is: ', [n * s2 / c2, n * s2 / c1])
        new_x = np.c_[np.ones(n), x1, x2].T
        matrix = np.linalg.inv(new_x @ new_x.T)
        err_var = (s2 * n / (n-k))
        c = -1 * stats.t.ppf(a / 2., n - k)
        bb0 = c * (matrix[0][0] * err_var)**.5
        print('The confidence interval of b0 is: ', [b0 - bb0, b0 + bb0])
        bb1 = c * (matrix[1][1] * err_var)**.5
        print('The confidence interval of b_x1 is: ', [b_x1 - bb1, b_x1 + bb1])
        bb2 = c * (matrix[2][2] * err_var)**.5
        print('The confidence interval of b_x2 is: ', [b_x2 - bb2, b_x2 + bb2])
        print()

    def R_square_calc(self):
        n = len(y)
        R_square = sum([(b0 + np.array([b_x1, b_x2]).T @ np.array([x1[i], x2[i]]) - y.mean())**2 for i in range(n)])/sum([(y[i] - y.mean())**2 for i in range(n)])
        print('Coefficient of determination =', R_square, '\n')

    def F_test(self):
        n, k = len(y), 3
        R_square = sum([(b0 + np.array([b_x1, b_x2]).T @ np.array([x1[i], x2[i]]) - y.mean())**2 for i in range(n)])/sum([(y[i] - y.mean())**2 for i in range(n)])
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

# initial data
x1 = np.array([11., 10., 12., 18., 15., 13., 13., 15., 16., 17.])
x2 = np.array([3., 2., 4., 9., 11., 5., 3., 7., 8., 7.])
y = np.array([2., 1., 3., 8., 7., 5., 4., 6., 7., 7.])
a = 0.05

# for black work

# output
case_2 = Case(x1, x2, y, a)
print(case_2.req_eq(), '\n',
      case_2.error_variance(), '\n',
      case_2.conf_int_pr_ent(), '\n',
      case_2.T_test_for_coef(), '\n',
      case_2.conf_inter_for_reg_param(), '\n',
      case_2.R_square_calc(), '\n',
      case_2.F_test())