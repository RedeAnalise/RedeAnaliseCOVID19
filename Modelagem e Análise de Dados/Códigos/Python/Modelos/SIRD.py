import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse


def prediction(beta, gamma, rho, population, i0, r0, d0, time_predict):
    def SIRD_model(y, t, beta, gamma, rho, population):
        s, i, r, d = y
        dSdt = -beta * s * i / population
        dIdt = beta * s * i / population - gamma * i - rho * i
        dRdt = gamma * i
        dDdt = rho * i
        return [dSdt, dIdt, dRdt, dDdt]

    s0 = population - i0 - r0 - d0
    y_0 = [s0, i0, r0, d0]
    sol = odeint(SIRD_model, y_0, time_predict, args=(beta, gamma, rho, population))
    sol = np.transpose(sol)
    return sol


def error_model(point, cases, population, infected_0, recovered_0, dead_0):
    beta, gamma, rho = point

    def SIRD_model(y, t, beta, gamma, rho, population):
        s, i, r, d = y
        dSdt = -beta * s * i / population
        dIdt = beta * s * i / population - gamma * i - rho * i
        dRdt = gamma * i
        dDdt = rho * i
        return [dSdt, dIdt, dRdt, dDdt]

    suscepted_0 = population - infected_0 - recovered_0 - dead_0
    y0 = [suscepted_0, infected_0, recovered_0, dead_0]
    sol = odeint(SIRD_model, y0, np.arange(1, len(cases) + 1), args=(beta, gamma, rho, population))
    sol = np.transpose(sol)
    error = mse(cases, sol[1])
    return error


def trainer(cases, population, infected_0, recovered_0, dead_0):
    optimal = minimize(error_model, np.array([0.001, 0.001, 0.001]), args=(cases, population, infected_0, recovered_0, dead_0),
                       method='L-BFGS-B', bounds=[(0.000001, 1.0), (0.000001, 1.0), (0.000001, 1.0)])
    beta, gamma, rho = optimal.x
    return beta, gamma, rho

  

def plot(s, i, r, d, initials_state, state_name, city_name, period_predict, time_predict, population, date_city, initial_month, last_month, last_day, local_insert=None):
    plt.figure()
    plt.title('Projeção do total de habitantes sucetíveis, infectados, recuperados e mortos em ' + city_name + '/' + initials_state + ' a partir de ' + str(last_day) + ' de ' + last_month + ', em dias.', fontsize=18)
    plt.xlabel('Dias', fontsize=15)
    plt.ylabel('Número de habitantes', fontsize=15)
    plt.yticks(np.arange(0, population, step=population * 0.03))
    plt.plot(time_predict, s, 'b', label='Sucetíveis')
    plt.plot(time_predict, i, 'r', label='Infectados')
    plt.plot(time_predict, r, 'g', label='Recuperados')
    plt.plot(time_predict, d, 'k', label='Mortos')
    plt.legend(loc='center left', bbox_to_anchor=(1.002, 0.7), fontsize=14)
    plt.rcParams["figure.figsize"] = (15, 10)
    plt.show()
    if local_insert != None:
        path = 'local_insert'+state_name+'/'+city_name+'-'+initials_state
        path = path[0]
        if not os.path.exists(path):
            os.makedirs(path)
        date_plot = str(max(date_city)).replace('00:','').replace('00','')
        date_plot = date_plot.replace(' ','')
        plt.savefig(path+'Projecao_Modelo_SIRD'+date_plot+'.png', format="PNG")
        plt.clf()
    else:
        pass
