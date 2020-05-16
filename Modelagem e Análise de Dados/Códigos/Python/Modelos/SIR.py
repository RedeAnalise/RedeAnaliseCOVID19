import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse


def prediction(beta, gamma, population, i0, r0, time_predict):
    def SIR_model(y, t, beta, gamma, population):
        s, i, r = y
        dSdt = -beta * s * i / population
        dIdt = beta * s * i / population - gamma * i
        dRdt = gamma * i
        return [dSdt, dIdt, dRdt]

    s0 = population - i0 - r0
    y_0 = [s0, i0, r0]
    sol = odeint(SIR_model, y_0, time_predict, args=(beta, gamma, population))
    sol = np.transpose(sol)
    return sol


def error_model(point, cases, population, infected_0, recovered_0):
    beta, gamma = point

    def SIR_model(y, t, beta, gamma, population):
        s, i, r = y
        dSdt = -beta * s * i / population
        dIdt = beta * s * i / population - gamma * i
        dRdt = gamma * i
        return [dSdt, dIdt, dRdt]

    suscepted_0 = population - infected_0 - recovered_0
    y0 = [suscepted_0, infected_0, recovered_0]
    sol = odeint(SIR_model, y0, np.arange(1, len(cases) + 1), args=(beta, gamma, population))
    sol = np.transpose(sol)
    error = mse(cases, sol[1])
    return error


def trainer(cases, population, infected_0, recovered_0):
    optimal = minimize(error_model, np.array([0.001, 0.001]), args=(cases, population, infected_0, recovered_0),
                       method='L-BFGS-B', bounds=[(0.000001, 1.0), (0.000001, 1.0)])
    beta, gamma = optimal.x
    return beta, gamma

  

def plot(s, i, r, initials_state, state_name, city_name, period_predict, time_predict, population, date_city, initial_month, last_month, last_day, local_insert=None):
    plt.figure()
    plt.title('Projeção do total de habitantes sucetíveis, infectados e recuperados em ' + city_name + '/' + initials_state + ' a partir de ' + str(last_day) + ' de ' + last_month + ', em dias.', fontsize=18)
    plt.xlabel('Dias', fontsize=15)
    plt.ylabel('Número de habitantes', fontsize=15)
    plt.yticks(np.arange(0, population, step=population * 0.03))
    plt.plot(time_predict, s, 'b', label='Sucetíveis')
    plt.plot(time_predict, i, 'r', label='Infectados')
    plt.plot(time_predict, r, 'g', label='Recuperados')
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
        plt.savefig(path+'Projecao_Modelo_SIR'+date_plot+'.png', format="PNG")
        plt.clf()
    else:
        pass