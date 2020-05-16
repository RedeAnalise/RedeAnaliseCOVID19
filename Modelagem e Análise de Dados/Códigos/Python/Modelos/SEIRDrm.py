import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse


def prediction(beta, alfa, f, t_i, population, e0, i0, r0, d0, time_predict):
    """
    beta is the infection rate
    alfa is the number of days that the virus is take to me transmissible (incubation rate)
    f is fraction of indivuals who die
    population is the total population of the problem, assumed constant
    e0 is the initial individuals exposed (with the virus but not transmissing)
    i0 is the initial individuals infected and transmissing the virus
    r0 is the initial individuals recovered
    d0 is the initial deads
    time_predict is the time (in days) to make a prediction
    t_i is the average infectious period
    """
    def SEIRDrm_model(y, t, beta, alfa, f, t_i, population):
        s, e, i, r, d = y
        dSdt = -beta * s * i / population
        dEdt = beta * s * i / population - alfa * e
        dIdt = alfa * e - i / t_i
        dRdt = (1 - f) * i / t_i
        dDdt = f * i / t_i
        return [dSdt, dEdt, dIdt, dRdt, dDdt]
    s0 = population - e0 - i0 - r0 - d0
    y_0 = [s0, e0, i0, r0, d0]
    sol = odeint(SEIRDrm_model, y_0, time_predict, args=(beta, alfa, f, t_i, population))
    sol = np.transpose(sol)
    return sol


def error_model(point, t_i, cases, population, exposed_0, infected_0, recovered_0, dead_0):
    
    beta, alfa, f = point

    def SEIRDrm_model(y, t, beta, alfa, f, t_i, population):
        s, e, i, r, d = y
        dSdt = -beta * s * i / population
        dEdt = beta * s * i / population - alfa * e
        dIdt = alfa * e - i / t_i
        dRdt = (1 - f) * i / t_i
        dDdt = f * i / t_i
        return [dSdt, dEdt, dIdt, dRdt, dDdt]
    
    suscepted_0 = population - exposed_0 - infected_0 - recovered_0 - dead_0
    y0 = [suscepted_0, exposed_0, infected_0, recovered_0, dead_0]
    t = np.arange(1, cases.size + 1)
    sol = odeint(SEIRDrm_model, y0, t, args=(beta, alfa, f, t_i, population))
    sol = np.transpose(sol)
    error = mse(cases, sol[1])/2
    return error


def trainer(t_i, cases, population, exposed_0, infected_0, recovered_0, dead_0):
    optimal = minimize(error_model, np.array([0.001, 0.5, 0.001]), args=(t_i, cases, population, exposed_0, infected_0, recovered_0, dead_0),
                       method='L-BFGS-B', bounds=[(0.000001, 1.0), (1.0, 15.0), (0.000001, 1.0)])
    beta, alfa, f = optimal.x
    return beta, alfa, f

  

def plot(s, e, i, r, d, initials_state, state_name, city_name, period_predict, time_predict, population, date_city, initial_month, last_month, last_day, local_insert=None):
    plt.figure()
    plt.title('Projeção do total de habitantes sucetíveis, expostos, infectados, recuperados e mortos em ' + city_name + '/' + initials_state + ' a partir de ' + str(last_day) + ' de ' + last_month + ', em dias.', fontsize=18)
    plt.xlabel('Dias', fontsize=15)
    plt.ylabel('Número de habitantes', fontsize=15)
    plt.yticks(np.arange(0, population, step=population * 0.03))
    plt.plot(time_predict, s, 'b', label='Sucetíveis')
    plt.plot(time_predict, i, 'r', label='Infectados')
    plt.plot(time_predict, r, 'g', label='Recuperados')
    plt.plot(time_predict, e, 'y', label='Expostos')
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
        plt.savefig(path+'Projecao_Modelo_SEIRDrm'+date_plot+'.png', format="PNG")
        plt.clf()
    else:
        pass
