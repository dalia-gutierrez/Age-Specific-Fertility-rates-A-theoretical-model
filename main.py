from multiprocessing import Pool
import numpy as np
from scipy.integrate import solve_bvp, quad
from scipy.stats import lognorm
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import pandas as pd

# Parameters
pitol_values = [0.0004] # [1.0, 0.5, 0.1, 0.05, 0.03, 0.02, 0.015, 0.014, 0.013, 0.012, 0.011, 0.01, 0.008, 0.006, 0.005, 0.001, 0.0008, 0.0006, 0.0005, 0.0004]
params = {
    'pitol': pitol_values[-1], 'xi': 2.4, 'w0': 0.1, 'z': 0.1, 'Gamma_1': 0.9/50, 
    'Gamma_2': 0.29, 'gamma': 1, 'T': 70, 'epsilon': 1, 'sigma': 0.001,
    'h_0': 1.0, 'g': 0.02, 'g0': 0.02, 'sd': 0.35, 'mean_pi': 3.2, 
    'ddelta': 0.05, 'rho': 0.02, 'alpha': 0.25, 'phi': 0.6, 'psi': 1.9, 
    'N_bar': 25, 'factor': 2.0, 'bc_epsilon': 1e-4
}

# Excel file load guess
excel_file = "initial_guess.xlsx"

# Cohort grid
tau_grid = np.array([0, 35, 70])

# Function to load initial guess from excel
def load_excel_guess(filepath, params, n_points=100):

    df = pd.read_excel(filepath)
    
    a_data = df.iloc[:, 0].to_numpy()
    var_data = [df.iloc[:, i].to_numpy() for i in range(1, 7)]
    var_names = df.columns[1:7].tolist()
    
    a_eval = np.linspace(0, params['T'], n_points)
    
    y_guess_array = np.zeros((6, n_points))
    for i in range(6):
        interp_fn = interp1d(a_data, var_data[i], kind='linear', fill_value="extrapolate")
        y_guess_array[i] = interp_fn(a_eval)
    
    return y_guess_array, a_eval, var_names

# Interest rate function (constant)
def r_traj_func(a, r):
    return r * np.ones_like(a)

# Vectorized probability functions
def pi(a, params):
    return lognorm.pdf(a, params['sd'], scale=np.exp(params['mean_pi'])) + params['pitol']

def dlnpidt(a, params):
    h = 1e-8
    a_upper = a + h
    a_lower = a - h
    a_lower = np.maximum(a_lower, 0)
    return (np.log(pi(a_upper, params)) - np.log(pi(a_lower, params))) / (2 * h)

# ODE system
def model_odes(a, y, tau, r, params):
    c, l, n, h, b, N_tilde = y
    c = np.maximum(c, 1e-10)
    l = np.maximum(l, 1e-10)
    h = np.maximum(h, 1e-10)
    r_val = r_traj_func(a + tau, r)
    
    w0, g, g0 = params['w0'], params['g'], params['g0']
    pi_a = pi(a, params)
    dlnpidt_a = dlnpidt(a, params)
    
    dcdt = -(params['rho'] - r_val)/params['epsilon'] * c
    dhdt = params['z'] * h**params['phi'] * l**params['alpha'] - params['ddelta'] * (h - params['h_0'])
    
    term1 = l/(1-params['alpha']) * (params['ddelta'] + r_val - g + (1-params['phi'])*params['ddelta']*(1 - params['h_0']/h))
    term2 = params['z'] * h**(params['phi']-1) * l**params['alpha'] / (1-params['alpha']) * (l + params['alpha']*(1 - params['Gamma_1']*n - l))
    dldt = term1 - term2
    
    term3 = c**(-params['epsilon']) * ((r_val - g - dhdt/h)*w0*np.exp(g*(a + tau))*params['Gamma_1']*h + (r_val - g0)*params['Gamma_2']*np.exp(g0*(a + tau)))
    term4 = -params['sigma'] * np.maximum(N_tilde + params['N_bar'], 1e-10)**(-params['gamma'])
    
    n_reg = np.clip(n, 1e-5, 0.1)
    pi_a_safe = np.maximum(pi_a, 1e-10)
    psi_term = np.maximum(params['psi'], 1e-3)
    dndt = (term3 + term4) * n_reg**(1 + params['psi']) / (params['xi'] * psi_term * pi_a_safe**params['psi'])
    dndt += (dlnpidt_a - params['rho']/psi_term)*n
    
    dbdt = r_val*b + w0*np.exp(g*(a + tau))*h*(1 - l - params['Gamma_1']*n) - c - params['Gamma_2']*np.exp(g0*(a + tau))*n
    dN_tildedt = n
    
    return np.vstack([dcdt, dldt, dndt, dhdt, dbdt, dN_tildedt])

# Boundary conditions
def bc(ya, yb, tau, r, params):
    return np.array([
        ya[3] - params['h_0'],
        ya[4],
        ya[5],
        yb[1],
        yb[4],
        yb[2] - params['bc_epsilon'] # (0.00018+params['pitol'])*(0.1*50/0.9)**(1/1.9)
    ])

# Solve BVP for a cohort
def solve_cohort(tau, r, params, n_points=100, y_guess_array=None):
    a_eval = np.linspace(0, params['T'], n_points)
    
    if y_guess_array is None:
        y_guess, a_eval, var_names = load_excel_guess(excel_file, params, n_points=100)
#       y_guess = np.zeros((6, n_points))
#       y_guess[0] = params['w0'] * np.exp(params['g']*(tau + a_eval))
#       y_guess[1] = 0.99 * (1 - a_eval/params['T']) * (a_eval < params['T'])
#       y_guess[2] = params['bc_epsilon'] * np.ones_like(a_eval)  # Adjusted guess for n
#       y_guess[3] = params['h_0'] * np.ones_like(a_eval)
#       y_guess[4] = np.zeros_like(a_eval)
#       y_guess[5] = np.zeros_like(a_eval)
    else:
        y_guess = y_guess_array
    
    sol = solve_bvp(
        fun=lambda a, y: model_odes(a, y, tau, r, params),
        bc=lambda ya, yb: bc(ya, yb, tau, r, params),
        x=a_eval,
        y=y_guess,
        max_nodes=100000,
        verbose=0
    )
    return sol

# Continuation method for pitol
def solve_with_continuation(tau, r, params):
    sol = None
    for pitol in pitol_values:
        print(f"  Solving with pitol = {pitol}")
        current_params = params.copy()
        current_params['pitol'] = pitol
        
        if sol is None:
            sol = solve_cohort(tau, r, current_params)
        else:
            n_points = 100
            a_eval = np.linspace(0, current_params['T'], n_points)
            y_guess_interp = np.zeros((6, n_points))
            for i in range(6):
                interp_fn = interp1d(sol.x, sol.y[i], kind='linear', fill_value="extrapolate")
                y_guess_interp[i] = interp_fn(a_eval)
            sol = solve_cohort(tau, r, current_params, y_guess_array=y_guess_interp)
        
        if not sol.success:
            print(f"    Continuation step failed with pitol={pitol}: {sol.message}")
            break
    
    return sol

# Get b and n functions
def get_b_n_functions(r, params):
    sol = solve_with_continuation(0, r, params)
    if not sol.success:
        raise ValueError(f"Failed to solve for cohort 0 with r={r}: {sol.message}")
    
    b_interp = interp1d(sol.x, sol.y[4], kind='cubic', fill_value="extrapolate")
    n_interp = interp1d(sol.x, sol.y[2], kind='cubic', fill_value="extrapolate")
    
    def b(a):
        return b_interp(np.clip(a, 0, params['T']))
    
    def n(a):
        return n_interp(np.clip(a, 0, params['T']))  # Keep scaling as in original
    
    return b, n

# Fixed point function with split integration
def fixed_point(g_n, n, params):
    def integrand(a):
        return n(a) * np.exp(-g_n * a)
    
    T = params['T']
    intervals = [(0, T/3), (T/3, 2*T/3), (2*T/3, T)]
    integral = 0
    for a, b in intervals:
        result, _ = quad(integrand, a, b, limit=100, epsabs=1e-10, epsrel=1e-10)
        integral += result
    
    return 1 - integral

# Find equilibrium g_n
def find_equilibrium_g_n(r, params, g_n_bounds=(-0.1, 1)):
    _, n = get_b_n_functions(r, params)
    try:
        f_a = fixed_point(g_n_bounds[0], n, params)
        f_b = fixed_point(g_n_bounds[1], n, params)
        print(f"r={r:.6f}, fixed_point(g_n={g_n_bounds[0]})={f_a:.6f}, fixed_point(g_n={g_n_bounds[1]})={f_b:.6f}")
        
        if f_a * f_b >= 0:
            raise ValueError(f"No sign change in fixed_point for r={r} over g_n_bounds={g_n_bounds}")
        
        result = root_scalar(
            lambda g_n: fixed_point(g_n, n, params),
            bracket=g_n_bounds,
            method='brentq',
            xtol=1e-8,
            rtol=1e-8
        )
        return result.root
    except Exception as e:
        print(f"Error finding g_n for r={r}: {str(e)}")
        return None

# Aggregate bonds function
def agg_bonds(r, params):
    g_n = find_equilibrium_g_n(r, params)
    if g_n is None:
        return np.nan
    
    def integrand(tau):
        return np.exp(g_n * tau) * bonds(params['T'] - tau, tau, r, params)
    
    T = params['T']
    intervals = [(0, T/3), (T/3, 2*T/3), (2*T/3, T)]
    integral = 0
    for a, b in intervals:
        result, _ = quad(integrand, a, b, limit=100, epsabs=1e-10, epsrel=1e-10)
        integral += result
    
    return integral

# Bonds function
def bonds(a, tau, r, params):
    b, _ = get_b_n_functions(r, params)
    return np.exp(params['g'] * tau) * b(a)

# Find equilibrium r
def find_equilibrium_r(params, r_bounds=(0.03, 0.1)):
    try:
        f_a = agg_bonds(r_bounds[0], params)
        f_b = agg_bonds(r_bounds[1], params)
        print(f"agg_bonds(r={r_bounds[0]})={f_a:.6f}, agg_bonds(r={r_bounds[1]})={f_b:.6f}")
        
        if np.isnan(f_a) or np.isnan(f_b):
            raise ValueError("agg_bonds returned NaN")
        if f_a * f_b >= 0:
            raise ValueError(f"No sign change in agg_bonds over r_bounds={r_bounds}")
        
        result = root_scalar(
            lambda r: agg_bonds(r, params),
            bracket=r_bounds,
            method='brentq',
            xtol=1e-8,
            rtol=1e-8
        )
        return result.root
    except Exception as e:
        print(f"Error finding equilibrium r: {str(e)}")
        return None

# Define a function for processing a single epsilon value
def process_epsilon(eps):
    print(f"\nProcessing epsilon = {eps}")
    current_params = params.copy()
    current_params['bc_epsilon'] = eps
    r_equilibrium = find_equilibrium_r(current_params)
    if r_equilibrium is not None:
        g_n_equilibrium = find_equilibrium_g_n(r_equilibrium, current_params)
        if g_n_equilibrium is not None:
            sol = solve_with_continuation(0, r_equilibrium, current_params)
            if sol.success:
                yb_0 = sol.y[0][-1]
                yb_2 = sol.y[2][-1]
                yb_3 = sol.y[3][-1]
                denom = (current_params['w0'] * np.exp(current_params['g'] * current_params['T']) * yb_3 * current_params['Gamma_1'] +
                         current_params['Gamma_2'] * np.exp(current_params['g0'] * current_params['T']))
                if denom <= 0:
                    print(f"Denominator non-positive for epsilon={eps}")
                    return (eps, None, None, None, np.inf, None)
                term = (current_params['xi'] * yb_0 ** current_params['epsilon'] / denom) ** (1 / current_params['psi'])
                theoretical = pi(current_params['T'] , current_params) * term
                diff = abs(yb_2 - theoretical)
                print(f"epsilon={eps}, diff={diff:.6f}, theoretical={theoretical:.6f}")
                return (eps, r_equilibrium, g_n_equilibrium, sol, diff, theoretical)
            else:
                print(f"Failed to solve BVP for epsilon={eps}")
        else:
            print(f"Failed to find g_n for epsilon={eps}")
    else:
        print(f"Failed to find equilibrium r for epsilon={eps}")
    return (eps, None, None, None, np.inf, None)

# Main execution with parallel loop
if __name__ == '__main__':
    np.random.seed(123)

    epsilons = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3,
		1.1e-3, 1.2e-3, 1.3e-3, 1.4e-3, 1.5e-3, 1.6e-3, 1.7e-3, 1.8e-3, 1.9e-3]


    with Pool() as pool:
        results = pool.map(process_epsilon, epsilons)
    
    # Filter out failed results and find the best
    valid_results = [r for r in results if r[1] is not None]
    num_unique = len(valid_results)
    if valid_results:
        best_result = min(valid_results, key=lambda x: x[4])
        eps_best, r_equilibrium, g_n_equilibrium, sol, diff_best, theo_best = best_result

        if diff_best/eps_best > 0.1:
            print(f"Error greater than 10 percent: {diff_best/eps_best:.6f}")
            offsets = [-9e-5, -8e-5, -7e-5, -6e-5, -5e-5, -4e-5, -3e-5, -2e-5, -1e-5, 1e-5, 2e-5, 3e-5, 4e-5, 5e-5, 6e-5, 7e-5, 8e-5, 9e-5]
            new_epsilons = [eps_best + offset for offset in offsets]

            with Pool() as pool:
                new_results = pool.map(process_epsilon, new_epsilons)

            new_results.append(best_result)
            valid_new_results = [r for r in new_results if r[1] is not None]
            num_unique = len(valid_new_results) + num_unique - 1
            best_new_result = min(valid_new_results, key=lambda x: x[4])
            eps_best, r_equilibrium, g_n_equilibrium, sol, diff_best, theo_best = best_new_result
            
        print(f"\nBest epsilon: {eps_best}, with diff: {diff_best:.6f}, theoretical: {theo_best:.6f}")
        print(f"Equilibrium interest rate r: {r_equilibrium:.6f}")
        print(f"Equilibrium g_n: {g_n_equilibrium:.6f}")
        print(f"Number of succesful simulations: {num_unique}")
        
        # Plot results
        plt.figure(figsize=(12, 8))
        variables = ['Consumo (c)', 'Capital humano (l)', 'CBR/1000 (n)', 
                     'Capital humano acumulado (h)', 'Bonos (b)', 'Hijos acumulados (N_tilde)']
        for i, var in enumerate(variables):
            plt.subplot(3, 2, i+1)
            if var in ['CBR/1000 (n)', 'Hijos acumulados (N_tilde)']:
                plt.plot(sol.x, sol.y[i], label='tau=0')
            else:
                plt.plot(sol.x, sol.y[i], label='tau=0')
            plt.title(f'{var}')
            plt.xlabel('a')
            plt.legend()
        plt.tight_layout()
        plt.savefig("results_plot.png")
        print("Plot saved to results_plot.png")
        
        # Convertir resultados a DataFrame
        data = {'a': sol.x}
        for i, var in enumerate(variables):
            if var in ['CBR/1000 (n)', 'Hijos acumulados (N_tilde)']:
                data[var] = sol.y[i]
            else:
                data[var] = sol.y[i]
        df = pd.DataFrame(data)
        df.to_excel("resultados.xlsx", index=False)
        print("Resultados exportados a resultados.xlsx")
    else:
        print("No successful results found.")
