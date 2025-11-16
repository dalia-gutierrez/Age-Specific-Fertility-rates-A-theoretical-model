import numpy as np
import pandas as pd
from scipy import interpolate, integrate

# Load the Excel file
excel_file = "resultados.xlsx"
data = pd.read_excel(excel_file)

# Extract the relevant columns
# Assume the first column is x-values (s), and the 4th column is y-values (n(s))
s_values = data.iloc[:, 0].values
n_values = data.iloc[:, 3].values

# Interpolation function for n(s)
n_interp = interpolate.interp1d(s_values, n_values, kind='linear', fill_value='extrapolate')

# Ask the user for g_n
g_n = float(input("Enter value for g_n: "))

# Define integrand functions
def numerator_integrand(s):
    return n_interp(s) * np.exp(-g_n * s)

def denominator_integrand(s):
    return np.exp(-g_n * s)

# Values of a
a_values = [15, 20, 25, 30, 35, 40, 45]

# Compute the sum
total = 0.0
for a in a_values:
    num, _ = integrate.quad(numerator_integrand, a, a + 5)
    den, _ = integrate.quad(denominator_integrand, a, a + 5)
    total += num / den

total = total*10

print(f"\nResult: {total:.6f}")
