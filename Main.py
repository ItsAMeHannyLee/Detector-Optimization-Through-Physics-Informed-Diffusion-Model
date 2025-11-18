import numpy as np
from scipy.constants import c as speed_of_light  # m/s
from scipy.special import betainc  # For approximations if needed

# Material dictionary (from your table + trackers; units: rho kg/m3, k W/m-K, c J/kg-K, X0 g/cm2, Tmax K, cost $/kg)
materials = {
    'Al': {'rho': 2700, 'k': 237, 'c': 900, 'X0': 24.01, 'Tmax': 993.47, 'cost': 3, 'type': 'dead'},
    'Fe': {'rho': 7874, 'k': 79.5, 'c': 450, 'X0': 13.84, 'Tmax': 1811, 'cost': 1, 'type': 'dead'},
    'Cu': {'rho': 8960, 'k': 398, 'c': 385, 'X0': 12.86, 'Tmax': 1357.8, 'cost': 10, 'type': 'dead'},
    'Ti': {'rho': 4500, 'k': 22, 'c': 523, 'X0': 16.16, 'Tmax': 1941, 'cost': 20, 'type': 'dead'},
    'Brass': {'rho': 8600, 'k': 120, 'c': 380, 'X0': 13.16, 'Tmax': 1178, 'cost': 5, 'type': 'dead'},
    'Si_pixel': {'rho': 2330, 'k': 148, 'c': 678, 'X0': 21.82, 'Tmax': 1687, 'cost': 10000, 'type': 'tracker'},
    'Si_strip': {'rho': 2330, 'k': 148, 'c': 678, 'X0': 21.82, 'Tmax': 1687, 'cost': 8000, 'type': 'tracker'}
}

# Simple Bethe-Bloch approximation for dE/dx (MeV cm2/g); for relativistic particles
def bethe_bloch(E, mass, Z, A, rho, I=170e-6):  # I in MeV, default for Si ~170 eV=1.7e-4 MeV
    beta = np.sqrt(1 - (mass**2 / E**2))  # Assuming E >> mass, beta ~1
    gamma = 1 / np.sqrt(1 - beta**2)
    K = 0.307075  # MeV cm2 / mol
    z = 1  # Charge of particle
    me = 0.511  # Electron mass MeV
    Tmax = (2 * me * beta**2 * gamma**2) / (1 + 2*gamma*me/E + (me/E)**2)
    dedx = (K * z**2 * Z / A / beta**2) * (0.5 * np.log(2*me*beta**2*gamma**2*Tmax / I**2) - beta**2)
    return dedx  # Electronic only, density effect neglected for simplicity

# Simulate one event
def simulate_event(config, E_range=(1,100), eta_range=(-3,3), num_events=5000):
    # config: list of dicts [{'mat': 'Al', 'dr': 0.01, 'p': 1.0, 'r_start': 0.0}]  # dr in m, r in m
    # Filter present layers (p > random)
    layers = [layer for layer in config if np.random.rand() < layer['p']]
    
    # Generate events
    types = np.random.choice(['muon', 'pion'], size=num_events, p=[0.5, 0.5])
    Es = np.exp(np.random.uniform(np.log(E_range[0]), np.log(E_range[1]), num_events))  # Log-uniform E in GeV
    etas = np.random.uniform(eta_range[0], eta_range[1], num_events)
    thetas = 2 * np.arctan(np.exp(-etas))  # θ in radians from beam axis
    
    efficiencies = []
    thermal_loads = np.zeros(len(layers))  # Total energy deposited per layer (J)
    total_cost = 0
    total_X0 = 0
    
    muon_mass = 0.1057  # GeV
    pion_mass = 0.1396  # GeV
    
    for i in range(num_events):
        particle_type = types[i]
        E = Es[i] * 1e3  # MeV
        theta = thetas[i]
        sin_theta = np.sin(theta)
        if sin_theta == 0: sin_theta = 1e-6  # Avoid div0
        
        hits = 0
        for j, layer in enumerate(layers):
            mat = materials[layer['mat']]
            path_length = layer['dr'] * 1e2 / sin_theta  # cm (convert m to cm)
            grammage = mat['rho'] * 1e-3 * path_length  # g/cm2 (rho kg/m3 = g/10^-3 cm3)
            
            if particle_type == 'muon':
                dedx = 2.0  # MIP approx MeV/g cm2
            else:  # Pion
                mass = pion_mass * 1e3 if particle_type == 'pion' else muon_mass * 1e3  # MeV
                dedx = bethe_bloch(E, mass, 14, 28, mat['rho'], I=0.000173)  # For Si-like, adjust Z/A per mat
            
            dE = dedx * grammage  # MeV
            E -= dE  # Update energy (simple, ignore straggling)
            
            thermal_loads[j] += dE * 1.602e-13  # Convert MeV to J
            
            if mat['type'] == 'tracker' and dE > 0.1:  # Threshold for hit
                hits += 1
                
            total_X0 += mat['X0'] * (grammage / mat['X0'])  # Wait, total X0 fraction = grammage / X0
            
        efficiency = hits / max(1, sum(1 for l in layers if materials[l['mat']]['type'] == 'tracker'))
        efficiencies.append(efficiency)
        
        # Cost: mass = rho * volume, assume area=1 m2 for normalization, dr in m
        mass = mat['rho'] * layer['dr'] * 1  # kg (per m2)
        total_cost += mat['cost'] * mass
    
    avg_efficiency = np.mean(efficiencies)
    max_thermal = thermal_loads / (np.array([materials[l['mat']]['c'] * (materials[l['mat']]['rho'] * l['dr'] * 1) for l in layers]))  # Delta T approx = E / (c * mass)
    violated = np.any(max_thermal > np.array([materials[l['mat']]['Tmax'] - 300 for l in layers]))  # From room T
    
    return {'efficiency': avg_efficiency, 'total_cost': total_cost, 'total_X0': total_X0 / num_events, 'thermal_violated': violated}

# Example usage: Define a config (r_start cumulative)
config = [
    {'mat': 'Al', 'dr': 0.01, 'p': 1.0, 'r_start': 0.0},
    {'mat': 'Si_strip', 'dr': 0.005, 'p': 0.8, 'r_start': 0.01},
    # Add more layers...
]

results = simulate_event(config)
print(results)