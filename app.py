import numpy as np
from iapws import IAPWS97

def calculate_pressure_drop_sizing(P_in, T_in, flow_kg_h, L_m, roughness_mm=0.045, max_dp_ratio=0.10, max_dp_barg=1.0):
    # Convert inputs to SI units
    P_mpa = (P_in * 0.1) + 0.101325
    steam = IAPWS97(P=P_mpa, T=T_in+273.15) if T_in else IAPWS97(P=P_mpa, x=1)
    
    rho = 1 / steam.v  # kg/m^3
    mu = steam.mu      # Pa.s (dynamic viscosity)
    mass_flow_kg_s = flow_kg_h / 3600
    
    # Establish max allowable pressure drop constraints
    allowed_dp_pa = min(P_in * max_dp_ratio, max_dp_barg) * 100000 
    
    # Placeholder standard pipe internal diameters (mm) - requires specific schedules
    standard_diameters_mm = [15, 20, 25, 32, 40, 50, 65, 80, 100, 150, 200, 250, 300]
    
    for D_mm in standard_diameters_mm:
        D_m = D_mm / 1000
        area = (np.pi * D_m**2) / 4
        velocity = mass_flow_kg_s / (rho * area)
        
        # Reynolds number
        Re = (rho * velocity * D_m) / mu
        
        # Haaland Equation for Friction Factor
        epsilon_D = (roughness_mm / 1000) / D_m
        if Re > 4000: # Turbulent
            f = ( -1.8 * np.log10( (epsilon_D / 3.7)**1.11 + (6.9 / Re) ) )**-2
        else:
            f = 64 / Re if Re > 0 else 0
            
        # Darcy-Weisbach Pressure Drop
        dp_pa = f * (L_m / D_m) * (rho * velocity**2) / 2
        
        if dp_pa <= allowed_dp_pa:
            return {
                "Optimal_Diameter_mm": D_mm,
                "Velocity_m_s": velocity,
                "Pressure_Drop_bar": dp_pa / 100000,
                "Friction_Factor": f,
                "Reynolds_Number": Re
            }
            
    return {"Error": "Exceeds max diameter constraints."}
