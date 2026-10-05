import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from iapws import IAPWS97
from datetime import datetime
import pytz
from scipy.optimize import brentq
import io

# =====================================================================
# PAGE CONFIG & CONSTANTS
# =====================================================================
st.set_page_config(page_title="Steam Engineering Suite", layout="wide", initial_sidebar_state="expanded")

# Standard commercial pipe internal diameters (mm) - roughly Schedule 40
STD_PIPES_MM = [15.8, 20.9, 26.6, 35.1, 40.9, 52.5, 62.7, 77.9, 102.3, 128.2, 154.1, 202.7, 254.5, 304.8, 336.6, 381.0, 477.8]

ROUGHNESS_MAP = {
    "Commercial Steel / Wrought Iron": 0.045,
    "Cast Iron": 0.26,
    "Galvanized Iron": 0.15,
    "Drawn Tubing (Copper/Brass)": 0.0015,
    "Smooth Pipe (Plastic/Glass)": 0.0001
}

# =====================================================================
# CORE THERMO SOLVER (LEGACY NOMOGRAM)
# =====================================================================
def solve_single_case(steam_type, T, P, F, V, d):
    steam_type = 'saturated' if steam_type.lower().startswith('sat') else 'superheated'

    if steam_type == 'saturated':
        if P is not None:
            P_mpa = (P * 0.1) + 0.101325
            state = IAPWS97(P=P_mpa, x=1)
            T = state.T - 273.15
            vg = state.v
        elif T is not None:
            state = IAPWS97(T=T+273.15, x=1)
            P = (state.P - 0.101325) / 0.1
            vg = state.v
        else:
            vg = None
    else:
        if P is not None and T is not None:
            P_mpa = (P * 0.1) + 0.101325
            vg = IAPWS97(T=T+273.15, P=P_mpa).v
        else:
            vg = None

    if d is None and F is not None and V is not None and vg is not None:
        Q = (F / 3600) * vg
        d = 1000 * np.sqrt((4 * Q) / (np.pi * V))
    elif V is None and F is not None and d is not None and vg is not None:
        Q = (F / 3600) * vg
        V = (4 * Q) / (np.pi * (d / 1000)**2)
    elif F is None and V is not None and d is not None and vg is not None:
        Q = (np.pi * (d / 1000)**2 * V) / 4
        F = (Q * 3600) / vg
    elif P is None and T is None and steam_type == 'saturated':
        Q = (np.pi * (d / 1000)**2 * V) / 4
        vg = (Q * 3600) / F
        def find_tsat(t_k): return IAPWS97(T=t_k, x=1).v - vg
        T_k = brentq(find_tsat, 273.15 + 1, 273.15 + 373)
        state = IAPWS97(T=T_k, x=1)
        T = state.T - 273.15
        P = (state.P - 0.101325) / 0.1
    elif steam_type == 'superheated' and F is not None and V is not None and d is not None:
        Q = (np.pi * (d / 1000)**2 * V) / 4
        vg = (Q * 3600) / F
        if P is None:
            state = IAPWS97(T=T+273.15, v=vg)
            P = (state.P - 0.101325) / 0.1
        elif T is None:
            P_mpa = (P * 0.1) + 0.101325
            state = IAPWS97(P=P_mpa, v=vg)
            T = state.T - 273.15

    Q_trace = (F / 3600) * vg if (F and vg) else 0
    return {'type': steam_type, 'T': T, 'P': P, 'F': F, 'V': V, 'd': d, 'vg': vg, 'Q_trace': Q_trace}

# =====================================================================
# MODULE 1: NOMOGRAM PLOTTER
# =====================================================================
def solve_and_plot_steam_chart(types, Ts, Ps, Fs, Vs, ds):
    cases_data = []
    for i in range(len(types)):
        solved = solve_single_case(types[i], Ts[i], Ps[i], Fs[i], Vs[i], ds[i])
        cases_data.append(solved)

    fig = plt.figure(figsize=(15, 16))
    ax = fig.subplots(2, 2, gridspec_kw={'top': 0.95, 'bottom': 0.08})
    plt.subplots_adjust(wspace=0, hspace=0) 
    tl, tr, bl, br = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    tr.axis('off')

    tr.text(0.5, 0.95, "STEAM LINE SIZING NOMOGRAM", fontsize=16, weight='bold', ha='center', va='top', transform=tr.transAxes, color='#333333')
    
    table_cols = ["Case", "Type", "T (°C)", "P (bar g)", "F (kg/h)", "V (m/s)", "d (mm)"]
    table_data = []
    for i, c in enumerate(cases_data):
        st_type = "Sat" if c['type'] == 'saturated' else "Sup"
        table_data.append([f"C{i+1}", st_type, f"{c['T']:.1f}", f"{c['P']:.2f}", f"{c['F']:.0f}", f"{c['V']:.1f}", f"{c['d']:.1f}"])

    table = tr.table(cellText=table_data, colLabels=table_cols, loc='center', cellLoc='center', bbox=[0.05, 0.4, 0.95, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#5C2D91') 
        else:
            cell.set_facecolor('#f9f9f9')

    scalar_formatter = ticker.ScalarFormatter()
    scalar_formatter.set_scientific(False)

    def style_grid(axis):
        axis.grid(True, which='major', color='gray', linestyle='--', alpha=0.6)
        axis.grid(True, which='minor', color='gray', linestyle=':', alpha=0.3)

    temps = np.linspace(100, 500, 100)
    for p_curve in [0, 1, 5, 10, 20, 50, 100]:
        vgs = [IAPWS97(T=t+273.15, P=(p_curve*0.1)+0.101325).v if t > IAPWS97(P=(p_curve*0.1)+0.101325, x=1).T-273.15 else np.nan for t in temps]
        br.plot(temps, vgs, color='tab:blue', alpha=0.5, lw=1.2)
        valid_vgs = [v for v in vgs if not np.isnan(v)]
        if valid_vgs: br.text(495, valid_vgs[-1], f"{p_curve} bar", color='tab:blue', fontsize=8, va='center', ha='right')

    T_sat = np.linspace(100, 373.9, 100)
    v_sat = [IAPWS97(T=t+273.15, x=1).v for t in T_sat]
    br.plot(T_sat, v_sat, color='darkmagenta', lw=2.5)
    br.text(250, IAPWS97(T=250+273.15, x=1).v, " Saturation Curve", color='darkmagenta', fontsize=10, va='bottom', ha='left', rotation=8, weight='bold')
            
    br.set_yscale('log')
    br.set_xlim(100, 500) 
    br.set_ylim(0.01, 10) 
    br.yaxis.set_major_formatter(scalar_formatter)
    style_grid(br)
    br.set_title('Temperature vs. Specific Volume', y=-0.12) 
    br.yaxis.tick_right()
    br.yaxis.set_label_position("right")

    Qs = np.logspace(-3, 2, 100)
    for m_curve in [100, 500, 1000, 3000, 10000, 50000]:
        bl.plot(Qs, Qs / (m_curve / 3600), color='tab:green', alpha=0.5, lw=1.2)
        bl.text((m_curve / 3600) * 1.0, 1.0, f" {m_curve} kg/h", rotation=45, color='tab:green', fontsize=8, va='bottom', ha='left')

    bl.set_xscale('log')
    bl.set_yscale('log')
    bl.set_xlim(Qs.min(), Qs.max()) 
    bl.set_ylim(0.01, 10) 
    bl.xaxis.set_major_formatter(scalar_formatter)
    bl.yaxis.set_major_formatter(scalar_formatter)
    style_grid(bl)
    bl.set_title('Volumetric Flow vs. Specific Volume', y=-0.12) 
    bl.xaxis.tick_bottom()
    bl.xaxis.set_label_position("bottom")
    bl.yaxis.tick_left()
    bl.yaxis.set_label_position("left")

    for v_curve in [5, 10, 20, 40, 50, 100]:
        y_vals = 1000 * np.sqrt((4 * Qs) / (np.pi * v_curve))
        tl.plot(Qs, y_vals, color='tab:orange', alpha=0.5, lw=1.2)
        tl.text(Qs[0], y_vals[0], f" {v_curve} m/s", color='tab:orange', fontsize=8, va='bottom', ha='left')
        
    tl.set_xscale('log')
    tl.set_yscale('log')
    tl.set_xlim(Qs.min(), Qs.max()) 
    tl.set_ylim(10, 600)
    tl.set_yticks([10, 20, 50, 100, 200, 500])
    tl.yaxis.set_major_formatter(scalar_formatter)
    style_grid(tl)
    tl.set_title('Volumetric Flow vs. Diameter')

    colors = ['#e63946', '#1d3557', '#2a9d8f', '#f4a261', '#9c6644', '#6a4c93']
    for i, c in enumerate(cases_data):
        col = colors[i % len(colors)]
        T, vg, Q_trace, d = c['T'], c['vg'], c['Q_trace'], c['d']
        box_props = dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5, alpha=0.9)
        offset_y = 25 + (i * 25)

        br.plot([T, br.get_xlim()[0]], [vg, vg], color=col, linestyle='-', lw=1.5, alpha=0.85)
        br.plot(T, vg, marker='o', color=col, markersize=5)
        br.annotate(f"C{i+1}: {T:.1f}°C, {vg:.3f}m³/kg", xy=(T, vg), xytext=(20, offset_y), textcoords="offset points", bbox=box_props, ha='left', va='bottom', fontsize=8)

        bl.plot([bl.get_xlim()[1], Q_trace], [vg, vg], color=col, linestyle='-', lw=1.5, alpha=0.85)
        bl.plot([Q_trace, Q_trace], [vg, bl.get_ylim()[1]], color=col, linestyle='-', lw=1.5, alpha=0.85)
        bl.plot(Q_trace, vg, marker='o', color=col, markersize=5)
        bl.annotate(f"C{i+1}: {Q_trace:.3f}m³/s", xy=(Q_trace, vg), xytext=(20, offset_y), textcoords="offset points", bbox=box_props, ha='left', va='bottom', fontsize=8)

        tl.plot([Q_trace, Q_trace], [tl.get_ylim()[0], d], color=col, linestyle='-', lw=1.5, alpha=0.85)
        tl.plot([Q_trace, tl.get_xlim()[0]], [d, d], color=col, linestyle='-', lw=1.5, alpha=0.85)
        tl.plot(Q_trace, d, marker='o', color=col, markersize=5)
        tl.annotate(f"C{i+1}: {d:.1f}mm", xy=(Q_trace, d), xytext=(20, offset_y), textcoords="offset points", bbox=box_props, ha='left', va='bottom', fontsize=8)

    ist_tz = pytz.timezone('Asia/Kolkata')
    fig.text(0.05, 0.02, f"Generated on: {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')}", ha="left", va="bottom", fontsize=9, color="gray")
    
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    return fig, pdf_buffer

# =====================================================================
# MODULE 2: MOODY CHART PLOTTER
# =====================================================================
def plot_moody_chart(Re_op, f_op, ed_op, calc_data):
    fig, ax = plt.subplots(figsize=(12, 7))
    Re_arr = np.logspace(3, 8, 400)
    
    # Laminar Line
    Re_lam = np.linspace(1000, 2300, 50)
    ax.plot(Re_lam, 64/Re_lam, 'k-', lw=2, label="Laminar Flow (64/Re)")
    
    # Transition Zone Shade
    ax.axvspan(2300, 4000, color='yellow', alpha=0.2, label='Transition Zone')
    
    # Turbulent curves for common relative roughness
    ed_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    for ed in ed_list:
        f_arr = (-1.8 * np.log10((ed/3.7)**1.11 + 6.9/Re_arr))**-2
        ax.plot(Re_arr, f_arr, color='gray', alpha=0.5, lw=1)
        ax.text(Re_arr[-1]*1.1, f_arr[-1], f"{ed}", fontsize=8, va='center', color='gray')
        
    ax.text(Re_arr[-1]*1.1, 0.08, r"$\epsilon/D$", fontsize=10, weight='bold', color='gray')
    
    # Plot Operating Point
    box_props = dict(boxstyle="round,pad=0.4", fc="#e63946", ec="white", lw=2, alpha=0.9)
    ax.plot(Re_op, f_op, marker='o', markersize=10, color='#e63946', markeredgecolor='white', markeredgewidth=2, zorder=5)
    ax.annotate(f"Operating Point\nRe: {Re_op:.2e}\nf: {f_op:.4f}", 
                xy=(Re_op, f_op), xytext=(-20, 30), textcoords='offset points', 
                bbox=box_props, color='white', weight='bold', ha='right', arrowprops=dict(arrowstyle="->", color='#e63946', lw=2))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(0.008, 0.1)
    
    ax.set_xlabel("Reynolds Number, Re", fontsize=12, weight='bold')
    ax.set_ylabel("Friction Factor, f", fontsize=12, weight='bold')
    ax.set_title("Interactive Moody Chart Validation", fontsize=16, weight='bold', pad=15)
    ax.grid(True, which='both', color='gray', linestyle=':', alpha=0.6)
    ax.legend(loc='upper right')
    
    # Information Box
    info_text = (f"DESIGN SUMMARY:\n"
                 f"Pipe Dia: {calc_data['D_mm']} mm\n"
                 f"Velocity: {calc_data['Vel']:.1f} m/s\n"
                 f"Allowed Drop: {calc_data['Max_dp']:.2f} bar\n"
                 f"Actual Drop: {calc_data['Actual_dp']:.2f} bar")
    ax.text(0.02, 0.05, info_text, transform=ax.transAxes, fontsize=10, family='monospace', 
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="gray", alpha=0.9))

    ist_tz = pytz.timezone('Asia/Kolkata')
    fig.text(0.98, 0.02, f"Generated on: {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')}", ha="right", va="bottom", fontsize=8, color="gray")
    
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    return fig, pdf_buffer

# =====================================================================
# UI LAYOUT & ROUTING
# =====================================================================
st.sidebar.title("App Navigation")
mode = st.sidebar.radio("Select Engineering Module:", [
    "1. Line Sizing (Velocity/Nomogram)",
    "2. Design Validation (Calculated vs Actual)",
    "3. Pressure Drop Sizing (Darcy-Weisbach)"
])
st.sidebar.markdown("---")
st.sidebar.info("Developed for Advanced Steam Thermodynamics.")

# ---------------------------------------------------------
# MODE 1: VELOCITY NOMOGRAM
# ---------------------------------------------------------
if "Nomogram" in mode:
    st.title("Velocity Method: Line Sizing Nomogram")
    st.markdown("Leave **exactly one variable blank** per row to calculate it.")
    
    num_cases = st.number_input("Number of Cases to Compare:", min_value=1, max_value=10, value=1)
    
    df_init = pd.DataFrame({
        "Steam Type": ["Superheated"] * num_cases,
        "T (°C)": [200.0] * num_cases,
        "P (bar g)": [10.0] * num_cases,
        "Flow (kg/h)": [5000.0] * num_cases,
        "Vel (m/s)": [35.0] * num_cases,
        "Dia (mm)": [None] * num_cases,
    })

    edited_df = st.data_editor(df_init, use_container_width=True)

    def safe_float(val):
        return None if pd.isna(val) or val == "" else float(val)

    if st.button("Generate Nomogram & PDF", type="primary"):
        with st.spinner("Processing thermodynamic vectors..."):
            types, Ts, Ps, Fs, Vs, ds = [], [], [], [], [], []
            for _, row in edited_df.iterrows():
                types.append(row["Steam Type"])
                Ts.append(safe_float(row["T (°C)"]))
                Ps.append(safe_float(row["P (bar g)"]))
                Fs.append(safe_float(row["Flow (kg/h)"]))
                Vs.append(safe_float(row["Vel (m/s)"]))
                ds.append(safe_float(row["Dia (mm)"]))
                
            try:
                fig, pdf_bytes = solve_and_plot_steam_chart(types, Ts, Ps, Fs, Vs, ds)
                st.pyplot(fig)
                st.download_button(label="📥 Download Nomogram PDF", data=pdf_bytes, file_name="Steam_Nomogram.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Calculation Error: Ensure exactly one value is blank per row. Details: {e}")

# ---------------------------------------------------------
# MODE 2: VALIDATION (CALCULATED VS ACTUAL)
# ---------------------------------------------------------
elif "Validation" in mode:
    st.title("Design Validation: Installed Line Sizing")
    st.markdown("Enter process conditions and the **Actual Installed Diameter**. The engine determines if the line is undersized, optimal, or oversized based on standard ±15% variance rules.")
    
    val_num = st.number_input("Items to Validate:", min_value=1, max_value=20, value=2)
    val_df_init = pd.DataFrame({
        "Steam Type": ["Saturated"] * val_num,
        "P (bar g)": [10.0] * val_num,
        "T (°C)": [None] * val_num,
        "Flow (kg/h)": [3000.0] * val_num,
        "Target Vel (m/s)": [25.0] * val_num,
        "Actual Dia (mm)": [80.0, 150.0][:val_num] if val_num >= 2 else [80.0]*val_num
    })
    
    v_df = st.data_editor(val_df_init, use_container_width=True)
    
    if st.button("Run Validation Engine", type="primary"):
        results = []
        for i, row in v_df.iterrows():
            stype = row["Steam Type"]
            
            # Thermo resolving
            if stype == "Saturated" and pd.notna(row["P (bar g)"]):
                P_mpa = (row["P (bar g)"] * 0.1) + 0.101325
                vg = IAPWS97(P=P_mpa, x=1).v
            elif stype == "Superheated" and pd.notna(row["P (bar g)"]) and pd.notna(row["T (°C)"]):
                P_mpa = (row["P (bar g)"] * 0.1) + 0.101325
                vg = IAPWS97(T=row["T (°C)"]+273.15, P=P_mpa).v
            else:
                st.error(f"Row {i+1}: Missing thermo data. Saturated needs P; Superheated needs P and T.")
                continue
                
            Q = (row["Flow (kg/h)"] / 3600) * vg
            req_d = 1000 * np.sqrt((4 * Q) / (np.pi * row["Target Vel (m/s)"]))
            act_d = row["Actual Dia (mm)"]
            
            variance = ((act_d - req_d) / req_d) * 100
            
            # Ruling logic
            if variance < -5:
                status = "🔴 Undersized (High Vel Risk)"
            elif variance > 25:
                status = "🟡 Oversized (High CapEx/Heat Loss)"
            else:
                status = "🟢 Optimal Design"
                
            results.append({
                "Case": i+1,
                "Req. Dia (mm)": f"{req_d:.1f}",
                "Actual Dia (mm)": f"{act_d:.1f}",
                "Variance (%)": f"{variance:+.1f}%",
                "Validation Ruling": status
            })
            
        res_df = pd.DataFrame(results)
        st.write("### Validation Report")
        st.dataframe(res_df.style.applymap(lambda x: "background-color: #ffcccc" if "Undersized" in x else ("background-color: #ffffcc" if "Oversized" in x else ("background-color: #ccffcc" if "Optimal" in x else "")), subset=["Validation Ruling"]), use_container_width=True)

# ---------------------------------------------------------
# MODE 3: DARCY-WEISBACH & MOODY CHART
# ---------------------------------------------------------
elif "Pressure Drop" in mode:
    st.title("Darcy-Weisbach: Pressure Drop Sizing")
    st.markdown("Calculates optimal pipe size based on strict allowable pressure loss, dynamically mapping friction factors via the Haaland approximation.")
    
    col1, col2, col3, col4 = st.columns(4)
    stype_dw = col1.selectbox("Steam Type", ["Saturated", "Superheated"])
    P_in = col2.number_input("Inlet P (bar g)", value=10.0)
    T_in = col3.number_input("Inlet T (°C)", value=250.0) if stype_dw == "Superheated" else None
    flow = col4.number_input("Mass Flow (kg/h)", value=5000.0)
    
    col5, col6 = st.columns(2)
    L_m = col5.number_input("Equivalent Pipe Length (m)", value=100.0)
    material = col6.selectbox("Pipe Material", list(ROUGHNESS_MAP.keys()))
    roughness = ROUGHNESS_MAP[material]
    
    if st.button("Calculate Exact Sizing & Plot Moody Chart", type="primary"):
        with st.spinner("Iterating pipe dimensions and integrating friction models..."):
            max_dp_allowed = min(P_in * 0.10, 1.0) # 10% or 1 bar max
            
            P_mpa = (P_in * 0.1) + 0.101325
            try:
                steam = IAPWS97(P=P_mpa, T=T_in+273.15) if stype_dw == "Superheated" else IAPWS97(P=P_mpa, x=1)
                rho = 1 / steam.v
                mu = steam.mu
            except Exception as e:
                st.error("Invalid Thermodynamics parameters for IAPWS-IF97 formulation.")
                st.stop()
                
            mass_flow_kg_s = flow / 3600
            
            optimal_d = None
            final_dp, final_vel, final_f, final_Re = 0, 0, 0, 0
            
            # Loop standard pipes ascending to find smallest viable pipe
            for D_mm in STD_PIPES_MM:
                D_m = D_mm / 1000
                area = (np.pi * D_m**2) / 4
                velocity = mass_flow_kg_s / (rho * area)
                
                Re = (rho * velocity * D_m) / mu
                ed = (roughness / 1000) / D_m
                
                if Re > 4000:
                    f = (-1.8 * np.log10((ed/3.7)**1.11 + 6.9/Re))**-2
                else:
                    f = 64 / Re if Re > 0 else 0
                    
                dp_pa = f * (L_m / D_m) * (rho * velocity**2) / 2
                dp_bar = dp_pa / 100000
                
                if dp_bar <= max_dp_allowed:
                    optimal_d = D_mm
                    final_dp = dp_bar
                    final_vel = velocity
                    final_f = f
                    final_Re = Re
                    final_ed = ed
                    break
                    
            if optimal_d:
                colA, colB, colC = st.columns(3)
                colA.metric("Required Inner Diameter", f"{optimal_d} mm")
                colB.metric("Resulting Pressure Drop", f"{final_dp:.3f} bar", f"Limit: {max_dp_allowed:.2f} bar", delta_color="off")
                colC.metric("Pipeline Velocity", f"{final_vel:.1f} m/s")
                
                calc_dict = {'D_mm': optimal_d, 'Vel': final_vel, 'Max_dp': max_dp_allowed, 'Actual_dp': final_dp}
                fig, pdf_bytes = plot_moody_chart(final_Re, final_f, final_ed, calc_dict)
                st.pyplot(fig)
                st.download_button(label="📥 Download Moody Chart PDF", data=pdf_bytes, file_name="Moody_Chart_Validation.pdf", mime="application/pdf")
            else:
                st.error(f"Velocity constraints exceeded. A pipe larger than {STD_PIPES_MM[-1]}mm is required to maintain pressure drop below {max_dp_allowed:.2f} bar.")
