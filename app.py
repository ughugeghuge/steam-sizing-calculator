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
# PAGE CONFIG
# =====================================================================
st.set_page_config(page_title="Steam Sizing Calculator", layout="wide")
st.title("Steam Line Sizing Calculator")
st.markdown("Fill in the known variables in the table below. **Leave the value you want to calculate blank.**")

# =====================================================================
# CORE SOLVER AND PLOTTER
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

    if d is None:
        Q = (F / 3600) * vg
        d = 1000 * np.sqrt((4 * Q) / (np.pi * V))
    elif V is None:
        Q = (F / 3600) * vg
        V = (4 * Q) / (np.pi * (d / 1000)**2)
    elif F is None:
        Q = (np.pi * (d / 1000)**2 * V) / 4
        F = (Q * 3600) / vg
    elif P is None and T is None and steam_type == 'saturated':
        Q = (np.pi * (d / 1000)**2 * V) / 4
        vg = (Q * 3600) / F
        def find_tsat(t_k):
            return IAPWS97(T=t_k, x=1).v - vg
        T_k = brentq(find_tsat, 273.15 + 1, 273.15 + 373)
        state = IAPWS97(T=T_k, x=1)
        T = state.T - 273.15
        P = (state.P - 0.101325) / 0.1
    elif steam_type == 'superheated':
        Q = (np.pi * (d / 1000)**2 * V) / 4
        vg = (Q * 3600) / F
        if P is None:
            state = IAPWS97(T=T+273.15, v=vg)
            P = (state.P - 0.101325) / 0.1
        elif T is None:
            P_mpa = (P * 0.1) + 0.101325
            state = IAPWS97(P=P_mpa, v=vg)
            T = state.T - 273.15

    Q_trace = (F / 3600) * vg
    return {'type': steam_type, 'T': T, 'P': P, 'F': F, 'V': V, 'd': d, 'vg': vg, 'Q_trace': Q_trace}

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

    tr.text(0.5, 0.95, "STEAM LINE SIZING CALCULATION", fontsize=16, weight='bold', 
            ha='center', va='top', transform=tr.transAxes, color='#333333')
    
    table_cols = ["Case", "Type", "T (°C)", "P (bar g)", "F (kg/h)", "V (m/s)", "d (mm)"]
    table_data = []
    for i, c in enumerate(cases_data):
        st_type = "Sat" if c['type'] == 'saturated' else "Sup"
        table_data.append([
            f"Case {i+1}", st_type, 
            f"{c['T']:.1f}", f"{c['P']:.2f}", 
            f"{c['F']:.0f}", f"{c['V']:.1f}", f"{c['d']:.1f}"
        ])

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
        if valid_vgs:
            br.text(495, valid_vgs[-1], f"{p_curve} bar", color='tab:blue', fontsize=8, va='center', ha='right')

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
    br.set_xlabel('Temperature (°C)')
    br.set_ylabel('Specific Volume (m$^3$/kg)')
    br.yaxis.tick_right()
    br.yaxis.set_label_position("right")

    Qs = np.logspace(-3, 2, 100)
    for m_curve in [100, 500, 1000, 3000, 10000, 50000]:
        y_vals = Qs / (m_curve / 3600)
        bl.plot(Qs, y_vals, color='tab:green', alpha=0.5, lw=1.2)
        x_label_pos = (m_curve / 3600) * 1.0
        bl.text(x_label_pos, 1.0, f" {m_curve} kg/h", rotation=45, color='tab:green', fontsize=8, va='bottom', ha='left')

    bl.set_xscale('log')
    bl.set_yscale('log')
    bl.set_xlim(Qs.min(), Qs.max()) 
    bl.set_ylim(0.01, 10) 
    bl.xaxis.set_major_formatter(scalar_formatter)
    bl.yaxis.set_major_formatter(scalar_formatter)
    style_grid(bl)
    bl.set_title('Volumetric Flow vs. Specific Volume', y=-0.12) 
    bl.set_xlabel('Volumetric Flow (m$^3$/s)')
    bl.set_ylabel('Specific Volume (m$^3$/kg)')
    bl.xaxis.tick_bottom()
    bl.xaxis.set_label_position("bottom")
    bl.yaxis.tick_left()
    bl.yaxis.set_label_position("left")
    bl.tick_params(labelbottom=True, labelleft=True)

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
    tl.set_xlabel('Volumetric Flow (m$^3$/s)')
    tl.set_ylabel('Diameter (mm)')

    colors = ['#e63946', '#1d3557', '#2a9d8f', '#f4a261', '#9c6644', '#6a4c93', '#d90429', '#023e8a', '#0077b6']
    
    for i, c in enumerate(cases_data):
        col = colors[i % len(colors)]
        T, vg, Q_trace, d = c['T'], c['vg'], c['Q_trace'], c['d']
        box_props = dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5, alpha=0.9)
        arrow_props = dict(arrowstyle="->", color=col, lw=1.5, alpha=0.8)
        
        offset_x = 20
        offset_y = 25 + (i * 25)

        br.plot([T, br.get_xlim()[0]], [vg, vg], color=col, linestyle='-', lw=1.5, alpha=0.85)
        br.plot(T, vg, marker='o', color=col, markersize=5)
        br.annotate(f"C{i+1}: {T:.1f}°C, {vg:.3f}m³/kg", xy=(T, vg), xytext=(offset_x, offset_y), 
                    textcoords="offset points", bbox=box_props, arrowprops=arrow_props, ha='left', va='bottom', fontsize=8, zorder=6)

        bl.plot([bl.get_xlim()[1], Q_trace], [vg, vg], color=col, linestyle='-', lw=1.5, alpha=0.85)
        bl.plot([Q_trace, Q_trace], [vg, bl.get_ylim()[1]], color=col, linestyle='-', lw=1.5, alpha=0.85)
        bl.plot(Q_trace, vg, marker='o', color=col, markersize=5)
        bl.annotate(f"C{i+1}: {Q_trace:.3f}m³/s", xy=(Q_trace, vg), xytext=(offset_x, offset_y), 
                    textcoords="offset points", bbox=box_props, arrowprops=arrow_props, ha='left', va='bottom', fontsize=8, zorder=6)

        tl.plot([Q_trace, Q_trace], [tl.get_ylim()[0], d], color=col, linestyle='-', lw=1.5, alpha=0.85)
        tl.plot([Q_trace, tl.get_xlim()[0]], [d, d], color=col, linestyle='-', lw=1.5, alpha=0.85)
        tl.plot(Q_trace, d, marker='o', color=col, markersize=5)
        tl.annotate(f"C{i+1}: {d:.1f}mm", xy=(Q_trace, d), xytext=(offset_x, offset_y), 
                    textcoords="offset points", bbox=box_props, arrowprops=arrow_props, ha='left', va='bottom', fontsize=8, zorder=6)

    ist_tz = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist_tz).strftime("%Y-%m-%d %H:%M:%S IST")
    fig.text(0.05, 0.02, f"Generated on: {current_time}", ha="left", va="bottom", fontsize=9, color="gray")
    fig.text(0.95, 0.02, "prepared by UmeshGhuge", ha="right", va="bottom", fontsize=9, color="gray")

    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    
    return fig, pdf_buffer


# =====================================================================
# UI FRONTEND
# =====================================================================
num_cases = st.number_input("How many cases would you like to compare?", min_value=1, max_value=50, value=1)

# Create an empty dataframe to act as our spreadsheet
df_init = pd.DataFrame({
    "Steam Type": ["Superheated"] * num_cases,
    "T (°C)": [None] * num_cases,
    "P (bar g)": [None] * num_cases,
    "Flow (kg/h)": [None] * num_cases,
    "Vel (m/s)": [None] * num_cases,
    "Dia (mm)": [None] * num_cases,
})

# Display the interactive spreadsheet grid normal
st.write("### Input Parameters")
edited_df = st.data_editor(
    df_init,
    column_config={
        "Steam Type": st.column_config.SelectboxColumn(options=["Superheated", "Saturated"], required=True),
        "T (°C)": st.column_config.NumberColumn(),
        "P (bar g)": st.column_config.NumberColumn(),
        "Flow (kg/h)": st.column_config.NumberColumn(),
        "Vel (m/s)": st.column_config.NumberColumn(),
        "Dia (mm)": st.column_config.NumberColumn(),
    },
    use_container_width=True,
    hide_index=False
)

def safe_float(val):
    if pd.isna(val) or val == "":
        return None
    return float(val)

if st.button("Generate Nomogram & Report", type="primary"):
    with st.spinner("Calculating thermodynamics and rendering PDF..."):
        types, Ts, Ps, Fs, Vs, ds = [], [], [], [], [], []
        
        for index, row in edited_df.iterrows():
            types.append(row["Steam Type"])
            Ts.append(safe_float(row["T (°C)"]))
            Ps.append(safe_float(row["P (bar g)"]))
            Fs.append(safe_float(row["Flow (kg/h)"]))
            Vs.append(safe_float(row["Vel (m/s)"]))
            ds.append(safe_float(row["Dia (mm)"]))
            
        try:
            fig, pdf_bytes = solve_and_plot_steam_chart(types, Ts, Ps, Fs, Vs, ds)
            
            # Show the graph on the webpage
            st.pyplot(fig)
            
            # Provide the PDF download button
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"Steam_Sizing_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error in calculations: {e}. Please ensure exactly one variable is left blank per row based on the steam conditions.")
