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
    "Carbon Steel (Standard Steam)": 0.045,
    "Commercial Steel / Wrought Iron": 0.045,
    "Galvanized Iron": 0.15,
    "Cast Iron": 0.26,
    "Drawn Tubing (Copper/Brass)": 0.0015,
    "Smooth Pipe (Plastic/Glass)": 0.0001
}

def get_standard_pipe(min_d):
    for p in STD_PIPES_MM:
        if p >= min_d: return p
    return STD_PIPES_MM[-1]

def get_pipe_index(d):
    for i, p in enumerate(STD_PIPES_MM):
        if p >= d: return i
    return len(STD_PIPES_MM) - 1

def safe_float(val):
    try:
        if pd.isna(val) or val == "": return None
        return float(val)
    except:
        return None

# =====================================================================
# THERMO & PRESSURE DROP SOLVERS
# =====================================================================
def calc_dp(D_mm, mass_flow_kg_s, rho, mu, L_m, roughness):
    D_m = D_mm / 1000.0
    area = np.pi * (D_m**2) / 4.0
    vel = mass_flow_kg_s / (rho * area)
    Re = (rho * vel * D_m) / mu
    ed = (roughness/1000.0) / D_m
    if Re > 4000:
        f = (-1.8 * np.log10((ed/3.7)**1.11 + 6.9/Re))**-2
    else:
        f = 64/Re if Re>0 else 0
    dp_bar = (f * (L_m/D_m) * (rho * vel**2) / 2) / 100000.0
    return dp_bar, vel, f, Re, ed

def get_thermo_props(stype, P_bar, T_c):
    if pd.isna(P_bar): return None, None, None
    P_mpa = (P_bar * 0.1) + 0.101325
    try:
        if stype.lower().startswith('sat'):
            state = IAPWS97(P=P_mpa, x=1)
        else:
            if pd.isna(T_c): return None, None, None
            state = IAPWS97(P=P_mpa, T=T_c+273.15)
        return 1/state.v, state.mu, state.v
    except:
        return None, None, None

def solve_single_case(steam_type, T, P, F, V, d):
    steam_type = 'saturated' if steam_type.lower().startswith('sat') else 'superheated'
    rho, mu, vg = get_thermo_props(steam_type, P, T)
    
    if steam_type == 'saturated' and P is not None and T is None:
        T = IAPWS97(P=(P * 0.1) + 0.101325, x=1).T - 273.15

    if pd.isna(d) and pd.notna(F) and pd.notna(V) and vg is not None:
        Q = (F / 3600) * vg
        d = 1000 * np.sqrt((4 * Q) / (np.pi * V))
    elif pd.isna(V) and pd.notna(F) and pd.notna(d) and vg is not None:
        Q = (F / 3600) * vg
        V = (4 * Q) / (np.pi * (d / 1000)**2)
    elif pd.isna(F) and pd.notna(V) and pd.notna(d) and vg is not None:
        Q = (np.pi * (d / 1000)**2 * V) / 4
        F = (Q * 3600) / vg

    Q_trace = (F / 3600) * vg if (pd.notna(F) and vg) else 0
    return {'type': steam_type, 'T': T, 'P': P, 'F': F, 'V': V, 'd': d, 'vg': vg, 'Q_trace': Q_trace}

# =====================================================================
# PDF REPORT GENERATORS (Modernized Tables)
# =====================================================================
def add_watermark(fig):
    fig.text(0.98, 0.02, "prepared by- Umesh Ghuge", ha="right", va="bottom", fontsize=10, color="lightgray", style="italic")

def generate_validation_pdf(df, title="STEAM LINE ADEQUACY REPORT"):
    fig, ax = plt.subplots(figsize=(16, min(4 + len(df)*0.5, 12)))
    ax.axis('off')
    
    ax.text(0.5, 0.95, title, fontsize=20, weight='bold', ha='center', va='top', color='#1F4E79')
    
    table_data = [df.columns.to_list()] + df.values.tolist()
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0.0, 0.1, 1.0, 0.75])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#E0E0E0')
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor('#2C3E50')
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')
            if "Undersized" in str(cell.get_text().get_text()):
                cell.set_text_props(color='#C0392B', weight='bold')
            elif "Optimal" in str(cell.get_text().get_text()):
                cell.set_text_props(color='#27AE60', weight='bold')

    ist_tz = pytz.timezone('Asia/Kolkata')
    fig.text(0.02, 0.02, f"Date: {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')}", ha="left", va="bottom", fontsize=9, color="gray")
    add_watermark(fig)
    
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    plt.close(fig)
    return pdf_buffer

def solve_and_plot_steam_chart(types, Ts, Ps, Fs, Vs, ds):
    cases_data = []
    for i in range(len(types)):
        cases_data.append(solve_single_case(types[i], Ts[i], Ps[i], Fs[i], Vs[i], ds[i]))

    fig = plt.figure(figsize=(15, 16))
    ax = fig.subplots(2, 2, gridspec_kw={'top': 0.95, 'bottom': 0.08})
    plt.subplots_adjust(wspace=0, hspace=0) 
    tl, tr, bl, br = ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]
    tr.axis('off')

    tr.text(0.5, 0.95, "STEAM LINE SIZING SUMMARY", fontsize=18, weight='bold', ha='center', va='top', color='#1F4E79')
    
    table_cols = ["Case", "Type", "T (°C)", "P (bar g)", "F (kg/h)", "V (m/s)", "d (mm)"]
    table_data = []
    for i, c in enumerate(cases_data):
        st_type = "Sat" if c['type'] == 'saturated' else "Sup"
        table_data.append([f"C{i+1}", st_type, f"{c['T']:.1f}" if pd.notna(c['T']) else "-", 
                           f"{c['P']:.2f}", f"{c['F']:.0f}", f"{c['V']:.1f}", f"{c['d']:.1f}"])

    # Modernized Table
    table = tr.table(cellText=table_data, colLabels=table_cols, loc='center', cellLoc='center', bbox=[0.05, 0.4, 0.9, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#E0E0E0')
        if row == 0:
            cell.set_facecolor('#2C3E50') 
            cell.set_text_props(weight='bold', color='white')
        else:
            cell.set_facecolor('#F8F9FA' if row % 2 == 0 else '#FFFFFF')

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
        if pd.isna(T) or pd.isna(vg) or pd.isna(d): continue
        
        box_props = dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5, alpha=0.9)
        arrow_props = dict(arrowstyle="->", color=col, lw=1.5, alpha=0.8)
        offset_x, offset_y = 20, 25 + (i * 25)

        br.plot([T, br.get_xlim()[0]], [vg, vg], color=col, lw=1.5, alpha=0.85)
        br.plot(T, vg, marker='o', color=col)
        br.annotate(f"C{i+1}", xy=(T, vg), xytext=(offset_x, offset_y), textcoords="offset points", bbox=box_props, arrowprops=arrow_props, fontsize=8)

        bl.plot([bl.get_xlim()[1], Q_trace], [vg, vg], color=col, lw=1.5, alpha=0.85)
        bl.plot([Q_trace, Q_trace], [vg, bl.get_ylim()[1]], color=col, lw=1.5, alpha=0.85)
        bl.plot(Q_trace, vg, marker='o', color=col)
        bl.annotate(f"C{i+1}", xy=(Q_trace, vg), xytext=(offset_x, offset_y), textcoords="offset points", bbox=box_props, arrowprops=arrow_props, fontsize=8)

        tl.plot([Q_trace, Q_trace], [tl.get_ylim()[0], d], color=col, lw=1.5, alpha=0.85)
        tl.plot([Q_trace, tl.get_xlim()[0]], [d, d], color=col, lw=1.5, alpha=0.85)
        tl.plot(Q_trace, d, marker='o', color=col)
        tl.annotate(f"C{i+1}", xy=(Q_trace, d), xytext=(offset_x, offset_y), textcoords="offset points", bbox=box_props, arrowprops=arrow_props, fontsize=8)

    ist_tz = pytz.timezone('Asia/Kolkata')
    fig.text(0.05, 0.02, f"Date: {datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S IST')}", ha="left", va="bottom", fontsize=9, color="gray")
    add_watermark(fig)
    
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    plt.close(fig)
    return fig, pdf_buffer

def plot_moody_chart(Re_op, f_op, ed_op, calc_data):
    fig, ax = plt.subplots(figsize=(12, 7))
    Re_arr = np.logspace(3, 8, 400)
    
    Re_lam = np.linspace(1000, 2300, 50)
    ax.plot(Re_lam, 64/Re_lam, color='#2C3E50', lw=2, label="Laminar Flow")
    ax.axvspan(2300, 4000, color='#F1C40F', alpha=0.2, label='Transition Zone')
    
    ed_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(ed_list)))
    
    for ed, color in zip(ed_list, colors):
        f_arr = (-1.8 * np.log10((ed/3.7)**1.11 + 6.9/Re_arr))**-2
        ax.plot(Re_arr, f_arr, color=color, alpha=0.7, lw=1.5)
        ax.text(Re_arr[-1]*1.1, f_arr[-1], f"{ed}", fontsize=8, va='center', color=color, weight='bold')
        
    ax.text(Re_arr[-1]*1.1, 0.08, r"$\epsilon/D$", fontsize=10, weight='bold', color='#333')
    
    box_props = dict(boxstyle="round,pad=0.5", fc="#E74C3C", ec="white", lw=2, alpha=0.95)
    ax.plot(Re_op, f_op, marker='*', markersize=18, color='#E74C3C', markeredgecolor='white', markeredgewidth=1.5, zorder=5)
    ax.annotate(f"Final Design Point\nRe: {Re_op:.2e}\nf: {f_op:.4f}\nD: {calc_data['D_mm']} mm", 
                xy=(Re_op, f_op), xytext=(-30, 40), textcoords='offset points', 
                bbox=box_props, color='white', weight='bold', ha='right', arrowprops=dict(arrowstyle="->", color='#E74C3C', lw=2))
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1e3, 1e8)
    ax.set_ylim(0.008, 0.1)
    
    ax.set_xlabel("Reynolds Number (Re)", fontsize=12, weight='bold')
    ax.set_ylabel("Friction Factor (f)", fontsize=12, weight='bold')
    ax.set_title("Interactive Moody Chart: Friction Validation", fontsize=16, weight='bold', color='#1F4E79')
    
    ax.grid(True, which='major', color='#BDC3C7', linestyle='-', alpha=0.8)
    ax.grid(True, which='minor', color='#BDC3C7', linestyle=':', alpha=0.5)
    ax.legend(loc='upper right')

    add_watermark(fig)
    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)
    plt.close(fig)
    return fig, pdf_buffer


# =====================================================================
# UI LAYOUT & ROUTING
# =====================================================================
st.sidebar.title("App Navigation")
mode = st.sidebar.radio("Select Engineering Module:", [
    "1. Line Sizing (Velocity Nomogram)",
    "2. Line Adequacy & Validation (Bulk)",
    "3. Pressure Drop Sizing (Moody Chart)"
])
st.sidebar.markdown("---")

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

    edited_df = st.data_editor(
        df_init,
        column_config={"Steam Type": st.column_config.SelectboxColumn(options=["Superheated", "Saturated"], required=True)},
        use_container_width=True
    )

    if st.button("Generate Nomogram & PDF", type="primary"):
        with st.spinner("Processing thermodynamic vectors..."):
            types = edited_df["Steam Type"].tolist()
            Ts = [safe_float(x) for x in edited_df["T (°C)"]]
            Ps = [safe_float(x) for x in edited_df["P (bar g)"]]
            Fs = [safe_float(x) for x in edited_df["Flow (kg/h)"]]
            Vs = [safe_float(x) for x in edited_df["Vel (m/s)"]]
            ds = [safe_float(x) for x in edited_df["Dia (mm)"]]
            
            try:
                fig, pdf_bytes = solve_and_plot_steam_chart(types, Ts, Ps, Fs, Vs, ds)
                st.pyplot(fig)
                st.download_button(label="📥 Download Nomogram PDF", data=pdf_bytes, file_name="Steam_Nomogram.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"Calculation Error: Ensure data is entered correctly. Details: {e}")

# ---------------------------------------------------------
# MODE 2: LINE ADEQUACY VALIDATION (BULK UPLOAD)
# ---------------------------------------------------------
elif "Adequacy" in mode:
    st.title("Line Adequacy Validation")
    st.markdown("Upload an Excel file or use the table below to validate existing steam lines. The engine sizes the line using both **Velocity** and **Pressure Drop** limits, then compares them against your **Actual Dia** to rule if the line is Adequate, Undersized, or Oversized.")

    tab1, tab2 = st.tabs(["Manual Entry", "Excel Upload"])
    
    with tab1:
        val_num = st.number_input("Rows:", min_value=1, max_value=20, value=2)
        val_df_init = pd.DataFrame({
            "Case ID": [f"Line-{i+1}" for i in range(val_num)],
            "Steam Type": ["Saturated"] * val_num,
            "P (bar g)": [10.0] * val_num,
            "T (°C)": [None] * val_num,
            "Flow (kg/h)": [3000.0] * val_num,
            "Target Vel (m/s)": [25.0] * val_num,
            "Length (m)": [100.0] * val_num,
            "Material": ["Carbon Steel (Standard Steam)"] * val_num,
            "Actual Dia (mm)": [80.0] * val_num
        })
        input_df = st.data_editor(
            val_df_init,
            column_config={
                "Steam Type": st.column_config.SelectboxColumn(options=["Superheated", "Saturated"]),
                "Material": st.column_config.SelectboxColumn(options=list(ROUGHNESS_MAP.keys()))
            },
            use_container_width=True
        )

    with tab2:
        st.markdown("**Upload Format Requirements:** Excel must contain columns exactly matching the manual table above.")
        template_csv = val_df_init.to_csv(index=False).encode('utf-8')
        st.download_button("Download Template (CSV)", data=template_csv, file_name="validation_template.csv", mime="text/csv")
        
        uploaded_file = st.file_uploader("Upload filled Excel/CSV file", type=['xlsx', 'csv'])
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    input_df = pd.read_csv(uploaded_file)
                else:
                    input_df = pd.read_excel(uploaded_file)
                st.success("File uploaded successfully! Click Validate below.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    if st.button("Validate Line Adequacy", type="primary"):
        results = []
        for i, row in input_df.iterrows():
            cid = row.get("Case ID", f"Line-{i+1}")
            stype = row["Steam Type"]
            P_bar = safe_float(row["P (bar g)"])
            T_c = safe_float(row.get("T (°C)", None))
            flow = safe_float(row["Flow (kg/h)"])
            t_vel = safe_float(row["Target Vel (m/s)"])
            L_m = safe_float(row["Length (m)"])
            mat = row.get("Material", "Carbon Steel (Standard Steam)")
            act_d = safe_float(row["Actual Dia (mm)"])
            
            rho, mu, vg = get_thermo_props(stype, P_bar, T_c)
            if rho is None:
                results.append({"Case": cid, "Remark": "Error: Missing Thermo Data"})
                continue
                
            mass_flow_kg_s = flow / 3600
            Q = mass_flow_kg_s / rho
            max_dp = min(P_bar * 0.10, 1.0)
            roughness = ROUGHNESS_MAP.get(mat, 0.045)
            
            # 1. Velocity Calc
            req_d_vel = 1000 * np.sqrt((4 * Q) / (np.pi * t_vel))
            std_d_vel = get_standard_pipe(req_d_vel)
            
            # 2. DP Calc
            std_d_dp = std_d_vel
            dp_dp, _, _, _, _ = calc_dp(std_d_dp, mass_flow_kg_s, rho, mu, L_m, roughness)
            while dp_dp > max_dp:
                idx = get_pipe_index(std_d_dp) + 1
                if idx >= len(STD_PIPES_MM): break
                std_d_dp = STD_PIPES_MM[idx]
                dp_dp, _, _, _, _ = calc_dp(std_d_dp, mass_flow_kg_s, rho, mu, L_m, roughness)
                
            # 3. Actual Line Checks
            act_dp, act_vel, _, _, _ = calc_dp(act_d, mass_flow_kg_s, rho, mu, L_m, roughness)
            
            # Remark Logic based on DP standard
            idx_act = get_pipe_index(act_d)
            idx_req = get_pipe_index(std_d_dp)
            
            if act_d < std_d_dp:
                status = "🔴 Undersized"
            elif idx_act > idx_req + 1:
                status = "🟡 Oversized"
            else:
                status = "🟢 Optimal"
                
            results.append({
                "Case ID": cid,
                "Calc Dia (Vel) mm": f"{std_d_vel:.1f}",
                "Calc Dia (DP) mm": f"{std_d_dp:.1f}",
                "Actual Dia mm": f"{act_d:.1f}",
                "Actual Vel (m/s)": f"{act_vel:.1f}",
                "Actual DP (bar)": f"{act_dp:.3f}",
                "Remark": status
            })
            
        res_df = pd.DataFrame(results)
        st.write("### Validation Report")
        
        def color_rules(val):
            if isinstance(val, str):
                if "Undersized" in val: return "background-color: #FADBD8; color: #900"
                if "Oversized" in val: return "background-color: #FCF3CF; color: #880"
                if "Optimal" in val: return "background-color: #D5F5E3; color: #080"
            return ""
            
        st.dataframe(res_df.style.map(color_rules, subset=["Remark"]), use_container_width=True)
        
        pdf_report = generate_validation_pdf(res_df)
        st.download_button("📥 Download PDF Report", data=pdf_report, file_name="Line_Adequacy_Report.pdf", mime="application/pdf")

# ---------------------------------------------------------
# MODE 3: DARCY-WEISBACH & MOODY CHART
# ---------------------------------------------------------
elif "Pressure Drop" in mode:
    st.title("Pressure Drop Sizing (Moody Chart)")
    st.markdown("Calculates optimal pipe size based on allowable pressure drop and plots the friction operating point.")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    stype_dw = col1.selectbox("Steam Type", ["Saturated", "Superheated"])
    P_in = col2.number_input("Inlet P (bar g)", value=10.0)
    T_in = col3.number_input("Inlet T (°C)", value=250.0) if stype_dw == "Superheated" else None
    flow = col4.number_input("Mass Flow (kg/h)", value=5000.0)
    target_vel = col5.number_input("Target Velocity (m/s)", value=25.0)
    
    colA, colB = st.columns(2)
    L_m = colA.number_input("Equivalent Pipe Length (m)", value=100.0)
    material = colB.selectbox("Pipe Material", list(ROUGHNESS_MAP.keys()))
    roughness = ROUGHNESS_MAP[material]
    
    if st.button("Calculate Sizing & Plot Moody Chart", type="primary"):
        with st.spinner("Calculating..."):
            max_dp_allowed = min(P_in * 0.10, 1.0)
            rho, mu, vg = get_thermo_props(stype_dw, P_in, T_in)
            
            if rho is None:
                st.error("Invalid Thermodynamics parameters.")
                st.stop()
                
            mass_flow_kg_s = flow / 3600
            Q = mass_flow_kg_s / rho
            
            # Initial Velocity Method
            d_req_vel = 1000 * np.sqrt((4 * Q) / (np.pi * target_vel))
            d_std_vel = get_standard_pipe(d_req_vel)
            dp_vel, v_vel, f_vel, re_vel, ed_vel = calc_dp(d_std_vel, mass_flow_kg_s, rho, mu, L_m, roughness)
            
            # Iterate for Pressure Drop
            d_std_dp = d_std_vel
            dp_dp, v_dp, f_dp, re_dp, ed_dp = dp_vel, v_vel, f_vel, re_vel, ed_vel
            
            while dp_dp > max_dp_allowed:
                idx = get_pipe_index(d_std_dp) + 1
                if idx >= len(STD_PIPES_MM): break
                d_std_dp = STD_PIPES_MM[idx]
                dp_dp, v_dp, f_dp, re_dp, ed_dp = calc_dp(d_std_dp, mass_flow_kg_s, rho, mu, L_m, roughness)

            comp_data = [
                {"Method": "Velocity Method", "Dia (mm)": d_std_vel, "Vel (m/s)": round(v_vel, 1), "DP (bar)": round(dp_vel, 3)},
                {"Method": "Pressure Drop Method", "Dia (mm)": d_std_dp, "Vel (m/s)": round(v_dp, 1), "DP (bar)": round(dp_dp, 3)}
            ]
            st.info(f"**Limit:** Max Allowable Pressure Drop is **{max_dp_allowed:.2f} bar**.")
            st.table(pd.DataFrame(comp_data))
            
            calc_dict = {'D_mm': d_std_dp, 'Vel': v_dp, 'Max_dp': max_dp_allowed, 'Actual_dp': dp_dp}
            fig, pdf_bytes = plot_moody_chart(re_dp, f_dp, ed_dp, calc_dict)
            st.pyplot(fig)
            st.download_button(label="📥 Download Moody Chart PDF", data=pdf_bytes, file_name="Moody_Chart.pdf", mime="application/pdf")
