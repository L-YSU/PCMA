
from pathlib import Path
import pandas as pd
import numpy as np
import pandapower as pp

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
IEEE = ROOT / 'IEEE_33_bus_Baran_Wu'
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"
OUT.mkdir(exist_ok=True)

def build_ieee33():
    buses = pd.read_csv(IEEE / 'bus_loads.csv')
    branches = pd.read_csv(IEEE / 'branches.csv')
    net = pp.create_empty_network(sn_mva=100.0)
    bus_map = {}
    for bus in buses['bus']:
        bus_map[int(bus)] = pp.create_bus(net, vn_kv=12.66, name=f'bus_{int(bus)}')
    pp.create_ext_grid(net, bus_map[1], vm_pu=1.0, name='slack')
    for _, row in branches.iterrows():
        pp.create_line_from_parameters(
            net,
            from_bus=bus_map[int(row['from_bus'])],
            to_bus=bus_map[int(row['to_bus'])],
            length_km=1.0,
            r_ohm_per_km=float(row['r_ohm']),
            x_ohm_per_km=float(row['x_ohm']),
            c_nf_per_km=0.0,
            max_i_ka=0.4,
            name=f"{int(row['from_bus'])}-{int(row['to_bus'])}"
        )
    for _, row in buses.iterrows():
        if int(row['bus']) == 1:
            continue
        pp.create_load(net, bus_map[int(row['bus'])], p_mw=float(row['p_kw'])/1000.0, q_mvar=float(row['q_kvar'])/1000.0, name=f"load_{int(row['bus'])}")
    return net, bus_map

def ramp_limited_smooth(raw, max_ramp_mw=0.35):
    target = pd.Series(raw).rolling(3, min_periods=1, center=True).mean().to_numpy()
    smoothed = np.zeros_like(target, dtype=float)
    smoothed[0] = target[0]
    for i in range(1, len(target)):
        delta = np.clip(target[i] - smoothed[i-1], -max_ramp_mw, max_ramp_mw)
        smoothed[i] = smoothed[i-1] + delta
    return smoothed

def run_case(ai_bus_no=33):
    scenario = pd.read_csv(OUT / 'ieee33_ai_datacenter_24h_scenario.csv')
    buses_base = pd.read_csv(IEEE / 'bus_loads.csv')
    raw = scenario['ai_load_mw'].to_numpy(dtype=float)
    smooth = ramp_limited_smooth(raw)
    records = []
    for mode in ['no_ai', 'raw_ai', 'smoothed_ai']:
        for h, row in scenario.iterrows():
            net, bus_map = build_ieee33()
            multiplier = float(row.get('background_load_multiplier', 1.0))
            # scale base loads by time-varying background multiplier
            net.load['p_mw'] *= multiplier
            net.load['q_mvar'] *= multiplier
            if mode == 'raw_ai':
                pp.create_load(net, bus_map[ai_bus_no], p_mw=float(raw[h]), q_mvar=0.25*float(raw[h]), name='ai_datacenter_raw')
            elif mode == 'smoothed_ai':
                pp.create_load(net, bus_map[ai_bus_no], p_mw=float(smooth[h]), q_mvar=0.25*float(smooth[h]), name='ai_datacenter_smoothed')
            try:
                pp.runpp(net, algorithm='bfsw', max_iteration=100, tolerance_mva=1e-6)
                converged = bool(net.converged)
                min_vm = float(net.res_bus.vm_pu.min()) if converged else np.nan
                max_line_loading = float(net.res_line.loading_percent.max()) if converged else np.nan
                overloaded_lines = int((net.res_line.loading_percent > 100).sum()) if converged else np.nan
                undervoltage_buses = int((net.res_bus.vm_pu < 0.95).sum()) if converged else np.nan
                losses_mw = float(net.res_line.pl_mw.sum()) if converged else np.nan
                slack_mw = float(net.res_ext_grid.p_mw.sum()) if converged else np.nan
            except Exception as e:
                converged = False
                min_vm = max_line_loading = losses_mw = slack_mw = np.nan
                overloaded_lines = undervoltage_buses = np.nan
            records.append({
                'ai_bus': ai_bus_no,
                'hour': int(h),
                'mode': mode,
                'converged': converged,
                'base_load_multiplier': multiplier,
                'ai_load_raw_mw': float(raw[h]) if mode != 'no_ai' else 0.0,
                'ai_load_grid_seen_mw': 0.0 if mode == 'no_ai' else (float(raw[h]) if mode == 'raw_ai' else float(smooth[h])),
                'battery_or_ups_compensation_mw': 0.0 if mode != 'smoothed_ai' else float(raw[h] - smooth[h]),
                'min_voltage_pu': min_vm,
                'max_line_loading_percent': max_line_loading,
                'overloaded_line_count': overloaded_lines,
                'undervoltage_bus_count_lt_0p95': undervoltage_buses,
                'line_losses_mw': losses_mw,
                'slack_power_mw': slack_mw,
            })
    return pd.DataFrame(records)

if __name__ == '__main__':
    all_results = pd.concat([run_case(18), run_case(33)], ignore_index=True)
    all_results.to_csv(OUT / 'ieee33_ai_load_powerflow_results.csv', index=False, encoding='utf-8-sig')
    summary = all_results.groupby(['ai_bus','mode']).agg(
        converged_hours=('converged','sum'),
        min_voltage_min=('min_voltage_pu','min'),
        min_voltage_mean=('min_voltage_pu','mean'),
        max_line_loading_max=('max_line_loading_percent','max'),
        overloaded_line_hours=('overloaded_line_count','sum'),
        undervoltage_bus_hours=('undervoltage_bus_count_lt_0p95','sum'),
        losses_mw_mean=('line_losses_mw','mean'),
        slack_mw_peak=('slack_power_mw','max'),
        ai_grid_seen_mw_peak=('ai_load_grid_seen_mw','max'),
    ).reset_index()
    summary.to_csv(OUT / 'ieee33_ai_load_powerflow_summary.csv', index=False, encoding='utf-8-sig')
    print(summary.to_string(index=False))
