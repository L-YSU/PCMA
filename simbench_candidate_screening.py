from pathlib import Path

import pandas as pd
import pandapower as pp
import simbench as sb


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ROOT = PACKAGE_ROOT / "datasets"
OUT = PACKAGE_ROOT / "results" / "00_analysis_outputs"


def main():
    codes = sb.collect_all_simbench_codes()
    preferred = [
        code
        for code in codes
        if ("MVLV" in code or "HVMV" in code or "-LV" in code)
        and "mixed" in code
        and code.endswith("-no_sw")
    ]
    preferred += [
        code
        for code in codes
        if ("MVLV" in code or "HVMV" in code or "-LV" in code)
        and "mixed" in code
        and code.endswith("-sw")
    ]
    if not preferred:
        preferred = codes[:10]

    records = []
    for code in preferred[:20]:
        try:
            net = sb.get_simbench_net(code)
            pp.runpp(
                net,
                algorithm="nr",
                numba=False,
                max_iteration=100,
                calculate_voltage_angles=False,
                check_connectivity=True,
            )
            records.append(
                {
                    "code": code,
                    "bus_count": len(net.bus),
                    "line_count": len(net.line),
                    "load_count": len(net.load),
                    "sgen_count": len(net.sgen),
                    "trafo_count": len(net.trafo),
                    "converged": bool(net.converged),
                    "min_voltage_pu": float(net.res_bus.vm_pu.min()),
                    "max_line_loading_percent": float(net.res_line.loading_percent.max())
                    if len(net.line)
                    else 0.0,
                }
            )
        except Exception as exc:
            records.append({"code": code, "error": repr(exc)})

    df = pd.DataFrame(records)
    df.to_csv(OUT / "simbench_candidate_screening.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
