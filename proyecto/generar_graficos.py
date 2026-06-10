from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "figures"
TABLES_DIR = BASE_DIR / "tables"

RANDOM_SEED = 20260610
N_CANDIDATES = 1500
ACCEPTANCE_RATE = 0.05
INFECTIOUS_WINDOW_DAYS = 14
SMOOTHING_WINDOW_DAYS = 7
POLICY_LAG_DAYS = 14
MAX_WEIGHT = 0.8
MAX_RT = 1.25

COUNTRY_ORDER = ["AUS", "BRA", "CAN", "CHN", "GBR", "IND", "USA"]
COUNTRY_LABELS = {
    "AUS": "Australia",
    "BRA": "Brasil",
    "CAN": "Canadá",
    "CHN": "China",
    "GBR": "Reino Unido",
    "IND": "India",
    "USA": "Estados Unidos",
}

POLICY_COLUMNS = {
    "Cierre escuelas": ("C1E_School closing", "C1E_School.closing"),
    "Cierre trabajo": ("C2E_Workplace closing", "C2E_Workplace.closing"),
    "Eventos públicos": ("C3E_Cancel public events", "C3E_Cancel.public.events"),
    "Reuniones": ("C4E_Restrictions on gatherings", "C4E_Restrictions.on.gatherings"),
    "Quedarse en casa": ("C6E_Stay at home requirements", "C6E_Stay.at.home.requirements"),
    "Movilidad interna": (
        "C7E_Restrictions on internal movement",
        "C7E_Restrictions.on.internal.movement",
    ),
    "Viajes internacionales": (
        "C8E_International travel controls",
        "C8E_International.travel.controls",
    ),
    "Mascarillas": ("H6E_Facial Coverings", "H6E_Facial.Coverings"),
    "Vacunación": ("H7_Vaccination policy", "H7_Vaccination.policy"),
}


def read_national_data() -> pd.DataFrame:
    columns = [
        "CountryName",
        "CountryCode",
        "Jurisdiction",
        "Date",
        "ConfirmedCases",
        "ConfirmedDeaths",
    ]
    frames: list[pd.DataFrame] = []

    for path in sorted(DATA_DIR.glob("OxCGRT_fullwithnotes_*_v1.csv")):
        header = pd.read_csv(path, nrows=0).columns.tolist()
        policy_usecols = []
        policy_renames = {}
        for label, variants in POLICY_COLUMNS.items():
            source = next((variant for variant in variants if variant in header), None)
            if source is not None:
                policy_usecols.append(source)
                policy_renames[source] = label

        df = pd.read_csv(
            path,
            usecols=[*columns, *policy_usecols],
            dtype={"Date": str},
            low_memory=False,
        )
        df = df.rename(columns=policy_renames)
        for label in POLICY_COLUMNS:
            if label not in df.columns:
                df[label] = 0.0
        df = df[df["Jurisdiction"].eq("NAT_TOTAL")].copy()
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No se encontraron CSV en {DATA_DIR}")

    data = pd.concat(frames, ignore_index=True)
    data["Date"] = pd.to_datetime(data["Date"], format="%Y%m%d")
    data["ConfirmedCases"] = pd.to_numeric(data["ConfirmedCases"], errors="coerce")
    data["ConfirmedDeaths"] = pd.to_numeric(data["ConfirmedDeaths"], errors="coerce")
    for column in POLICY_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.sort_values(["CountryCode", "Date"]).reset_index(drop=True)
    return data


def add_variables(data: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in data.groupby("CountryCode", sort=False):
        group = group.sort_values("Date").copy()
        group["new_cases"] = group["ConfirmedCases"].diff().fillna(0).clip(lower=0)
        group["new_deaths"] = group["ConfirmedDeaths"].diff().fillna(0).clip(lower=0)
        group["new_cases_ma7"] = (
            group["new_cases"].rolling(SMOOTHING_WINDOW_DAYS, min_periods=1).mean()
        )
        group["infectious_obs"] = (
            group["new_cases"]
            .shift(1)
            .rolling(INFECTIOUS_WINDOW_DAYS, min_periods=1)
            .sum()
            .fillna(0)
        )
        group["rt_obs"] = group["new_cases_ma7"] / group["infectious_obs"].clip(lower=1)
        group["rt_obs"] = group["rt_obs"].replace([np.inf, -np.inf], np.nan).fillna(0)

        for label in POLICY_COLUMNS:
            max_value = group[label].max(skipna=True)
            if pd.isna(max_value) or max_value <= 0:
                group[f"policy_{label}"] = 0.0
            else:
                group[f"policy_{label}"] = (
                    group[label].ffill().bfill().fillna(0).clip(lower=0) / max_value
                )

        group["CountryLabel"] = group["CountryCode"].map(COUNTRY_LABELS).fillna(
            group["CountryName"]
        )
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def expected_curve_from_weights(
    infectious: np.ndarray,
    policy_matrix: np.ndarray,
    beta: np.ndarray,
    alpha: float,
    start_idx: int,
    max_cases: float,
) -> np.ndarray:
    expected = np.full(len(infectious), np.nan)
    rt_path = np.exp(alpha - policy_matrix[start_idx:] @ beta)
    rt_path = np.clip(rt_path, 0.0, MAX_RT)
    expected[start_idx:] = np.minimum(rt_path * infectious[start_idx:], max_cases)

    return expected


def score_curve(observed: np.ndarray, expected: np.ndarray, start_idx: int) -> float:
    valid = np.isfinite(expected[start_idx:])
    if not valid.any():
        return float("inf")
    observed_log = np.log1p(observed[start_idx:][valid])
    expected_log = np.log1p(expected[start_idx:][valid])
    return float(np.sqrt(np.mean((expected_log - observed_log) ** 2)))


def calibrate_country(group: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    group = group.sort_values("Date").copy()
    policy_cols = [f"policy_{label}" for label in POLICY_COLUMNS]
    policy_matrix = group[policy_cols].shift(POLICY_LAG_DAYS).fillna(0).to_numpy(dtype=float)
    observed = group["new_cases_ma7"].to_numpy(dtype=float)
    infectious = group["infectious_obs"].to_numpy(dtype=float)
    rt_obs = group["rt_obs"].to_numpy(dtype=float)
    start_idx = max(INFECTIOUS_WINDOW_DAYS, POLICY_LAG_DAYS)
    max_cases = max(float(np.nanmax(observed)) * 3.0, 1.0)

    valid = np.isfinite(rt_obs) & (rt_obs > 0) & (np.arange(len(group)) >= start_idx)
    if valid.sum() < 30:
        raise ValueError(f"No hay suficientes datos válidos para {group['CountryCode'].iloc[0]}")

    candidates = []
    curves = []
    for candidate_idx in range(N_CANDIDATES):
        beta = rng.uniform(0.0, MAX_WEIGHT, size=len(POLICY_COLUMNS))
        alpha = float(np.mean(np.log(rt_obs[valid]) + policy_matrix[valid] @ beta))
        expected = expected_curve_from_weights(
            infectious=infectious,
            policy_matrix=policy_matrix,
            beta=beta,
            alpha=alpha,
            start_idx=start_idx,
            max_cases=max_cases,
        )
        score = score_curve(observed, expected, start_idx)
        candidates.append(
            {
                "CountryCode": group["CountryCode"].iloc[0],
                "Zona": group["CountryLabel"].iloc[0],
                "candidate_idx": candidate_idx,
                "alpha": alpha,
                "score": score,
                **{label: beta[idx] for idx, label in enumerate(POLICY_COLUMNS)},
            }
        )
        curves.append(expected)

    candidates_df = pd.DataFrame(candidates).sort_values("score").reset_index(drop=True)
    accepted_count = max(1, int(N_CANDIDATES * ACCEPTANCE_RATE))
    accepted = candidates_df.head(accepted_count).copy()
    best = accepted.iloc[0]
    best_beta = best[list(POLICY_COLUMNS)].to_numpy(dtype=float)
    best_alpha = float(best["alpha"])
    best_curve = curves[int(best["candidate_idx"])]
    ideal_policy_matrix = policy_matrix.copy()
    ideal_policy_matrix[start_idx:] = 1.0
    ideal_curve = expected_curve_from_weights(
        infectious=infectious,
        policy_matrix=ideal_policy_matrix,
        beta=best_beta,
        alpha=best_alpha,
        start_idx=start_idx,
        max_cases=max_cases,
    )

    curve_df = group[
        ["CountryCode", "CountryLabel", "Date", "new_cases", "new_cases_ma7"]
    ].copy()
    curve_df["best_expected"] = best_curve
    curve_df["ideal_expected"] = ideal_curve
    curve_df["start_idx"] = start_idx
    curve_df["best_score"] = float(accepted["score"].iloc[0])
    return accepted, curve_df


def calibrate_all(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    accepted_frames = []
    curve_frames = []
    error_rows = []

    for country_code in COUNTRY_ORDER:
        group = data[data["CountryCode"].eq(country_code)]
        accepted, curve = calibrate_country(group, rng)
        accepted_frames.append(accepted)
        curve_frames.append(curve)

        start_idx = int(curve["start_idx"].iloc[0])
        observed = curve["new_cases_ma7"].to_numpy(dtype=float)
        expected = curve["best_expected"].to_numpy(dtype=float)
        ideal = curve["ideal_expected"].to_numpy(dtype=float)
        valid = np.isfinite(expected) & (np.arange(len(curve)) >= start_idx)
        error = expected[valid] - observed[valid]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        mape_mask = observed[valid] >= 1
        mape = float(np.mean(np.abs(error[mape_mask]) / observed[valid][mape_mask]) * 100)
        expected_total = float(np.sum(expected[valid]))
        ideal_total = float(np.sum(ideal[valid]))
        ideal_reduction = (
            (expected_total - ideal_total) / expected_total * 100 if expected_total > 0 else 0.0
        )

        error_rows.append(
            {
                "Zona": curve["CountryLabel"].iloc[0],
                "RMSE": rmse,
                "MAE": mae,
                "MAPE": mape,
                "Score log": float(accepted["score"].iloc[0]),
                "Aceptadas": len(accepted),
                "Reduccion ideal": ideal_reduction,
            }
        )

    return (
        pd.concat(accepted_frames, ignore_index=True),
        pd.concat(curve_frames, ignore_index=True),
        pd.DataFrame(error_rows),
    )


def make_zone_table(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in data.groupby("CountryCode", sort=False):
        rows.append(
            {
                "Zona": group["CountryLabel"].iloc[0],
                "Codigo": group["CountryCode"].iloc[0],
                "Inicio": group["Date"].min(),
                "Fin": group["Date"].max(),
                "Dias": len(group),
                "Casos acumulados": group["ConfirmedCases"].max(),
            }
        )
    return pd.DataFrame(rows)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_table(path: Path, headers: list[str], rows: list[list[object]]) -> None:
    column_spec = "l" + "r" * (len(headers) - 1)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(item) for item in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_tables(data: pd.DataFrame, accepted: pd.DataFrame, errors: pd.DataFrame) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    zones = make_zone_table(data)
    write_latex_table(
        TABLES_DIR / "zonas.tex",
        ["Zona", "Código", "Inicio", "Fin", "Días", "Casos acumulados"],
        [
            [
                row["Zona"],
                row["Codigo"],
                row["Inicio"].strftime("%Y-%m-%d"),
                row["Fin"].strftime("%Y-%m-%d"),
                f"{int(row['Dias']):,}",
                f"{int(row['Casos acumulados']):,}",
            ]
            for _, row in zones.iterrows()
        ],
    )

    write_latex_table(
        TABLES_DIR / "errores.tex",
        ["Zona", "RMSE", "MAE", "MAPE", "Score log", "Aceptadas", "Red. ideal"],
        [
            [
                row["Zona"],
                f"{row['RMSE']:,.1f}",
                f"{row['MAE']:,.1f}",
                f"{row['MAPE']:,.1f}%",
                f"{row['Score log']:.3f}",
                f"{int(row['Aceptadas'])}",
                f"{row['Reduccion ideal']:.1f}%",
            ]
            for _, row in errors.iterrows()
        ],
    )

    weight_rows = []
    for label in POLICY_COLUMNS:
        values = accepted[label].to_numpy(dtype=float)
        weight_rows.append(
            [
                label,
                f"{np.mean(values):.3f}",
                f"{np.percentile(values, 5):.3f}",
                f"{np.percentile(values, 95):.3f}",
            ]
        )
    write_latex_table(
        TABLES_DIR / "pesos.tex",
        ["Medida", "Media", "P5", "P95"],
        weight_rows,
    )


def setup_axes_grid() -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=False)
    return fig, axes.flatten()


def format_year_axis(axis: plt.Axes) -> None:
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.grid(True, alpha=0.25)


def plot_series_cases(data: pd.DataFrame) -> None:
    fig, axes = setup_axes_grid()
    for axis, country_code in zip(axes, COUNTRY_ORDER):
        group = data[data["CountryCode"].eq(country_code)]
        axis.plot(group["Date"], group["new_cases"], color="#9ca3af", linewidth=0.8, alpha=0.65)
        axis.plot(group["Date"], group["new_cases_ma7"], color="#2563eb", linewidth=1.5)
        axis.set_title(group["CountryLabel"].iloc[0])
        axis.set_ylabel("Casos nuevos")
        format_year_axis(axis)

    axes[-1].axis("off")
    fig.suptitle("Casos nuevos de COVID-19 por zona", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "series_casos.png", dpi=180)
    plt.close(fig)


def plot_best_curves(curves: pd.DataFrame) -> None:
    fig, axes = setup_axes_grid()
    for axis, country_code in zip(axes, COUNTRY_ORDER):
        group = curves[curves["CountryCode"].eq(country_code)]
        start_idx = int(group["start_idx"].iloc[0])
        axis.plot(group["Date"], group["new_cases_ma7"], color="#111827", linewidth=1.1)
        axis.plot(group["Date"], group["best_expected"], color="#dc2626", linewidth=1.2)
        axis.axvline(group["Date"].iloc[start_idx], color="#6b7280", linestyle="--", linewidth=0.8)
        axis.set_title(group["CountryLabel"].iloc[0])
        axis.set_ylabel("Casos nuevos")
        format_year_axis(axis)

    axes[-1].axis("off")
    fig.suptitle("Curva observada vs. mejor simulación aceptada", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mejores_simulaciones.png", dpi=180)
    plt.close(fig)


def plot_ideal_scenarios(curves: pd.DataFrame) -> None:
    fig, axes = setup_axes_grid()
    for axis, country_code in zip(axes, COUNTRY_ORDER):
        group = curves[curves["CountryCode"].eq(country_code)]
        start_idx = int(group["start_idx"].iloc[0])
        axis.plot(group["Date"], group["best_expected"], color="#dc2626", linewidth=1.1)
        axis.plot(group["Date"], group["ideal_expected"], color="#059669", linewidth=1.2)
        axis.axvline(group["Date"].iloc[start_idx], color="#6b7280", linestyle="--", linewidth=0.8)
        axis.set_title(group["CountryLabel"].iloc[0])
        axis.set_ylabel("Casos nuevos esperados")
        format_year_axis(axis)

    axes[-1].axis("off")
    fig.suptitle("Escenario calibrado vs. aplicación ideal de medidas", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "escenarios_ideales.png", dpi=180)
    plt.close(fig)


def plot_ideal_reductions(errors: pd.DataFrame) -> None:
    ordered = errors.sort_values("Reduccion ideal")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.barh(ordered["Zona"], ordered["Reduccion ideal"], color="#059669", alpha=0.85)
    axis.set_xlabel("Reducción acumulada esperada (%)")
    axis.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Reducción estimada bajo escenario ideal", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "reduccion_ideal.png", dpi=180)
    plt.close(fig)


def plot_weight_summary(accepted: pd.DataFrame) -> None:
    summary = []
    for label in POLICY_COLUMNS:
        values = accepted[label].to_numpy(dtype=float)
        summary.append(
            {
                "Medida": label,
                "Media": float(np.mean(values)),
                "P5": float(np.percentile(values, 5)),
                "P95": float(np.percentile(values, 95)),
            }
        )
    summary_df = pd.DataFrame(summary).sort_values("Media")

    fig, axis = plt.subplots(figsize=(9, 6))
    y_pos = np.arange(len(summary_df))
    axis.barh(
        y_pos,
        summary_df["Media"],
        xerr=[
            summary_df["Media"] - summary_df["P5"],
            summary_df["P95"] - summary_df["Media"],
        ],
        color="#2563eb",
        alpha=0.85,
    )
    axis.set_yticks(y_pos, summary_df["Medida"])
    axis.set_xlabel("Peso aceptado")
    axis.grid(True, axis="x", alpha=0.25)
    fig.suptitle("Pesos de medidas en simulaciones aceptadas", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pesos_medidas.png", dpi=180)
    plt.close(fig)


def write_outputs(
    data: pd.DataFrame,
    accepted: pd.DataFrame,
    curves: pd.DataFrame,
    errors: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    write_tables(data, accepted, errors)
    plot_series_cases(data)
    plot_best_curves(curves)
    plot_ideal_scenarios(curves)
    plot_ideal_reductions(errors)
    plot_weight_summary(accepted)


def main() -> None:
    print("Leyendo datos nacionales y medidas sanitarias...")
    raw = read_national_data()
    print("Calculando variables epidemiológicas...")
    data = add_variables(raw)
    print(
        f"Evaluando {N_CANDIDATES} combinaciones aleatorias de pesos por zona "
        f"y aceptando el {ACCEPTANCE_RATE:.0%} con menor error..."
    )
    accepted, curves, errors = calibrate_all(data)
    print("Escribiendo figuras y tablas...")
    write_outputs(data, accepted, curves, errors)
    print(f"Listo. Figuras en {FIGURES_DIR} y tablas en {TABLES_DIR}.")


if __name__ == "__main__":
    main()
