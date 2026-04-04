import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    from rma_kinetics.models.constitutive import erasable
    from rma_kinetics.solvers import Dopri5, Kvaerno3

    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt
    from pytensor.compile.ops import wrap_py

    import polars as pl
    import numpy as np
    import seaborn as sb
    import matplotlib.pyplot
    import marimo as mo
    import os

    from constitutive_fit import get_df

    return Kvaerno3, az, erasable, get_df, mo, np, os, pl, pm, pt, sb, wrap_py


@app.cell
def _(erasable):
    tev_schedule = erasable.create_tev_schedule(11, 168, repeat=1, interval=336)
    model = erasable.Model(tev_schedule, 0.4, 0.6, 0.007, .0015, 180, 0.5)
    return (model,)


@app.cell
def _(Kvaerno3, erasable, model):
    solution = model.solve(0, 504.5, 0.5, erasable.State(), Kvaerno3())
    return (solution,)


@app.cell
def _():
    import matplotlib.pyplot as plt

    return (plt,)


@app.cell
def _(plt, solution):
    plt.plot(solution.ts, solution.plasma_rma, 'k')
    #plt.vlines([168, 504], 0, 25, linestyles='--', color="lightgrey")
    plt.xlabel("Time (hr)")
    plt.ylabel("Concentration (nM)")
    plt.tight_layout()
    plt.gca()
    return


@app.cell
def _(mo):
    dataset_id = mo.ui.radio(options=["hippocampus", "midbrain", "striatum"], value="hippocampus", label="RMA Timecourse Dataset")
    dataset_id
    return (dataset_id,)


@app.cell
def _(dataset_id, os):
    data_dir = os.path.join("notebooks", "data", "constitutive_ferma", dataset_id.value)
    return (data_dir,)


@app.cell
def _(data_dir, dataset_id, get_df, plt, sb):
    df, summary_df = get_df(dataset_id.value, data_dir)
    df_plot = df.unpivot(
        on=["rlu", "concentration"],
        index="time",
        variable_name="output_type",
        value_name="value",
    )

    grid = sb.FacetGrid(data=df_plot, col="output_type", sharey=False)
    grid.map(sb.pointplot, "time", "value", order=[0, 168, 168.5, 504, 504.5])

    grid.axes[0][0].set_yscale("log")
    grid.axes[0][0].set_ylabel("RLU (a.u.)")
    grid.axes[0][0].set_xlabel("Time (hr)")
    grid.axes[0][0].set_title("")

    grid.axes[0][1].set_ylabel("Concentration (nM)")
    grid.axes[0][1].set_xlabel("Time (hr)")
    grid.axes[0][1].set_title("")

    grid.fig.suptitle(f"AAV RMA timecourse - {dataset_id.value}")
    plt.tight_layout()
    plt.gcf()
    return (df,)


@app.cell
def _(df, np, pl):
    fit_df = (
        df.select(["mouse_id", "time", "concentration"])
        .with_columns([
            pl.col("mouse_id").cast(pl.Utf8),
            pl.col("time").cast(pl.Float64),
            pl.col("concentration").cast(pl.Float64),
        ]).sort(["mouse_id", "time"])
    )

    obs_plasma_rma = fit_df["concentration"].to_numpy().astype(float)
    obs_time = fit_df["time"].to_numpy().astype(float)
    n_obs = obs_plasma_rma.size
    mouse_id = fit_df["mouse_id"].to_numpy().astype(int)
    n_mice = fit_df.group_by(pl.col("mouse_id")).n_unique().height
    tf = float(np.max(obs_time))

    fit_df
    return mouse_id, n_mice, n_obs, obs_plasma_rma, obs_time, tf


@app.cell
def _(
    Kvaerno3,
    erasable,
    mouse_id,
    n_mice,
    n_obs,
    np,
    obs_time,
    pt,
    tf,
    wrap_py,
):
    fit_tev_schedule = erasable.create_tev_schedule(11.4285714286, start_time=168, repeat=1, interval=336)

    @wrap_py(
        itypes=[
            pt.dvector, pt.dscalar, pt.dscalar,
            pt.dscalar, pt.dscalar, pt.dscalar],
        otypes=[pt.dvector]
    )
    def expectation(
        log_prod_mouse,
        log_bbb,
        log_deg,
        log_tev_vd,
        log_tev_deg,
        log_tev_cut
    ):
        rma_prod_mouse = np.exp(np.asarray(log_prod_mouse), dtype=float)
        bbb_transport = float(np.exp(log_bbb))
        deg = float(np.exp(log_deg))
        tev_vd = float(np.exp(log_tev_vd))
        tev_deg = float(np.exp(log_tev_deg))
        tev_cut = float(np.exp(log_tev_cut))

        pred = np.empty(n_obs, dtype=float)

        for m in range(n_mice):
            mask = mouse_id == m

            if not np.any(mask):
                continue

            model = erasable.Model(
                fit_tev_schedule,
                rma_prod_mouse[m],
                bbb_transport,
                deg,
                tev_vd,
                tev_deg, 
                tev_cut
            )

            try:
                solution = model.solve(0, tf, 0.5, erasable.State(), Kvaerno3())
                pred[mask] = np.interp(obs_time[mask], solution.ts, solution.plasma_rma)
                # pred[mask] = solution.plasma_rma[obs_time[mask]]
            except ValueError:
                continue

        return pred

    return (expectation,)


@app.cell
def _(expectation, n_mice, n_obs, np, obs_plasma_rma, pm):
    coords = {
        "mouse": np.arange(n_mice),
        "obs_id": np.arange(n_obs),
    }

    with pm.Model(coords=coords):
        mean_log_prod = pm.Normal("mu_log_prod", mu=np.log(0.4), sigma=0.35)
        log_prod_mouse = pm.Normal(
            "log_prod_mouse",
            mu=mean_log_prod,
            sigma=0.5,
            dims="mouse",
        )

        # population level priors (log-normal parameterized)
        log_bbb = pm.Normal("log_bbb", mu=np.log(0.6), sigma=0.15)
        log_deg = pm.Normal("log_deg", mu=np.log(0.007), sigma=0.36)
        var_obs = pm.HalfNormal("sigma_obs", sigma=0.3)
        # estimate about plasma volume
        log_tev_vd = pm.Normal("log_tev_vd", mu=np.log(0.0015), sigma=0.1)
        log_tev_deg = pm.Normal("log_tev_deg", mu=np.log(180), sigma=0.25)
        # kcat ~ 0.1-0.3 1/s and km ~ 20-50 µM
        log_tev_cut = pm.Normal("log_tev_cut", mu=np.log(0.5), sigma=0.1)

        mean_plasma_rma = pm.Deterministic(
            "mu",
            expectation(log_prod_mouse, log_bbb, log_deg, log_tev_vd, log_tev_deg, log_tev_cut),
            dims="obs_id"
        )

        pm.Normal(
           "y",
            mu=mean_plasma_rma,
            sigma=var_obs,
            observed=obs_plasma_rma,
            dims="obs_id"
        )

        idata = pm.sample(draws=100, tune=100, chains=6, cores=4, random_seed=42, step=pm.DEMetropolisZ(), return_inferencedata=True)

        ppc = pm.sample_posterior_predictive(
            idata,
            var_names=["y"],
            random_seed=42,
            return_inferencedata=True
        )
    return (idata,)


@app.cell
def _(az, idata):
    summary = az.summary(
        idata,
        var_names=[
            "mu_log_prod",
            "log_bbb",
            "log_deg",
            "log_tev_vd",
            "log_tev_deg",
            "log_tev_cut",
        ],
        round_to=4,
    )
    summary
    return


@app.cell
def _():
    # 2mg/ml - 160µL TEV dose
    # ~320 ng
    # 27,000 g/mol * 10^9 = ng/mol / 10^9 = ng/nmol
    # 0.0118518519 nmol
    return


@app.cell
def _():
    # rma prod - 0.3
    # bbb transport - 0.6
    # deg 0.007
    # tev amount is fixed
    # tev degradation - 
    # tev cut rate - 
    return


if __name__ == "__main__":
    app.run()
