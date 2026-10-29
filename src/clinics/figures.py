"""Figures for the README, regenerated from data/processed on every update.

    python -m clinics.figures --data data/processed --out docs/figures

Every figure uses the shared style (figstyle.py): the title states the finding, computed from
the data on each run so it stays true as quarters are added; milestones and findings are drawn
on the chart; source and method go in the footer; each figure is saved for light and dark
backgrounds (<name>-light.png / -dark.png).

Series are as reported. A period a later filing restated gets a hollow ring, and the indicator
recomputed with the restated figures is a separate diamond (balances in any quarter, flows only
in Q4), when it moves the indicator by more than 5%. Integrated groups (Isapre + clinics) are dashed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import Quarter
from .figstyle import charts
from .issuers import load_issuers
from .sequence import INDICATORS as SEQ_INDICATORS
from .sequence import RULING

ORDER = ["clc", "indisa", "redsalud", "banmedica", "bupa"]  # fixed palette slot per issuer
EVENTS = [("2022Q4", "Supreme Court ruling", 0.98), ("2024Q2", "Isapre short law", 0.90)]
START = "2016Q1"
MATERIAL = 0.05  # revised point shown when the restatement moves the indicator by more than 5%
SOURCE = "Source: CMF XBRL filings, as reported; github.com/IgnacioMendozaRojas/chile-clinics-benchmark."


def _x(period: str) -> int:
    q = Quarter.parse(period)
    return q.year * 4 + q.q - 1


def _pct(v: float) -> str:
    return f"{v:.1%}"


def revised_indicators(statements: pd.DataFrame) -> pd.DataFrame:
    """Indicators recomputed with restated figures, where a later filing restated a component.

    Balances use the revised figure in any quarter; flows only in Q4 (LTM = the restated year).
    Unrevised components keep their as-reported value.
    """
    s = statements.copy()
    q4 = s["period"].str.endswith("Q4")
    base = np.where(s["kind"] == "flow", s["ltm"], s["value"])
    usable = s["restated"] & s["revised"].notna() & ((s["kind"] == "stock") | q4)
    s["asrep"] = base
    s["rev"] = np.where(usable, s["revised"], base)
    s["hit"] = usable
    a = s.pivot_table(index=["issuer", "period"], columns="variable", values="asrep", aggfunc="first")
    r = s.pivot_table(index=["issuer", "period"], columns="variable", values="rev", aggfunc="first")
    hit = s.pivot_table(index=["issuer", "period"], columns="variable", values="hit", aggfunc="max").fillna(False)

    def col(frame, name, zero=False):
        c = frame[name] if name in frame else pd.Series(np.nan, index=frame.index)
        return c.fillna(0.0) if zero else c

    def ebitda(f):
        return (
            col(f, "gross_profit")
            - col(f, "distribution_costs", True)
            - col(f, "administrative_expense")
            - col(f, "impairment_credit_losses", True)
            + col(f, "depreciation_amortisation")
        )

    formulas = {
        "dso": (
            lambda f: col(f, "trade_receivables_current") / col(f, "revenue") * 365,
            ["trade_receivables_current", "revenue"],
        ),
        "current_ratio": (
            lambda f: col(f, "current_assets") / col(f, "current_liabilities"),
            ["current_assets", "current_liabilities"],
        ),
        "ebitda_margin": (
            lambda f: ebitda(f) / col(f, "revenue"),
            ["gross_profit", "administrative_expense", "depreciation_amortisation", "revenue"],
        ),
    }
    out = []
    for name, (fn, parts) in formulas.items():
        touched = hit.reindex(columns=parts).fillna(False).any(axis=1)
        before, after = fn(a)[touched], fn(r)[touched]
        changed = (after - before).abs() > MATERIAL * before.abs()
        rows = pd.DataFrame({"as_reported": before[changed], "revised": after[changed]}).reset_index()
        out.append(rows.assign(indicator=name))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── data helpers ───────────────────────────────────────────────────────────────


def _series(ind: pd.DataFrame, indicator: str, issuer: str) -> pd.Series:
    s = ind[(ind["indicator"] == indicator) & (ind["issuer"] == issuer) & (ind["period"] >= START)]
    return s.set_index("period")["value"].sort_index()


def _core(issuers) -> list[str]:
    return [k for k in ORDER if k in issuers and issuers[k].panel == "core"]


def _rise(ind, indicator, issuers, sign=+1):
    """(issuer, from period, from value, to period, to value): the core issuer with the largest move
    in the adverse direction from the Supreme Court ruling quarter to its extreme afterwards."""
    best = None
    for k in _core(issuers):
        s = _series(ind, indicator, k)
        after = s[s.index >= RULING]
        if after.empty or RULING not in s.index:
            continue
        p1 = after.idxmax() if sign > 0 else after.idxmin()
        move = (after[p1] - s[RULING]) * sign
        if best is None or move > best[0]:
            best = (move, k, RULING, s[RULING], p1, after[p1])
    return best[1:] if best else None


# ── drawing ────────────────────────────────────────────────────────────────────


def _axes(c, last: str, first: str = START):
    years = range(int(first[:4]), int(last[:4]) + 1)
    c.ax.set_xticks([y * 4 for y in years], [str(y) for y in years])
    c.ax.set_xlim(_x(first) - 1, _x(last) + 7)
    for period, label, y in EVENTS:
        c.event(_x(period), label, y_frac=y)


def _nudges(ends: dict, gap: float) -> dict:
    """Offsets in points that keep end labels at least `gap` (data units) apart."""
    out, prev = {}, None
    for k, v in sorted(ends.items(), key=lambda kv: kv[1]):
        out[k] = 0.0
        if prev is not None and v - prev[1] < gap:
            out[k] = out[prev[0]] + 9.0
        prev = (k, v)
    return out


def _lines(c, ind, indicator, issuers, scale=1.0, labels=None):
    last = ind[ind["indicator"] == indicator]["period"].max()
    data = ind[(ind["indicator"] == indicator) & (ind["period"] >= START)]["value"] * scale
    series = {k: _series(ind, indicator, k) for k in ORDER}
    ends = {k: s.iloc[-1] * scale for k, s in series.items() if not s.empty and s.index[-1] == last}
    dy = _nudges(ends, 0.05 * (data.max() - data.min()))
    for k, s in series.items():
        if s.empty:
            continue
        integrated = issuers[k].type == "integrated"
        label = (labels or {}).get(k, issuers[k].short + (" (integrated)" if integrated else ""))
        c.line(
            np.array([_x(p) for p in s.index]),
            s.to_numpy() * scale,
            ORDER.index(k),
            label,
            end_label=s.index[-1] == last,
            dy=dy.get(k, 0.0),
            linestyle=(0, (4, 2)) if integrated else "-",
        )
    return last


def _revised(c, revised, indicator, scale=1.0):
    if revised.empty:
        return False
    r = revised[revised["indicator"] == indicator]
    for row in r.itertuples(index=False):
        x, color = _x(row.period), c.color(ORDER.index(row.issuer))
        c.ax.scatter(
            [x], [row.as_reported * scale], s=70, facecolors=c.t["surface"], edgecolors=color, linewidths=1.8, zorder=4
        )
        c.ax.scatter(
            [x],
            [row.revised * scale],
            s=55,
            marker="D",
            color=color,
            edgecolors=c.t["surface"],
            linewidths=1.5,
            zorder=5,
        )
        c.ax.plot(
            [x, x], [row.as_reported * scale, row.revised * scale], color=color, linewidth=1, linestyle=":", zorder=3
        )
    if not r.empty:
        c.ax.scatter([], [], s=70, facecolors="none", edgecolors=c.t["muted"], label="as reported, later restated")
        c.ax.scatter([], [], s=55, marker="D", color=c.t["muted"], label="restated figure")
    return not r.empty


def _finish(c, out: Path, name: str, pct: bool):
    c.yaxis(pct=pct)
    c.legend(loc="upper left")
    return c.save(out, name)


# ── figures ────────────────────────────────────────────────────────────────────


def fig_allowance(ind, revised, issuers, out):
    k, p0, v0, p1, v1 = _rise(ind, "allowance_coverage", issuers)
    title = f"{issuers[k].short}'s credit-loss allowance rose from {_pct(v0)} to {_pct(v1)} of receivables"
    sub = f"Allowance for credit losses / trade receivables before allowance, {p0}–{p1}"
    method = "Method: CMF receivables note (receivables before and after the allowance), quarter-end balances."
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, "allowance_coverage", issuers)
        _axes(c, last)
        c.note((_x(p1), v1), f"{_pct(v1)} in {p1}", xytext=(14, 10))
        yield _finish(c, out, "allowance_coverage", True)


def fig_ageing(ind, revised, issuers, out):
    k, p0, v0, p1, v1 = _rise(ind, "receivables_over_180_share", issuers)
    title = f"{issuers[k].short}'s receivables over 180 days past due rose from {_pct(v0)} to {_pct(v1)}"
    sub = f"Share of gross trade receivables more than 180 days past due, {p0}–{p1}"
    method = (
        "Method: CMF portfolio stratification note. Each issuer buckets its portfolio its own way (CLC reports "
        "three or four buckets and added one in 2023Q1): read trends, not levels."
    )
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, "receivables_over_180_share", issuers)
        _axes(c, last)
        c.note((_x(p1), v1), f"{_pct(v1)} in {p1}", xytext=(-150, -30))
        yield _finish(c, out, "receivables_over_180", True)


def fig_dso(ind, revised, seq, issuers, out):
    hits = seq[(seq["indicator"] == "dso") & seq["first_break"].notna()].sort_values(["first_break", "issuer"])
    groups = hits.groupby("first_break", sort=True)["issuer"].apply(lambda g: ", ".join(issuers[i].short for i in g))
    title = (
        "DSO left its 2017–2021 range: " + "; ".join(f"{who} {when}" for when, who in groups.items())
        if not groups.empty
        else "No issuer's DSO left its 2017–2021 range"
    )
    sub = "Days sales outstanding: trade and other receivables / LTM revenue × 365"
    method = "Method: break = two consecutive quarters above the 2017–2021 maximum, from 2022."
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, "dso", issuers)
        _revised(c, revised, "dso")
        _axes(c, last)
        if not hits.empty:
            r = hits.iloc[0]
            c.note((_x(r.first_break), r.value_at_break), f"first break: {r.first_break}", xytext=(-100, 25))
        yield _finish(c, out, "dso", False)


def fig_payables(ind, revised, issuers, out):
    best = None
    for k in _core(issuers):
        s = _series(ind, "payables_overdue_share", k)
        s = s[s.index >= "2022Q1"]
        if not s.empty and (best is None or s.max() > best[2]):
            best = (k, s.idxmax(), s.max())
    k, p, v = best
    title = f"{issuers[k].short}'s overdue trade payables reached {_pct(v)} of the total in {p}"
    sub = "Share of trade payables past their due date. Highlighted: the highest core-panel reading since 2022"
    method = "Method: CMF payables note (payables not yet due and overdue, summed over supplier types)."
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, "payables_overdue_share", issuers)
        _axes(c, last)
        c.note((_x(p), v), f"{_pct(v)} in {p}", xytext=(-95, -5))
        yield _finish(c, out, "payables_overdue", True)


def fig_current_ratio(ind, revised, issuers, out):
    k, p0, v0, p1, v1 = _rise(ind, "current_ratio", issuers, sign=-1)
    title = f"{issuers[k].short}'s current ratio fell from {v0:.2f}x to {v1:.2f}x ({p0}–{p1})"
    sub = "Current assets / current liabilities"
    method = "Method: quarter-end balances; restated figures shown separately, not substituted."
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, "current_ratio", issuers)
        _revised(c, revised, "current_ratio")
        _axes(c, last)
        c.note((_x(p1), v1), f"{v1:.2f}x in {p1}", xytext=(14, -28))
        c.ax.yaxis.set_major_formatter(lambda v, _: f"{v:.1f}x")
        lo, hi = c.ax.get_ylim()
        c.ax.set_ylim(lo, hi + 0.5 * (hi - lo))  # room for the legend above the lines above the lines
        handles, labels = c.ax.get_legend_handles_labels()
        c.ax.legend(handles, labels, ncol=2, frameon=False, loc="upper left", fontsize=8.5, labelcolor=c.t["text"])
        yield c.save(out, "current_ratio")


def fig_cost_of_risk(ind, revised, issuers, out):
    core = _core(issuers)
    peaks = {k: _series(ind, "cost_of_risk", k) for k in core}
    peaks = {k: s[s.index >= RULING] for k, s in peaks.items() if not s.empty}
    lead = max(peaks, key=lambda k: peaks[k].max())
    p, v = peaks[lead].idxmax(), peaks[lead].max()
    others = [k for k in peaks if k != lead]
    lo = min(peaks[k].min() for k in others)
    hi = max(peaks[k].max() for k in others)
    first = min(peaks[k].index.min() for k in others)
    last_o = max(peaks[k].index.max() for k in others)
    names = " and ".join(issuers[k].short for k in others)
    title = f"{issuers[lead].short}'s cost of risk peaked at {_pct(v)}; {names}: {_pct(lo)}–{_pct(hi)}"
    sub = f"Credit losses / revenue, last twelve months. Peak and ranges over {first}–{last_o}"
    method = (
        "Method: Indisa's IFRS 9 line where filed; otherwise change in the allowance over four quarters plus "
        "write-offs (RedSalud only), a lower bound floored at zero."
    )
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, "cost_of_risk", issuers)
        _axes(c, last)
        c.note((_x(p), v), f"{_pct(v)} in {p}", xytext=(14, -4))
        yield _finish(c, out, "cost_of_risk", True)


def fig_ebitda(ind, revised, issuers, out):
    before = "ebitda_before_credit_losses_margin"
    lead, p, b = None, None, None
    for k in _core(issuers):
        s = _series(ind, before, k)
        s = s[s.index >= RULING]
        if not s.empty and (b is None or s.min() < b):
            lead, p, b = k, s.idxmin(), s.min()
    after = _series(ind, "ebitda_margin", lead)
    a_p, a = after[after.index >= RULING].idxmin(), after[after.index >= RULING].min()
    title = f"{issuers[lead].short}'s EBITDA margin bottomed at {_pct(b)} before credit losses, {_pct(a)} after"
    sub = f"EBITDA margin, last twelve months; solid lines before credit losses. Lows: {p} before, {a_p} after"
    method = (
        "Method: EBITDA before credit losses adds back the estimated charge (see cost of risk), a lower bound "
        "for issuers without the IFRS 9 line, so the margin before credit losses is at least as high as shown."
    )
    lows = [_series(ind, before, k).min() for k in ORDER if not _series(ind, before, k).empty]
    floor = np.floor((min(lows) - 0.05) * 20) / 20  # 5-point steps below the lowest before-loss margin
    for c in charts(title, sub, f"{SOURCE} {method}"):
        last = _lines(c, ind, before, issuers, labels={lead: f"{issuers[lead].short}, before credit losses"})
        xs = np.array([_x(q) for q in after.index])
        c.ax.plot(
            xs,
            np.maximum(after.to_numpy(), floor),
            color=c.color(ORDER.index(lead)),
            linewidth=1.4,
            linestyle=":",
            label=f"{issuers[lead].short}, after credit losses",
        )
        _axes(c, last)
        if a < floor:
            c.ax.set_ylim(bottom=floor)
            c.note(
                (_x(a_p), floor),
                f"{issuers[lead].short} after credit losses: {_pct(a)} in {a_p} (off scale)",
                xytext=(-220, 18),
            )
        c.note((_x(p), b), f"before: {_pct(b)}", xytext=(25, -5))
        c.yaxis(pct=True)
        c.legend(loc="lower left")
        yield c.save(out, "ebitda_margin")


SHORT = {"Days sales outstanding": "DSO", "Receivables > 180 days": "Receivables over 180 days"}


def fig_sequence(seq, issuers, out):
    keys = [k for k in ORDER if not seq[(seq["issuer"] == k) & seq["first_break"].notna()].empty]
    first = {}
    for k in keys:
        s = seq[(seq["issuer"] == k) & seq["first_break"].notna()]
        p = s["first_break"].min()
        first[k] = (p, sorted(s[s["first_break"] == p]["label"]))
    common = set.intersection(*(set(v[1]) for v in first.values())) if first else set()
    if len(common) == 1:
        label = SHORT.get(next(iter(common)), next(iter(common)))
        joint = [issuers[k].short for k, v in first.items() if len(v[1]) > 1]
        title = f"{label} broke first at every issuer that broke out" + (
            f", together with others at {', '.join(joint)}" if joint else ""
        )
    else:
        title = "First to break: " + "; ".join(
            f"{issuers[k].short}, {' / '.join(v[1])} ({v[0]})" for k, v in first.items()
        )
    sub = "First quarter from 2022 in which each indicator left its 2017–2021 range, in the adverse direction"
    method = (
        "Method: range = 2017–2021 min to max; a break is two quarters in a row outside it. "
        "◆ short baseline. RedSalud broke none; Bupa stopped filing in 2022Q1."
    )
    rows, ticks = [], []
    y = 0
    for k in keys:
        rows.append((k, None, y))
        y += 1
        for ind_key, (lab, _) in SEQ_INDICATORS.items():
            rows.append((k, ind_key, y))
            ticks.append((y, lab))
            y += 1
        y += 0.6
    periods = sorted(seq["first_break"].dropna())
    for c in charts(title, sub, f"{SOURCE} {method}", size=(9, 2.2 + 0.32 * y)):
        height = 2.2 + 0.32 * y
        c.fig.subplots_adjust(left=0.3, right=0.95, top=1 - 0.9 / height, bottom=0.85 / height)
        title_text, sub_text = c.fig.texts[0], c.fig.texts[1]  # figstyle places them for a 4.8in figure
        title_text.set_y(1 - 0.22 / height)
        sub_text.set_y(1 - 0.56 / height)
        c.ax.grid(axis="y", visible=False)
        c.ax.grid(axis="x", color=c.t["grid"], linewidth=0.8)
        x0 = _x("2022Q1")
        for k, ind_key, yy in rows:
            if ind_key is None:
                c.ax.text(
                    -0.42,
                    yy,
                    issuers[k].short + (" (integrated)" if issuers[k].type == "integrated" else ""),
                    transform=c.ax.get_yaxis_transform(),
                    color=c.t["text"],
                    fontsize=9.5,
                    weight="bold",
                    va="center",
                )
                continue
            r = seq[(seq["issuer"] == k) & (seq["indicator"] == ind_key)]
            if r.empty:
                continue
            r = r.iloc[0]
            if pd.notna(r["first_break"]):
                short = "short" in str(r["note"])
                c.ax.scatter(
                    [_x(r["first_break"])],
                    [yy],
                    s=55,
                    color=c.color(ORDER.index(k)),
                    marker="D" if short else "o",
                    edgecolors=c.t["surface"],
                    linewidths=1.2,
                    zorder=3,
                )
                c.ax.text(
                    _x(r["first_break"]) + 0.4, yy, r["first_break"], fontsize=7.5, color=c.t["text"], va="center"
                )
            else:
                c.ax.text(
                    x0 - 0.6,
                    yy,
                    "no break" if "no break" in str(r["note"]) else "no data",
                    fontsize=7.5,
                    color=c.t["muted"],
                    va="center",
                    style="italic",
                )
        c.ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks], fontsize=8)
        c.ax.set_ylim(y - 0.4, -0.6)
        last = max(periods) if periods else "2022Q4"
        years = range(2022, int(last[:4]) + 2)
        c.ax.set_xticks([yr * 4 for yr in years], [str(yr) for yr in years])
        c.ax.set_xlim(x0 - 1, _x(last) + 3)
        for period, lab, yf in EVENTS:
            c.event(_x(period), lab, y_frac=0.995 if yf > 0.95 else 0.975)
        yield c.save(out, "sequence")


FIGURES = [
    "allowance_coverage",
    "receivables_over_180",
    "dso",
    "payables_overdue",
    "current_ratio",
    "cost_of_risk",
    "ebitda_margin",
    "sequence",
]


def render(data: Path, out: Path) -> list[Path]:
    ind = pd.read_csv(data / "indicators.csv")
    statements = pd.read_csv(data / "statements.csv")
    seq = pd.read_csv(data / "sequence.csv")
    revised = revised_indicators(statements) if "revised" in statements else pd.DataFrame()
    issuers = load_issuers()
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for gen in (
        fig_allowance(ind, revised, issuers, out),
        fig_ageing(ind, revised, issuers, out),
        fig_dso(ind, revised, seq, issuers, out),
        fig_payables(ind, revised, issuers, out),
        fig_current_ratio(ind, revised, issuers, out),
        fig_cost_of_risk(ind, revised, issuers, out),
        fig_ebitda(ind, revised, issuers, out),
        fig_sequence(seq, issuers, out),
    ):
        paths.extend(gen)
    return paths


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=Path("data/processed"))
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = p.parse_args(argv)
    for path in render(args.data, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
