"""Template-based Q&A generation from structured CMM data.

Every answer is traceable to exact source data values -- no LLM generation.
Each QAPair stores ``source_data`` for full provenance.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from cmm_fine_tune.constants import COMMODITY_CONFIG, COUNTRY_CODES, FLOW_CODES
from cmm_fine_tune.data.csv_loader import _WITHHELD
from cmm_fine_tune.models import (
    QAPair,
    SalientRecord,
    TradeFlowRecord,
    WorldProductionRecord,
)

logger = logging.getLogger(__name__)


def _fmt_usd(value: float) -> str:
    """Format a USD value with appropriate scale."""
    if abs(value) >= 1e9:
        return f"${value / 1e9:,.2f} billion"
    if abs(value) >= 1e6:
        return f"${value / 1e6:,.2f} million"
    if abs(value) >= 1e3:
        return f"${value / 1e3:,.1f} thousand"
    return f"${value:,.2f}"


def _fmt_weight(kg: float) -> str:
    """Format weight in kg to the most readable unit."""
    if abs(kg) >= 1e6:
        return f"{kg / 1e6:,.1f} thousand metric tons"
    if abs(kg) >= 1e3:
        return f"{kg / 1e3:,.1f} metric tons"
    return f"{kg:,.1f} kg"


def _fmt_num(value: float, unit: str = "") -> str:
    """Format a generic number with optional unit."""
    s = f"{value:,.0f}" if abs(value) >= 1000000.0 or abs(value) >= 100 else f"{value:,.2f}"
    return f"{s} {unit}".strip() if unit else s


def _country_name(code: str) -> str:
    return COUNTRY_CODES.get(str(code), f"country {code}")


def _flow_name(code: str) -> str:
    return FLOW_CODES.get(code, code)


def _commodity_name(key: str) -> str:
    cfg = COMMODITY_CONFIG.get(key, {})
    return cfg.get("display_name", key.replace("_", " ").title())


# ===========================================================================
# Trade flow templates
# ===========================================================================


def _generate_trade_qa(commodity_key: str, records: list[TradeFlowRecord]) -> list[QAPair]:
    """Generate Q&A pairs from trade flow records."""
    pairs: list[QAPair] = []
    name = _commodity_name(commodity_key)

    # Group records for multi-record templates
    by_reporter_flow_year: dict[tuple, list[TradeFlowRecord]] = defaultdict(list)
    for r in records:
        by_reporter_flow_year[(r.reporter_code, r.flow_code, r.year)].append(r)

    for r in records:
        if r.primary_value is None:
            continue

        reporter = _country_name(r.reporter_code)
        partner = _country_name(r.partner_code)
        flow = _flow_name(r.flow_code)
        val = _fmt_usd(r.primary_value)
        src = {
            "commodity": commodity_key,
            "reporter": reporter,
            "partner": partner,
            "flow": r.flow_code,
            "year": r.year,
            "value_usd": r.primary_value,
        }

        # T1: Direct value lookup (L1)
        if r.partner_code == "0":  # world total
            q = f"What was the total value of {name} {flow.lower()} by {reporter} in {r.year}?"
            a = (
                f"In {r.year}, {reporter}'s total {name} {flow.lower()} "
                f"were valued at {val} (USD), based on UN Comtrade data "
                f"for HS code {r.hs_code}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="trade_total_value",
                    source_data=src,
                )
            )

        # T2: Bilateral trade (L1)
        if r.partner_code != "0":
            q = (
                f"How much did {reporter} {flow.lower()[:-1] if flow.endswith('s') else flow.lower()} "
                f"in {name} from {partner} in {r.year}?"
            )
            a = (
                f"{reporter} {flow.lower()[:-1] + 'ed' if flow == 'Exports' else 'imported'} "
                f"{val} worth of {name} (HS {r.hs_code}) "
                f"{'to' if r.flow_code == 'X' else 'from'} {partner} in {r.year}, "
                f"according to UN Comtrade data."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="trade_bilateral",
                    source_data=src,
                )
            )

        # T3: Weight/quantity (L1) -- only if weight available
        if r.net_weight and r.partner_code == "0":
            wt = _fmt_weight(r.net_weight)
            q = f"What was the quantity of {name} {flow.lower()} by {reporter} in {r.year}?"
            a = (
                f"{reporter} {flow.lower()[:-1] + 'ed' if flow == 'Exports' else 'imported'} "
                f"{wt} of {name} in {r.year} (HS {r.hs_code}), "
                f"with a trade value of {val}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="trade_quantity",
                    source_data={**src, "net_weight_kg": r.net_weight},
                )
            )

    # T4: Year-over-year change (L2)
    # Group by (reporter, partner, flow) to find consecutive years
    by_series: dict[tuple, list[TradeFlowRecord]] = defaultdict(list)
    for r in records:
        if r.primary_value is not None:
            by_series[(r.reporter_code, r.partner_code, r.flow_code)].append(r)

    for key, series in by_series.items():
        sorted_s = sorted(series, key=lambda x: x.year)
        for i in range(1, len(sorted_s)):
            prev, curr = sorted_s[i - 1], sorted_s[i]
            if curr.year != prev.year + 1:
                continue
            if prev.primary_value is None or curr.primary_value is None:
                continue
            if prev.primary_value == 0:
                continue

            reporter = _country_name(curr.reporter_code)
            partner = _country_name(curr.partner_code)
            flow = _flow_name(curr.flow_code)
            change = curr.primary_value - prev.primary_value
            pct = (change / prev.primary_value) * 100
            direction = "increased" if change > 0 else "decreased"

            q = (
                f"How did {reporter}'s {name} {flow.lower()} "
                f"{'to' if curr.flow_code == 'X' else 'from'} "
                f"{partner} change between {prev.year} and {curr.year}?"
            )
            a = (
                f"{reporter}'s {name} {flow.lower()} "
                f"{'to' if curr.flow_code == 'X' else 'from'} {partner} "
                f"{direction} from {_fmt_usd(prev.primary_value)} in {prev.year} "
                f"to {_fmt_usd(curr.primary_value)} in {curr.year}, "
                f"a change of {pct:+.1f}% ({_fmt_usd(abs(change))})."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L2",
                    template_id="trade_yoy_change",
                    source_data={
                        "commodity": commodity_key,
                        "reporter": reporter,
                        "partner": partner,
                        "flow": curr.flow_code,
                        "year_prev": prev.year,
                        "year_curr": curr.year,
                        "value_prev": prev.primary_value,
                        "value_curr": curr.primary_value,
                        "pct_change": pct,
                    },
                )
            )

    # T5: Trade balance (L2) -- for reporters with both M and X to world
    by_reporter_year: dict[tuple, dict] = defaultdict(dict)
    for r in records:
        if r.primary_value is not None and r.partner_code == "0":
            by_reporter_year[(r.reporter_code, r.year)][r.flow_code] = r.primary_value

    for (rcode, year), flows in by_reporter_year.items():
        if "M" in flows and "X" in flows:
            reporter = _country_name(rcode)
            imports_v = flows["M"]
            exports_v = flows["X"]
            balance = exports_v - imports_v
            status = "surplus" if balance > 0 else "deficit"

            q = f"What was {reporter}'s trade balance in {name} in {year}?"
            a = (
                f"In {year}, {reporter} had a trade {status} in {name} of "
                f"{_fmt_usd(abs(balance))}. Imports totaled {_fmt_usd(imports_v)} "
                f"while exports were {_fmt_usd(exports_v)}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L2",
                    template_id="trade_balance",
                    source_data={
                        "commodity": commodity_key,
                        "reporter": reporter,
                        "year": year,
                        "imports_usd": imports_v,
                        "exports_usd": exports_v,
                        "balance_usd": balance,
                    },
                )
            )

    # T6: Import concentration (L3)
    # For each (reporter, year, flow=M), compute partner shares
    for (rcode, flow, year), recs in by_reporter_flow_year.items():
        if flow != "M":
            continue
        world_recs = [r for r in recs if r.partner_code == "0" and r.primary_value]
        partner_recs = [r for r in recs if r.partner_code != "0" and r.primary_value]
        if not world_recs or not partner_recs:
            continue

        total_val = world_recs[0].primary_value
        if not total_val or total_val <= 0:
            continue

        reporter = _country_name(rcode)
        top_partners = sorted(partner_recs, key=lambda x: x.primary_value or 0, reverse=True)[:3]
        if len(top_partners) < 2:
            continue

        top_str = ", ".join(
            f"{_country_name(p.partner_code)} ({_fmt_usd(p.primary_value)}, "
            f"{(p.primary_value / total_val) * 100:.1f}%)"
            for p in top_partners
        )

        q = f"Which countries were the largest sources of {name} imports for {reporter} in {year}?"
        a = (
            f"In {year}, {reporter}'s top {name} import sources were: {top_str}. "
            f"Total imports were valued at {_fmt_usd(total_val)}."
        )
        pairs.append(
            QAPair(
                question=q,
                answer=a,
                commodity=commodity_key,
                complexity_level="L3",
                template_id="trade_import_concentration",
                source_data={
                    "commodity": commodity_key,
                    "reporter": reporter,
                    "year": year,
                    "total_imports_usd": total_val,
                    "top_partners": [
                        {
                            "partner": _country_name(p.partner_code),
                            "value_usd": p.primary_value,
                            "share_pct": (p.primary_value / total_val) * 100,
                        }
                        for p in top_partners
                    ],
                },
            )
        )

    return pairs


# ===========================================================================
# Salient statistics templates
# ===========================================================================


def _generate_salient_qa(commodity_key: str, records: list[SalientRecord]) -> list[QAPair]:
    """Generate Q&A pairs from USGS salient statistics."""
    pairs: list[QAPair] = []
    name = _commodity_name(commodity_key)

    for r in records:
        fields = r.fields
        year = r.year
        source = r.data_source

        # S1: Net import reliance (L1)
        nir = fields.get("NIR_pct")
        if nir is not None:
            nir_str = (
                str(nir)
                if isinstance(nir, str) and not str(nir).replace(".", "").isdigit()
                else f"{nir}%"
            )
            # Handle ">50", ">25" style values stored as raw strings
            if isinstance(nir, str) and nir.startswith(">"):
                nir_str = f"greater than {nir[1:]}%"
            elif isinstance(nir, (int, float)):
                nir_str = f"{int(nir)}%"

            q = f"What was the US net import reliance for {name} in {year}?"
            a = (
                f"According to the USGS Mineral Commodity Summaries ({source}), "
                f"US net import reliance for {name} in {year} was {nir_str} of "
                f"apparent consumption."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="salient_nir",
                    source_data={
                        "commodity": commodity_key,
                        "year": year,
                        "nir_pct": str(nir),
                        "source": source,
                    },
                )
            )

        # S2: US production (L1) -- look for any production-like field
        prod_fields = {k: v for k, v in fields.items() if "prod" in k.lower() and "USprod" in k}
        for pfield, pval in prod_fields.items():
            if pval is None:
                continue
            if isinstance(pval, str) and pval in ("W", "w", "0"):
                if pval in ("W", "w"):
                    continue  # skip withheld
                pval_f = 0.0
            else:
                pval_f = float(pval) if isinstance(pval, (int, float)) else None
                if pval_f is None:
                    continue

            # Extract unit hint from column name
            unit = "metric tons"
            if "_kg" in pfield:
                unit = "kilograms"
            elif "_kt" in pfield:
                unit = "thousand metric tons"

            # Clean field label
            label = (
                pfield.replace("USprod_", "")
                .replace("_t", "")
                .replace("_kg", "")
                .replace("_kt", "")
                .replace("_num", "")
            )
            label = label.replace("_", " ").strip()

            q = f"What was US {label} production of {name} in {year}?"
            a = (
                f"US {label} production of {name} in {year} was "
                f"{_fmt_num(pval_f, unit)}, according to {source}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="salient_us_production",
                    source_data={
                        "commodity": commodity_key,
                        "year": year,
                        "field": pfield,
                        "value": pval_f,
                        "unit": unit,
                        "source": source,
                    },
                )
            )

        # S3: Price (L1) -- look for price fields
        price_fields = {k: v for k, v in fields.items() if "price" in k.lower()}
        for pfield, pval in price_fields.items():
            if pval is None or (isinstance(pval, str) and pval in _WITHHELD):
                continue
            pval_f = float(pval) if isinstance(pval, (int, float)) else None
            if pval_f is None:
                continue

            # Extract price unit
            unit = "dollars"
            if "_dlb" in pfield:
                unit = "dollars per pound"
            elif "_dkg" in pfield:
                unit = "dollars per kilogram"
            elif "_dt" in pfield:
                unit = "dollars per metric ton"
            elif "_ctslb" in pfield:
                unit = "cents per pound"

            label = pfield.replace("Price_", "").split("_")[0]

            q = f"What was the {label} price of {name} in {year}?"
            a = (
                f"The {label} price of {name} in {year} was "
                f"{_fmt_num(pval_f, unit)}, as reported in {source}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="salient_price",
                    source_data={
                        "commodity": commodity_key,
                        "year": year,
                        "field": pfield,
                        "value": pval_f,
                        "unit": unit,
                        "source": source,
                    },
                )
            )

        # S4: Imports/Exports (L1)
        for trade_field, trade_val in fields.items():
            if not (trade_field.startswith("Imports_") or trade_field.startswith("Exports_")):
                continue
            if trade_val is None or (isinstance(trade_val, str) and trade_val in _WITHHELD):
                continue
            tv_f = float(trade_val) if isinstance(trade_val, (int, float)) else None
            if tv_f is None:
                continue

            direction = "imports" if trade_field.startswith("Imports") else "exports"
            unit = "metric tons"
            if "_kg" in trade_field:
                unit = "kilograms"
            elif "_kt" in trade_field:
                unit = "thousand metric tons"
            detail = trade_field.split("_", 1)[1] if "_" in trade_field else ""
            detail = detail.replace("_t", "").replace("_kg", "").replace("_kt", "")
            detail = detail.replace("_", " ").strip()
            detail_str = f" ({detail})" if detail else ""

            q = f"What were the US {name} {direction}{detail_str} in {year}?"
            a = (
                f"US {name} {direction}{detail_str} in {year} totaled "
                f"{_fmt_num(tv_f, unit)}, according to {source}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="salient_trade_volume",
                    source_data={
                        "commodity": commodity_key,
                        "year": year,
                        "field": trade_field,
                        "value": tv_f,
                        "unit": unit,
                        "source": source,
                    },
                )
            )

    # S5: Price trend (L2) -- consecutive years
    sorted_records = sorted(records, key=lambda x: x.year)
    for i in range(1, len(sorted_records)):
        prev, curr = sorted_records[i - 1], sorted_records[i]
        if curr.year != prev.year + 1:
            continue

        prev_prices = {
            k: v
            for k, v in prev.fields.items()
            if "price" in k.lower() and isinstance(v, (int, float))
        }
        curr_prices = {
            k: v
            for k, v in curr.fields.items()
            if "price" in k.lower() and isinstance(v, (int, float))
        }

        for pfield in set(prev_prices) & set(curr_prices):
            pv = float(prev_prices[pfield])
            cv = float(curr_prices[pfield])
            if pv == 0:
                continue
            change_pct = ((cv - pv) / pv) * 100
            direction = "rose" if change_pct > 0 else "fell"
            label = pfield.replace("Price_", "").split("_")[0]

            q = f"How did the {label} price of {name} change from {prev.year} to {curr.year}?"
            a = (
                f"The {label} price of {name} {direction} from "
                f"{_fmt_num(pv)} in {prev.year} to {_fmt_num(cv)} in {curr.year}, "
                f"a change of {change_pct:+.1f}%."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L2",
                    template_id="salient_price_trend",
                    source_data={
                        "commodity": commodity_key,
                        "year_prev": prev.year,
                        "year_curr": curr.year,
                        "field": pfield,
                        "value_prev": pv,
                        "value_curr": cv,
                        "pct_change": change_pct,
                    },
                )
            )

    # S6: Production-consumption gap (L2)
    for r in records:
        # Find any consumption field and any production field
        consump_fields = {
            k: v
            for k, v in r.fields.items()
            if "consump" in k.lower() and isinstance(v, (int, float))
        }
        prod_fields = {
            k: v for k, v in r.fields.items() if "USprod" in k and isinstance(v, (int, float))
        }
        if not consump_fields or not prod_fields:
            continue

        total_prod = sum(float(v) for v in prod_fields.values())
        # Use first consumption field
        consump_key = sorted(consump_fields.keys())[0]
        consump_val = float(consump_fields[consump_key])
        if consump_val == 0:
            continue

        gap = consump_val - total_prod
        pct = (gap / consump_val) * 100

        q = f"What was the gap between US production and consumption of {name} in {r.year}?"
        a = (
            f"In {r.year}, US domestic production of {name} totaled "
            f"{_fmt_num(total_prod)} while consumption was {_fmt_num(consump_val)}, "
            f"resulting in a gap of {_fmt_num(abs(gap))} "
            f"({abs(pct):.1f}% of consumption) that needed to be met through "
            f"{'imports' if gap > 0 else 'was in surplus'}."
        )
        pairs.append(
            QAPair(
                question=q,
                answer=a,
                commodity=commodity_key,
                complexity_level="L2",
                template_id="salient_prod_consump_gap",
                source_data={
                    "commodity": commodity_key,
                    "year": r.year,
                    "production": total_prod,
                    "consumption": consump_val,
                    "gap": gap,
                    "gap_pct": pct,
                },
            )
        )

    return pairs


# ===========================================================================
# World production templates
# ===========================================================================


def _generate_world_production_qa(
    commodity_key: str, records: list[WorldProductionRecord]
) -> list[QAPair]:
    """Generate Q&A pairs from world mine production data."""
    pairs: list[QAPair] = []
    name = _commodity_name(commodity_key)

    # Group by source (MCS year) for cross-country comparisons
    by_source: dict[str, list[WorldProductionRecord]] = defaultdict(list)
    for r in records:
        by_source[r.source].append(r)

    for r in records:
        # W1: Country production (L1) -- year 1
        if r.production_year1 is not None and r.country.lower() not in ("world total", "world"):
            q = (
                f"How much {name} did {r.country} produce in "
                f"{r.production_year1_label.replace(' (est.)', '')}?"
            )
            a = (
                f"{r.country} produced {_fmt_num(r.production_year1)} metric tons of "
                f"{name} in {r.production_year1_label.replace(' (est.)', '')}, "
                f"according to the USGS ({r.source})."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="world_country_production_y1",
                    source_data={
                        "commodity": commodity_key,
                        "country": r.country,
                        "year": r.production_year1_label,
                        "production": r.production_year1,
                        "source": r.source,
                    },
                )
            )

        # W2: Country production year 2 (L1)
        if r.production_year2 is not None and r.country.lower() not in ("world total", "world"):
            est = " (estimated)" if "est" in r.production_year2_label else ""
            year_clean = r.production_year2_label.replace(" (est.)", "")
            q = f"What was {r.country}'s {name} production in {year_clean}?"
            a = (
                f"{r.country}'s {name} production in {year_clean} was "
                f"{_fmt_num(r.production_year2)} metric tons{est}, "
                f"according to the USGS ({r.source})."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="world_country_production_y2",
                    source_data={
                        "commodity": commodity_key,
                        "country": r.country,
                        "year": r.production_year2_label,
                        "production": r.production_year2,
                        "source": r.source,
                    },
                )
            )

        # W3: Reserves (L1)
        if r.reserves is not None and r.country.lower() not in ("world total", "world"):
            q = f"What are {r.country}'s known {name} reserves?"
            a = (
                f"{r.country} has estimated {name} reserves of "
                f"{_fmt_num(r.reserves)} metric tons, "
                f"as reported by the USGS ({r.source})."
            )
            if r.reserves_notes:
                a += f" Note: {r.reserves_notes}"
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="world_reserves",
                    source_data={
                        "commodity": commodity_key,
                        "country": r.country,
                        "reserves": r.reserves,
                        "source": r.source,
                    },
                )
            )

    # Cross-country templates (per source year)
    for source, src_records in by_source.items():
        valid = [
            r
            for r in src_records
            if r.production_year2 is not None and r.country.lower() not in ("world total", "world")
        ]
        world_recs = [
            r
            for r in src_records
            if r.country.lower() in ("world total", "world") and r.production_year2 is not None
        ]

        if not valid:
            continue

        # W4: World total (L1)
        for wr in world_recs:
            year_clean = wr.production_year2_label.replace(" (est.)", "")
            q = f"What was total world {name} production in {year_clean}?"
            a = (
                f"Total world {name} production in {year_clean} was "
                f"{_fmt_num(wr.production_year2)} metric tons, "
                f"according to the USGS ({source})."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L1",
                    template_id="world_total_production",
                    source_data={
                        "commodity": commodity_key,
                        "year": wr.production_year2_label,
                        "total": wr.production_year2,
                        "source": source,
                    },
                )
            )

        # W5: Country share of world production (L2)
        world_total = world_recs[0].production_year2 if world_recs else None
        if world_total and world_total > 0:
            year_clean = world_recs[0].production_year2_label.replace(" (est.)", "")
            for cr in valid:
                share = (cr.production_year2 / world_total) * 100
                q = (
                    f"What share of global {name} production did {cr.country} "
                    f"account for in {year_clean}?"
                )
                a = (
                    f"{cr.country} accounted for approximately {share:.1f}% of global "
                    f"{name} production in {year_clean}, producing "
                    f"{_fmt_num(cr.production_year2)} metric tons out of a world total "
                    f"of {_fmt_num(world_total)} metric tons ({source})."
                )
                pairs.append(
                    QAPair(
                        question=q,
                        answer=a,
                        commodity=commodity_key,
                        complexity_level="L2",
                        template_id="world_country_share",
                        source_data={
                            "commodity": commodity_key,
                            "country": cr.country,
                            "year": year_clean,
                            "production": cr.production_year2,
                            "world_total": world_total,
                            "share_pct": share,
                            "source": source,
                        },
                    )
                )

        # W6: Top producers (L2)
        sorted_producers = sorted(valid, key=lambda x: x.production_year2 or 0, reverse=True)
        if len(sorted_producers) >= 3:
            top3 = sorted_producers[:3]
            year_clean = top3[0].production_year2_label.replace(" (est.)", "")
            top_str = ", ".join(
                f"{p.country} ({_fmt_num(p.production_year2)} metric tons)" for p in top3
            )
            q = f"Which countries were the top producers of {name} in {year_clean}?"
            a = f"The top three producers of {name} in {year_clean} were: {top_str} ({source})."
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L2",
                    template_id="world_top_producers",
                    source_data={
                        "commodity": commodity_key,
                        "year": year_clean,
                        "top_producers": [
                            {"country": p.country, "production": p.production_year2} for p in top3
                        ],
                        "source": source,
                    },
                )
            )

        # W7: Reserves-to-production ratio (L2)
        for cr in valid:
            if cr.reserves and cr.production_year2 and cr.production_year2 > 0:
                ratio = cr.reserves / cr.production_year2
                year_clean = cr.production_year2_label.replace(" (est.)", "")
                q = f"What is {cr.country}'s reserves-to-production ratio for {name}?"
                a = (
                    f"Based on {source} data, {cr.country}'s {name} reserves-to-production "
                    f"ratio is approximately {ratio:.0f} years (reserves of "
                    f"{_fmt_num(cr.reserves)} metric tons divided by {year_clean} "
                    f"production of {_fmt_num(cr.production_year2)} metric tons)."
                )
                pairs.append(
                    QAPair(
                        question=q,
                        answer=a,
                        commodity=commodity_key,
                        complexity_level="L2",
                        template_id="world_reserves_production_ratio",
                        source_data={
                            "commodity": commodity_key,
                            "country": cr.country,
                            "reserves": cr.reserves,
                            "production": cr.production_year2,
                            "ratio_years": ratio,
                            "source": source,
                        },
                    )
                )

        # W8: Production concentration / HHI (L3)
        if world_total and world_total > 0 and len(valid) >= 3:
            shares = [(r.country, (r.production_year2 / world_total) * 100) for r in valid]
            hhi = sum(s**2 for _, s in shares)
            top3_share = sum(s for _, s in sorted(shares, key=lambda x: -x[1])[:3])
            year_clean = world_recs[0].production_year2_label.replace(" (est.)", "")

            conc = (
                "highly concentrated"
                if hhi > 2500
                else "moderately concentrated"
                if hhi > 1500
                else "diversified"
            )
            q = f"How concentrated is global {name} production in {year_clean}?"
            a = (
                f"Global {name} production in {year_clean} is {conc}. "
                f"The top three producers account for {top3_share:.1f}% of world output "
                f"(HHI index: {hhi:.0f}). {source}."
            )
            pairs.append(
                QAPair(
                    question=q,
                    answer=a,
                    commodity=commodity_key,
                    complexity_level="L3",
                    template_id="world_production_concentration",
                    source_data={
                        "commodity": commodity_key,
                        "year": year_clean,
                        "hhi": hhi,
                        "top3_share_pct": top3_share,
                        "source": source,
                    },
                )
            )

    return pairs


# ===========================================================================
# Public API
# ===========================================================================


def generate_all_qa(
    trade_data: dict[str, list[TradeFlowRecord]],
    salient_data: dict[str, list[SalientRecord]],
    world_data: dict[str, list[WorldProductionRecord]],
) -> list[QAPair]:
    """Generate all Q&A pairs from the loaded data.

    Returns a flat list of QAPair objects across all commodities and templates.
    """
    all_pairs: list[QAPair] = []

    for commodity_key in set(list(trade_data) + list(salient_data) + list(world_data)):
        if commodity_key in trade_data:
            pairs = _generate_trade_qa(commodity_key, trade_data[commodity_key])
            logger.info("Generated %d trade Q&A pairs for %s", len(pairs), commodity_key)
            all_pairs.extend(pairs)

        if commodity_key in salient_data:
            pairs = _generate_salient_qa(commodity_key, salient_data[commodity_key])
            logger.info("Generated %d salient Q&A pairs for %s", len(pairs), commodity_key)
            all_pairs.extend(pairs)

        if commodity_key in world_data:
            pairs = _generate_world_production_qa(commodity_key, world_data[commodity_key])
            logger.info("Generated %d world production Q&A pairs for %s", len(pairs), commodity_key)
            all_pairs.extend(pairs)

    logger.info("Total Q&A pairs generated: %d", len(all_pairs))
    return all_pairs
