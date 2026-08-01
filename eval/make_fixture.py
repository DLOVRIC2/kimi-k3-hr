#!/usr/bin/env python3
"""Generate the long-context fixture with a known answer key.

The long_ctx probe asks the model to list every store that reported a stock
discrepancy in Q3. Hand-writing 32k tokens of filler would make that probe
subjective; generating it means the ground truth is exact, so this is the one
probe that scores itself.

Discrepancies are planted at controlled depths -- early, middle, and late in the
document -- because retrieval failure is usually positional. A model that finds
the first two and misses the last is failing differently from one that finds
none, and a single score would conflate them.

    uv run eval/make_fixture.py --tokens 32000
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

CITIES = [
    "Zagreb", "Split", "Rijeka", "Osijek", "Zadar", "Velika Gorica", "Slavonski Brod",
    "Pula", "Karlovac", "Sisak", "Varaždin", "Šibenik", "Dubrovnik", "Bjelovar",
    "Kaštela", "Samobor", "Vinkovci", "Koprivnica", "Đakovo", "Vukovar",
]
REGIONS = ["Sjever", "Jug", "Istok", "Zapad", "Središnja"]

FILLER = [
    "Footfall tracked within seasonal norms for the period.",
    "Chilled cabinet temperatures logged twice daily, no excursions recorded.",
    "Loyalty scheme enrolment continued its gradual upward trend.",
    "Scheduled maintenance on the refrigeration unit completed without incident.",
    "Staff rota fully covered; no unplanned absence above threshold.",
    "Bakery waste remained inside the tolerance band agreed at the regional review.",
    "Card terminal uptime nominal across the quarter.",
    "Supplier deliveries arrived within the agreed two-hour window.",
    "Shelf-edge label audit completed, no material errors found.",
    "Fire safety inspection passed with no actions raised.",
    "Weekly cash reconciliation balanced on every occasion.",
    "Promotional end-caps rotated on the published schedule.",
    "Deli counter queue times stayed under the four-minute target.",
    "No customer complaints escalated beyond store level.",
    "Energy consumption tracked marginally below the prior-year baseline.",
]

DISCREPANCY = (
    "A stock discrepancy was identified during the Q3 count: {n} units of SKU {sku} "
    "unaccounted for against system records. Escalated to regional loss prevention."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32000, help="approximate target")
    ap.add_argument("--chars-per-token", type=float, default=4.78,
                    help="measured with the K3 tokenizer on this content")
    ap.add_argument("--discrepancies", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260801)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    here = pathlib.Path(__file__).resolve().parent
    out = pathlib.Path(args.out) if args.out else here / "fixtures" / "long_context_32k.txt"
    out.parent.mkdir(parents=True, exist_ok=True)

    target_chars = int(args.tokens * args.chars_per_token)

    # Size the roster from a real sample entry rather than guessing. The previous
    # version measured an always-empty field, so the loop ran to its safety cap
    # and produced a 194k-token fixture for a 32k-token request.
    sample = "\n".join([
        f"## Store ST-100 — {CITIES[0]} (Region: {REGIONS[0]})", "",
        FILLER[0], FILLER[1], FILLER[2], "",
    ])
    entry_chars = len(sample) + 1
    n = max(1, target_chars // entry_chars)

    stores = [
        {
            "id": f"ST-{100 + i:03d}",
            "city": CITIES[i % len(CITIES)],
            "region": REGIONS[i % len(REGIONS)],
            "discrepancy": False,
        }
        for i in range(n)
    ]
    # Spread across the document: two early, two middle, two late (for the default 6).
    third = max(1, args.discrepancies // 3)
    picks: list[int] = []
    for lo, hi, k in ((0, n // 3, third),
                      (n // 3, 2 * n // 3, third),
                      (2 * n // 3, n, args.discrepancies - 2 * third)):
        picks += rng.sample(range(lo, max(hi, lo + 1)), min(k, max(hi - lo, 1)))
    picks = sorted(set(picks))[: args.discrepancies]

    for idx in picks:
        stores[idx]["discrepancy"] = True

    lines = [
        "REGIONAL RETAIL OPERATIONS — QUARTERLY STORE REPORT",
        "Period: Q3 2026 | Synthetic document, generated for evaluation",
        "",
        "The following report consolidates per-store operational summaries for the",
        "third quarter. Each entry records footfall, compliance checks, and any",
        "exceptions raised during the quarterly stock count.",
        "",
    ]
    for s in stores:
        lines += [
            f"## Store {s['id']} — {s['city']} (Region: {s['region']})",
            "",
            rng.choice(FILLER),
            rng.choice(FILLER),
        ]
        if s["discrepancy"]:
            lines.append(DISCREPANCY.format(n=rng.randint(11, 240),
                                            sku=rng.randint(80000, 99999)))
        lines += [rng.choice(FILLER), ""]

    text = "\n".join(lines)
    out.write_text(text)

    key = [{"store": stores[i]["id"], "city": stores[i]["city"],
            "region": stores[i]["region"],
            "position_pct": round(i / n * 100)} for i in picks]
    key_path = out.with_suffix(".key.json")
    key_path.write_text(json.dumps({"stores_total": n, "discrepancies": key}, indent=2,
                                   ensure_ascii=False))

    print(f"wrote {out}")
    print(f"  {len(text):,} chars over {n} stores (~{len(text)/args.chars_per_token:,.0f} tokens)")
    print(f"wrote {key_path}")
    print(f"\nanswer key — {len(key)} planted discrepancies:")
    for k in key:
        print(f"  {k['store']:<8} {k['city']:<16} {k['region']:<12} {k['position_pct']:>3}% through")


if __name__ == "__main__":
    main()
