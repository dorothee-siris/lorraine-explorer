# lib/taxonomy.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ------------------------- canonical order & palette -------------------------

_DOMAIN_ORDER_CANON = [
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Social Sciences",
    "Other",
]

_DOMAIN_COLORS = {
    "Health Sciences": "#F85C32",
    "Life Sciences":   "#0CA750",
    "Physical Sciences": "#8190FF",
    "Social Sciences": "#FFCB3A",
    "Other": "#7f7f7f",
}

# Default path: repo_root/data/all_topics.parquet
_DEFAULT_TOPICS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "all_topics.parquet"
)

# ----------------------------- data loader -----------------------------


@lru_cache(maxsize=1)
def _load_topics(topics_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load all_topics.parquet and normalize key columns.

    Expected columns (accept various casings):
      - domain_id, domain_name
      - field_id, field_name
      - subfield_id, subfield_name
      - topic_id (kept as string like 'T13054'), topic_name (optional)
    """
    path = Path(topics_path) if topics_path else _DEFAULT_TOPICS_PATH
    df = pd.read_parquet(path)

    # normalize expected columns (accept different casings)
    rename_map = {
        "domain_id": "domain_id",
        "Domain ID": "domain_id",
        "domain_name": "domain_name",
        "Domain name": "domain_name",
        "field_id": "field_id",
        "Field ID": "field_id",
        "field_name": "field_name",
        "Field name": "field_name",
        "subfield_id": "subfield_id",
        "Subfield ID": "subfield_id",
        "subfield_name": "subfield_name",
        "Subfield name": "subfield_name",
        "topic_id": "topic_id",
        "Topic ID": "topic_id",
        "topic_name": "topic_name",
        "Topic name": "topic_name",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = [
        "domain_id",
        "domain_name",
        "field_id",
        "field_name",
        "subfield_id",
        "subfield_name",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"all_topics.parquet is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # coerce ids to int where possible (EXCEPT topic_id which may be alphanumeric like 'T13054')
    for c in ("domain_id", "field_id", "subfield_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # keep topic_id as string, if present
    if "topic_id" in df.columns:
        df["topic_id"] = df["topic_id"].astype(str).str.strip()
        # Optional: treat blanks and 'nan'/'None' as missing
        df.loc[df["topic_id"].str.lower().isin({"", "nan", "none"}), "topic_id"] = pd.NA

    # strip whitespace on names
    for c in ("domain_name", "field_name", "subfield_name", "topic_name"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df


# ------------------------- lookup builder --------------------------


@lru_cache(maxsize=1)
def build_taxonomy_lookups(topics_path: Optional[str | Path] = None) -> Dict:
    """
    Build hierarchical mappings and canonical ordering from all_topics.parquet.

    Returns a dict with:
      - domain_order: [domain_name,...] (canonical order filtered to present domains)
      - fields_by_domain: {domain_name: [field_name,...]} (alphabetical within domain)
      - subfields_by_field: {field_name: [subfield_name,...]} (alphabetical)
      - canonical_fields: [field_name,...]  (domain-grouped alphabetical)
      - canonical_subfields: [subfield_name,...] (grouped by field)
      - id2name: {str(id): name} for domain/field/subfield/topic
      - name2id: {name: str(id)} inverse mapping
      - field_id_to_domain: {field_id_str: domain_name}
      - subfield_id_to_domain: {subfield_id_str: domain_name}
      - topic_id_to_domain: {topic_id_str: domain_name}
      - field_id_to_name: {field_id_str: field_name}
      - subfield_id_to_name: {subfield_id_str: subfield_name}
      - topic_id_to_name: {topic_id_str: topic_name}
    """
    t = _load_topics(topics_path)

    # Domains present (ordered by domain_id)
    present = (
        t[["domain_id", "domain_name"]]
        .drop_duplicates()
        .sort_values("domain_id", na_position="last")
    )
    present_names = present["domain_name"].tolist()

    # Canonical order filtered to what's present + any extras
    domain_order = [d for d in _DOMAIN_ORDER_CANON if d in present_names]
    extras = [d for d in present_names if d not in domain_order]
    domain_order += extras

    # Fields per domain (alphabetical)
    fields_by_domain: Dict[str, List[str]] = {}
    for d in domain_order:
        fields = (
            t.loc[t["domain_name"] == d, "field_name"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        fields_by_domain[d] = fields

    # Subfields per field (alphabetical)
    subfields_by_field: Dict[str, List[str]] = {}
    for f in t["field_name"].drop_duplicates().tolist():
        subs = (
            t.loc[t["field_name"] == f, "subfield_name"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        subfields_by_field[f] = subs

    # Canonical flat orders
    canonical_fields = [f for d in domain_order for f in fields_by_domain[d]]
    canonical_subfields: List[str] = []
    for f in canonical_fields:
        canonical_subfields.extend(subfields_by_field.get(f, []))

    # ---------- Robust ID→name and ID→domain maps ----------
    t_ids = t[
        [
            "domain_id",
            "domain_name",
            "field_id",
            "field_name",
            "subfield_id",
            "subfield_name",
            "topic_id",
            "topic_name",
        ]
    ].drop_duplicates()

    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}

    field_id_to_domain: Dict[str, str] = {}
    subfield_id_to_domain: Dict[str, str] = {}
    topic_id_to_domain: Dict[str, str] = {}

    field_id_to_name: Dict[str, str] = {}
    subfield_id_to_name: Dict[str, str] = {}
    topic_id_to_name_map: Dict[str, str] = {}

    for _, r in t_ids.iterrows():
        # domain
        if pd.notna(r["domain_id"]) and pd.notna(r["domain_name"]):
            did = str(int(r["domain_id"]))
            dnm = str(r["domain_name"])
            id2name[did] = dnm
            name2id[dnm] = did

        # field
        if pd.notna(r["field_id"]) and pd.notna(r["field_name"]):
            fid = str(int(r["field_id"]))
            fnm = str(r["field_name"])
            id2name[fid] = fnm
            name2id[fnm] = fid
            field_id_to_name[fid] = fnm
            if pd.notna(r["domain_name"]):
                field_id_to_domain[fid] = str(r["domain_name"])

        # subfield
        if pd.notna(r["subfield_id"]) and pd.notna(r["subfield_name"]):
            sid = str(int(r["subfield_id"]))
            snm = str(r["subfield_name"])
            id2name[sid] = snm
            name2id[snm] = sid
            subfield_id_to_name[sid] = snm
            if pd.notna(r["domain_name"]):
                subfield_id_to_domain[sid] = str(r["domain_name"])

        # topic
        if pd.notna(r["topic_id"]) and pd.notna(r["topic_name"]):
            tid = str(r["topic_id"]).strip()  # keep alphanumeric
            tnm = str(r["topic_name"])
            id2name[tid] = tnm
            name2id[tnm] = tid
            topic_id_to_name_map[tid] = tnm
            if pd.notna(r["domain_name"]):
                topic_id_to_domain[tid] = str(r["domain_name"])

    return {
        "domain_order": domain_order,
        "fields_by_domain": fields_by_domain,
        "subfields_by_field": subfields_by_field,
        "canonical_fields": canonical_fields,
        "canonical_subfields": canonical_subfields,
        "id2name": id2name,
        "name2id": name2id,
        "field_id_to_domain": field_id_to_domain,
        "subfield_id_to_domain": subfield_id_to_domain,
        "topic_id_to_domain": topic_id_to_domain,
        "field_id_to_name": field_id_to_name,
        "subfield_id_to_name": subfield_id_to_name,
        "topic_id_to_name": topic_id_to_name_map,
    }


# ---------------------------- color helpers -------------------------------


@lru_cache(maxsize=None)
def get_domain_color(name_or_id: str) -> str:
    """
    Map a domain name or ID to its hex color.
    Unknown domains -> 'Other'.
    """
    look = build_taxonomy_lookups()
    name = str(name_or_id)
    if name.isdigit():
        # convert id -> name
        name = look["id2name"].get(name, name)
    return _DOMAIN_COLORS.get(name, _DOMAIN_COLORS["Other"])


@lru_cache(maxsize=None)
def get_field_color(field_name_or_id: str) -> str:
    """
    Field inherits its domain color (via ID-aware mapping where possible).
    """
    look = build_taxonomy_lookups()
    tok = str(field_name_or_id).strip()
    # If it's an ID, resolve directly to domain
    if tok.isdigit():
        dom = look["field_id_to_domain"].get(tok)
        if dom:
            return get_domain_color(dom)
        # fall back: id -> name
        field_name = look["field_id_to_name"].get(tok, tok)
    else:
        field_name = tok

    # name path (deterministic within fields_by_domain)
    for d, fields in look["fields_by_domain"].items():
        if field_name in fields:
            return get_domain_color(d)
    return _DOMAIN_COLORS["Other"]


@lru_cache(maxsize=None)
def get_subfield_color(subfield_name_or_id: str) -> str:
    """
    Subfield inherits its parent domain color.
    Prefer ID-based mapping to avoid homonym collisions.
    """
    look = build_taxonomy_lookups()
    tok = str(subfield_name_or_id).strip()
    if tok.isdigit():
        dom = look["subfield_id_to_domain"].get(tok)
        if dom:
            return get_domain_color(dom)
        # fallback via field tree if needed
    else:
        # name path (can be ambiguous; try to find a unique field that contains it)
        for field_name, subs in look["subfields_by_field"].items():
            if tok in subs:
                dom = get_domain_for_field(field_name)
                return get_domain_color(dom)
    return _DOMAIN_COLORS["Other"]


@lru_cache(maxsize=None)
def get_topic_color(topic_name_or_id: str) -> str:
    """
    Color for a topic by its domain.
    Prefer topic_id→domain mapping; fallback via direct table scan.
    """
    look = build_taxonomy_lookups()
    tok = str(topic_name_or_id).strip()

    # If it's exactly an ID we know, use its domain directly
    dom = look["topic_id_to_domain"].get(tok)
    if dom:
        return get_domain_color(dom)

    # If it's a name, resolve to ID then to domain
    tid = look["name2id"].get(tok)
    if tid:
        dom = look["topic_id_to_domain"].get(tid)
        if dom:
            return get_domain_color(dom)

    # Last resort: scan the table (handles cases where name wasn't registered)
    tdf = _load_topics()
    m = tdf[(tdf["topic_id"].astype(str) == tok)]
    if m.empty and "topic_name" in tdf.columns:
        m = tdf[tdf["topic_name"].astype(str) == tok]
    if not m.empty and "domain_name" in m.columns and pd.notna(m["domain_name"].iloc[0]):
        return get_domain_color(str(m["domain_name"].iloc[0]))

    return _DOMAIN_COLORS["Other"]


# --------------------------- domain helpers --------------------------


@lru_cache(maxsize=None)
def get_domain_for_field(field_name_or_id: str) -> str:
    """
    Return the parent domain name for a given field (name or numeric id).
    If unknown, returns 'Other'.
    """
    look = build_taxonomy_lookups()
    tok = str(field_name_or_id).strip()
    if tok.isdigit():
        dom = look["field_id_to_domain"].get(tok)
        if dom:
            return dom
        tok = look["field_id_to_name"].get(tok, tok)

    for dom, fields in look["fields_by_domain"].items():
        if tok in fields:
            return dom
    return "Other"


@lru_cache(maxsize=None)
def get_domain_for_subfield(subfield_name_or_id: str) -> str:
    """
    Return the parent domain name for a given subfield (name or numeric id).
    Prefer ID-based mapping.
    """
    look = build_taxonomy_lookups()
    tok = str(subfield_name_or_id).strip()
    if tok.isdigit():
        dom = look["subfield_id_to_domain"].get(tok)
        return dom if dom else "Other"
    # name fallback
    for field_name, subs in look["subfields_by_field"].items():
        if tok in subs:
            return get_domain_for_field(field_name)
    return "Other"


# --------------------------- conveniences --------------------------


@lru_cache(maxsize=None)
def field_id_to_name(field_id_or_name: str) -> str:
    """
    Normalize a field token (either id or name) to a field name string.
    If not resolvable, returns the original token.
    """
    tok = str(field_id_or_name).strip()
    look = build_taxonomy_lookups()
    if tok.isdigit():
        return look["field_id_to_name"].get(tok, look["id2name"].get(tok, tok))
    return tok


@lru_cache(maxsize=None)
def topic_id_to_name(topic_id_or_name: str) -> str:
    """
    Normalize a topic token (id like 'T13054' or name) to a topic name string.
    Prefers direct lookup in topics table.
    """
    tok = str(topic_id_or_name).strip()
    look = build_taxonomy_lookups()
    # If it's an ID present in our maps → return its name
    name = look["topic_id_to_name"].get(tok)
    if name:
        return name

    # If it's already a name present in the registry → return as-is
    if tok in look["name2id"]:
        return tok

    # Fallback: scan the table
    tdf = _load_topics()
    m = tdf.loc[tdf["topic_id"].astype(str) == tok]
    if not m.empty and "topic_name" in m.columns:
        return str(m["topic_name"].iloc[0])
    return tok


@lru_cache(maxsize=1)
def canonical_field_order() -> List[str]:
    return build_taxonomy_lookups()["canonical_fields"]


@lru_cache(maxsize=1)
def canonical_subfield_order() -> List[str]:
    return build_taxonomy_lookups()["canonical_subfields"]
