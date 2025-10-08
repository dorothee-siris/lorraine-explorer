# taxonomy.py
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# Canonical domain order and palette (cascades to children)
_DOMAIN_ORDER_CANON = [
    "Health Sciences",
    "Life Sciences",
    "Physical Sciences",
    "Social Sciences",
    "Other",
]

_DOMAIN_COLORS = {
    "Health Sciences": "#F85C32",
    "Life Sciences": "#0CA750",
    "Physical Sciences": "#8190FF",
    "Social Sciences": "#FFCB3A",
    "Other": "#7f7f7f",
}

# Default path: repo_root/data/all_topics.parquet
_DEFAULT_TOPICS_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "all_topics.parquet"
)

# ----------------------------- loaders -----------------------------

@lru_cache(maxsize=1)
def _load_topics(topics_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load all_topics.parquet and normalize key columns.
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

    required = ["domain_id", "domain_name", "field_id", "field_name",
                "subfield_id", "subfield_name"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"all_topics.parquet is missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # coerce ids to int where possible (except topic_id which may be alphanumeric like 'T13054')
    for c in ("domain_id", "field_id", "subfield_id"):  # removed "topic_id" here
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # keep topic_id as string, if present
    if "topic_id" in df.columns:
        df["topic_id"] = df["topic_id"].astype(str).str.strip()

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

    # id/name maps
    id2name: Dict[str, str] = {}
    name2id: Dict[str, str] = {}
    for _, r in t.iterrows():
        mappings = [
            ("domain_id", "domain_name"),
            ("field_id", "field_name"),
            ("subfield_id", "subfield_name"),
        ]
        if "topic_id" in t.columns and "topic_name" in t.columns:
            mappings.append(("topic_id", "topic_name"))
        for id_col, name_col in mappings:
            _id = r[id_col]
            _nm = r[name_col]
            if pd.notna(_id) and pd.notna(_nm):
                if id_col == "topic_id":
                    id_key = str(_id).strip()       # keep alphanumeric topic ids as-is
                else:
                    id_key = str(int(_id))           # numeric ids as ints -> string
                id2name[id_key] = str(_nm)
                name2id[str(_nm)] = id_key

    return {
        "domain_order": domain_order,
        "fields_by_domain": fields_by_domain,
        "subfields_by_field": subfields_by_field,
        "canonical_fields": canonical_fields,
        "canonical_subfields": canonical_subfields,
        "id2name": id2name,
        "name2id": name2id,
    }


# ---------------------------- colors -------------------------------

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
    Field inherits its domain color.
    """
    look = build_taxonomy_lookups()
    # resolve field name if an id was passed
    field = field_name_or_id
    if str(field_name_or_id).isdigit():
        field = look["id2name"].get(str(field_name_or_id), str(field_name_or_id))

    # find parent domain
    for d, fields in look["fields_by_domain"].items():
        if field in fields:
            return get_domain_color(d)
    return _DOMAIN_COLORS["Other"]


@lru_cache(maxsize=None)
def get_subfield_color(subfield_name_or_id: str) -> str:
    """
    Subfield inherits its parent field's (domain) color.
    """
    look = build_taxonomy_lookups()

    subfield = subfield_name_or_id
    if str(subfield_name_or_id).isdigit():
        subfield = look["id2name"].get(str(subfield_name_or_id), str(subfield_name_or_id))

    # find field containing this subfield
    for f, subs in look["subfields_by_field"].items():
        if subfield in subs:
            return get_field_color(f)
    return _DOMAIN_COLORS["Other"]


@lru_cache(maxsize=None)
def get_domain_for_field(field_name_or_id: str) -> str:
    """
    Return the parent domain name for a given field (name or numeric id).
    If unknown, returns 'Other'.
    """
    look = build_taxonomy_lookups()
    field = str(field_name_or_id)
    if field.isdigit():
        field = look["id2name"].get(field, field)

    for dom, fields in look["fields_by_domain"].items():
        if field in fields:
            return dom
    return "Other"


@lru_cache(maxsize=None)
def field_id_to_name(field_id_or_name: str) -> str:
    """
    Normalize a field token (either id or name) to a field name string.
    If not resolvable, returns the original token.
    """
    tok = str(field_id_or_name).strip()
    look = build_taxonomy_lookups()
    if tok.isdigit():
        return look["id2name"].get(tok, tok)
    return tok

@lru_cache(maxsize=None)
def topic_id_to_name(topic_id_or_name: str) -> str:
    """
    Normalize a topic token (id like 'T13054' or name) to a topic name string.
    If not resolvable, returns the original token.
    """
    tok = str(topic_id_or_name).strip()
    look = build_taxonomy_lookups()
    # if already a name
    if tok in look["name2id"]:
        return tok
    # try id -> name
    return look["id2name"].get(tok, tok)

@lru_cache(maxsize=None)
def get_topic_color(topic_name_or_id: str) -> str:
    """
    Topic inherits its parent domain color.
    """
    tok = str(topic_name_or_id).strip()
    tdf = _load_topics()
    # Resolve by id first
    m = tdf.loc[tdf["topic_id"].astype(str) == tok]
    if m.empty:
        # try by name
        m = tdf.loc[tdf.get("topic_name", "").astype(str) == tok]
    if not m.empty:
        dom = m["domain_name"].iloc[0]
        return get_domain_color(dom)
    return _DOMAIN_COLORS["Other"]

# --------------------------- conveniences --------------------------

@lru_cache(maxsize=1)
def canonical_field_order() -> List[str]:
    return build_taxonomy_lookups()["canonical_fields"]

@lru_cache(maxsize=1)
def canonical_subfield_order() -> List[str]:
    return build_taxonomy_lookups()["canonical_subfields"]
