"""
columns.py — column maps + parsing utilities for the LUE Explorer app

This module documents schemas for all parquet tables and provides helpers to
normalize, parse and explode semi-structured columns (pipe-separated lists,
key (value) pairs, bracket-indexed arrays, etc.).

Glossary
- UL (Université de Lorraine): full perimeter of publications.
- LUE (Lorraine Université d’Excellence): subset of UL publications funded
  internally by the university. Most indicators include a LUE-specific slice
  so that you can compare the restrained perimeter vs the whole UL.

Design goals
1) Single source of truth for column names, descriptions and expected dtypes.
2) Robust parsers to unlock granularity (authors, labs, partners, topics,
   year/type distributions, "By X:" rollups) for streamlit visuals.
3) Gentle normalization helpers (snake_case + alias map) without forcing you
   to rename columns in persisted files.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional, Callable
import pandas as pd

# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSpec:
    name: str           # exact column name as it appears in the parquet file
    dtype: str          # human-readable expected dtype ("string", "int", "float", "bool", "json", "list", ...)
    desc: str           # short description

# ---- pubs_final.parquet ----------------------------------------------------
PUBS_FINAL: List[ColumnSpec] = [
    ColumnSpec("OpenAlex ID", "string", "OpenAlex work identifier (W...)"),
    ColumnSpec("DOI", "string", "Digital Object Identifier"),
    ColumnSpec("Title", "string", "Publication title"),
    ColumnSpec("Publication Year", "int", "Year of publication"),
    ColumnSpec("Publication Type", "string", "Work type (article, review, chapter, book, etc.)"),
    ColumnSpec("Authors", "list[str] (pipe or bracket-indexed)", "Ordered author names. Often like '[1] Alice | [2] Bob'"),
    ColumnSpec("Authors ID", "list[str] (pipe or bracket-indexed)", "OpenAlex Author IDs (A...) aligned with Authors"),
    ColumnSpec("Authors ORCID", "list[str] (pipe or bracket-indexed)", "Author ORCIDs aligned with Authors"),
    ColumnSpec("Institutions", "list[str] (pipe or bracket-indexed)", "Affiliated institutions per author"),
    ColumnSpec("Institution Types", "list[str] (pipe or bracket-indexed)", "Institution types (university, funder, facility, ... )"),
    ColumnSpec("Institution Countries", "list[str] (pipe or bracket-indexed)", "Institution countries (ISO names)"),
    ColumnSpec("Institutions ID", "list[str] (pipe or bracket-indexed)", "Institution IDs (I...) aligned with Institutions"),
    ColumnSpec("Institutions ROR", "list[str] (pipe or bracket-indexed)", "ROR IDs aligned with Institutions"),
    ColumnSpec("Additional Lineage", "list[str] (semicolon / pipe)", "Extra parent orgs / consortia when present"),
    ColumnSpec("FWCI_all", "float", "Field-Weighted Citation Impact (global baseline)"),
    ColumnSpec("Citation Count", "int", "Total citations to date"),
    ColumnSpec("Citations per Year", "list[year->count]", "Per-year citation counts formatted as 'count (year) | ...'"),
    ColumnSpec("Primary Topic", "string", "Primary OpenAlex topic ID (T...) for the work"),
    ColumnSpec("Primary Subfield ID", "int", "Primary subfield code (e.g., 2505)"),
    ColumnSpec("Primary Field ID", "int", "Primary field code (e.g., 25)"),
    ColumnSpec("Primary Domain ID", "int", "Primary domain code (1..4)"),
    ColumnSpec("In_LUE", "bool", "True if the publication is in the LUE perimeter"),
    ColumnSpec("Labs_RORs", "list[str] (pipe)", "RORs of UL labs acknowledged in the paper"),
    ColumnSpec("Is_PPtop10%_(field)", "bool", "Work is in top 10% by citations within its field"),
    ColumnSpec("Is_PPtop1%_(field)", "bool", "Work is in top 1% by citations within its field"),
    ColumnSpec("Is_PPtop10%_(subfield)", "bool", "Work is in top 10% by citations within its subfield"),
    ColumnSpec("Is_PPtop1%_(subfield)", "bool", "Work is in top 1% by citations within its subfield"),
    ColumnSpec("FWCI_FR", "float", "Field-Weighted Citation Impact using France baseline"),
    ColumnSpec("Is_international", "bool", "At least one non-French affiliation"),
    ColumnSpec("Is_company", "bool", "At least one company affiliation"),
]

# ---- ul_authors_indicators.parquet ----------------------------------------
UL_AUTHORS: List[ColumnSpec] = [
    ColumnSpec("Author Name", "string", "Preferred display name"),
    ColumnSpec("Normalized Name", "string", "Lowercased / normalized variant for grouping"),
    ColumnSpec("Author ID", "list[str] (pipe)", "OpenAlex Author ID(s) if merged"),
    ColumnSpec("ORCID", "list[str] (pipe)", "ORCID(s) if multiple identities"),
    ColumnSpec("All Institution RORs (|)", "list[str] (pipe)", "All associated RORs"),
    ColumnSpec("Publications (unique)", "int", "Distinct UL-linked publications"),
    ColumnSpec("Average FWCI_all", "float", "Mean FWCI (global baseline) across works"),
    ColumnSpec("Average FWCI_FR", "float", "Mean FWCI (France baseline) across works"),
    ColumnSpec("PPtop10% Count", "int", "Count of author's works in top 10% (field-normalized)"),
    ColumnSpec("PPtop10% Fields", "list[str] (pipe)", "Fields where author has top10% works with counts in parentheses"),
    ColumnSpec("PPtop10% Subfields", "list[str] (pipe)", "Subfields where author has top10% works with counts in parentheses"),
    ColumnSpec("PPtop1% Count", "int", "Count of author's works in top 1% (field-normalized)"),
    ColumnSpec("PPtop1% Fields", "list[str] (pipe)", "Fields for top1% with counts"),
    ColumnSpec("PPtop1% Subfields", "list[str] (pipe)", "Subfields for top1% with counts"),
    ColumnSpec("All Publication IDs (|)", "list[str] (pipe)", "All OpenAlex work IDs associated to the author"),
    ColumnSpec("Is Lorraine", "bool", "True if the author is UL-affiliated"),
    ColumnSpec("Lab(s)", "list[str] (pipe)", "Primary UL lab(s) associated with the author"),
]

# ---- ul_fields_indicators.parquet -----------------------------------------
UL_FIELDS: List[ColumnSpec] = [
    ColumnSpec("Field ID", "int", "OpenAlex field code"),
    ColumnSpec("Field name", "string", "Human-readable field name"),
    ColumnSpec("Pubs", "int", "Number of UL publications in this field"),
    ColumnSpec("Year distribution (2019-2023)", "list[int] (pipe)", "Per-year counts 2019|2020|2021|2022|2023"),
    ColumnSpec("Type distribution (articles|chapters|books|reviews, 2019-2023)", "list[int] (pipe)", "Counts per type aggregated over 2019-2023"),
    ColumnSpec("% Pubs (uni level)", "float", "Share of this field in all UL publications"),
    ColumnSpec("Pubs LUE", "int", "Number of LUE publications in this field"),
    ColumnSpec("% Pubs LUE (field level)", "float", "LUE share within this field"),
    ColumnSpec("% Pubs LUE (uni level)", "float", "This field's LUE share relative to all LUE publications"),
    ColumnSpec("PPtop10%", "int", "Count of UL top-10% papers in this field"),
    ColumnSpec("% PPtop10% (field level)", "float", "Share of field's papers that are top-10%"),
    ColumnSpec("% PPtop10% (uni level)", "float", "Share of UL top-10% papers contributed by this field"),
    ColumnSpec("PPtop1%", "int", "Count of UL top-1% papers in this field"),
    ColumnSpec("% PPtop1% (field level)", "float", "Share of field's papers that are top-1%"),
    ColumnSpec("% PPtop1% (uni level)", "float", "Share of UL top-1% papers contributed by this field"),
    ColumnSpec("Avg FWCI (France)", "float", "Average FWCI using FR baseline"),
    ColumnSpec("FWCI_FR min", "float", "Minimum FWCI_FR among works in this field"),
    ColumnSpec("FWCI_FR Q1", "float", "First quartile of FWCI_FR"),
    ColumnSpec("FWCI_FR Q2", "float", "Median FWCI_FR"),
    ColumnSpec("FWCI_FR Q3", "float", "Third quartile of FWCI_FR"),
    ColumnSpec("FWCI_FR max", "float", "Maximum FWCI_FR"),
    ColumnSpec("% internal collaboration", "float", "Share with another UL lab (intra-UL)"),
    ColumnSpec("% international", "float", "Share with at least one foreign org"),
    ColumnSpec("% industrial", "float", "Share with at least one company"),
    ColumnSpec("By subfield: count", "list[key(count)]", "Subfield counts formatted as 'subfield_id (count) | ...'"),
    ColumnSpec("By subfield: LUE count", "list[key(count)]", "LUE-only subfield counts"),
    ColumnSpec("By subfield: % of field pubs", "list[key(value)]", "Subfield share within this field"),
    ColumnSpec("By subfield: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by subfield"),
    ColumnSpec("By subfield: TOP10 count", "list[key(count)]", "Top-10% counts by subfield"),
    ColumnSpec("By subfield: TOP10 % of field pubs", "list[key(value)]", "Top-10% share by subfield"),
    ColumnSpec("By subfield: TOP1 count", "list[key(count)]", "Top-1% counts by subfield"),
    ColumnSpec("By subfield: TOP1 % of field pubs", "list[key(value)]", "Top-1% share by subfield"),
    ColumnSpec("Lab coverage", "float", "Share of field pubs linked to at least one UL lab"),
    ColumnSpec("By lab: count", "list[key(count)]", "Counts by UL lab (ROR or code)"),
    ColumnSpec("By lab: % of field pubs", "list[key(value)]", "Lab share within this field"),
    ColumnSpec("By lab: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by lab within this field"),
    ColumnSpec("By lab: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by lab"),
    ColumnSpec("Top 10 authors (name)", "list[str] (pipe)", "Top authors by pubs in this field"),
    ColumnSpec("Top 10 authors (pubs)", "list[int] (pipe)", "Parallel list of pub counts for top authors"),
    ColumnSpec("Top 10 authors (Orcid)", "list[str] (pipe)", "Parallel list of ORCIDs"),
    ColumnSpec("Top 10 authors (ID)", "list[str] (pipe)", "Parallel list of OpenAlex author IDs"),
    ColumnSpec("Top 10 authors (Average FWCI_FR)", "list[float] (pipe)", "Parallel list of avg FWCI_FR"),
    ColumnSpec("Top 10 authors (PPtop10% Count)", "list[int] (pipe)", "Parallel list of top10 counts"),
    ColumnSpec("Top 10 authors (PPtop1% Count)", "list[int] (pipe)", "Parallel list of top1 counts"),
    ColumnSpec("Top 10 authors (Is Lorraine)", "list[bool] (pipe)", "Parallel list: whether each author is UL"),
    ColumnSpec("Top 10 authors (Labs)", "list[list[str]] (pipe;semicolon)", "Parallel list of lab(s) per author; labs split by ';'"),
    ColumnSpec("Top 10 int partners (name)", "list[str] (pipe)", "Top international (non-FR) partners by co-pubs in this field"),
    ColumnSpec("Top 10 int partners (type)", "list[str] (pipe)", "Parallel list: partner types"),
    ColumnSpec("Top 10 int partners (country)", "list[str] (pipe)", "Parallel list: partner countries"),
    ColumnSpec("Top 10 int partners (totals copubs in this field)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 10 int partners (% of UL total copubs)", "list[float] (pipe)", "Parallel list: partner share vs UL copubs"),
    ColumnSpec("Top 10 FR partners (name)", "list[str] (pipe)", "Top French partners by co-pubs in this field"),
    ColumnSpec("Top 10 FR partners (type)", "list[str] (pipe)", "Parallel list: types"),
    ColumnSpec("Top 10 FR partners (totals copubs in this field)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 10 FR partners (% of UL total copubs)", "list[float] (pipe)", "Parallel list: share vs UL copubs"),
    ColumnSpec("See in OpenAlex", "url", "Convenience link to a pre-filtered OpenAlex query"),
]

# ---- ul_domains_indicators.parquet ----------------------------------------
UL_DOMAINS: List[ColumnSpec] = [
    ColumnSpec("Domain ID", "int", "OpenAlex domain code"),
    ColumnSpec("Domain name", "string", "Human-readable domain name"),
    ColumnSpec("Pubs", "int", "Number of UL publications in this domain"),
    ColumnSpec("Year distribution (2019-2023)", "list[int] (pipe)", "Per-year counts 2019|2020|2021|2022|2023"),
    ColumnSpec("Type distribution (articles|chapters|books|reviews, 2019-2023)", "list[int] (pipe)", "Counts per type aggregated 2019-2023"),
    ColumnSpec("% Pubs (uni level)", "float", "Share of this domain in all UL publications"),
    ColumnSpec("Pubs LUE", "int", "Number of LUE publications in this domain"),
    ColumnSpec("% Pubs LUE (domain level)", "float", "LUE share within this domain"),
    ColumnSpec("% Pubs LUE (uni level)", "float", "This domain's LUE share relative to all LUE"),
    ColumnSpec("PPtop10%", "int", "Count of UL top-10% papers in this domain"),
    ColumnSpec("% PPtop10% (domain level)", "float", "Share of domain's papers that are top-10%"),
    ColumnSpec("% PPtop10% (uni level)", "float", "Share of UL top-10% papers contributed by this domain"),
    ColumnSpec("PPtop1%", "int", "Count of UL top-1% papers in this domain"),
    ColumnSpec("% PPtop1% (domain level)", "float", "Share of domain's papers that are top-1%"),
    ColumnSpec("% PPtop1% (uni level)", "float", "Share of UL top-1% papers contributed by this domain"),
    ColumnSpec("Avg FWCI (France)", "float", "Average FWCI using FR baseline"),
    ColumnSpec("FWCI_FR min", "float", "Minimum FWCI_FR among works in this domain"),
    ColumnSpec("FWCI_FR Q1", "float", "First quartile of FWCI_FR"),
    ColumnSpec("FWCI_FR Q2", "float", "Median FWCI_FR"),
    ColumnSpec("FWCI_FR Q3", "float", "Third quartile of FWCI_FR"),
    ColumnSpec("FWCI_FR max", "float", "Maximum FWCI_FR"),
    ColumnSpec("% internal collaboration", "float", "Share with another UL lab (intra-UL)"),
    ColumnSpec("% international", "float", "Share with at least one foreign org"),
    ColumnSpec("% industrial", "float", "Share with at least one company"),
    ColumnSpec("By field: count", "list[key(count)]", "Field counts formatted as 'field_id (count) | ...'"),
    ColumnSpec("By field: LUE count", "list[key(count)]", "LUE-only field counts"),
    ColumnSpec("By field: % of domain pubs", "list[key(value)]", "Field share within this domain"),
    ColumnSpec("By field: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by field within this domain"),
    ColumnSpec("By field: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by field"),
    ColumnSpec("By field: TOP10 count", "list[key(count)]", "Top-10% counts by field"),
    ColumnSpec("By field: TOP10 % of domain pubs", "list[key(value)]", "Top-10% share by field"),
    ColumnSpec("By field: TOP1 count", "list[key(count)]", "Top-1% counts by field"),
    ColumnSpec("By field: TOP1 % of domain pubs", "list[key(value)]", "Top-1% share by field"),
    ColumnSpec("Lab coverage", "float", "Share of domain pubs linked to at least one UL lab"),
    ColumnSpec("By lab: count", "list[key(count)]", "Counts by UL lab (ROR or code)"),
    ColumnSpec("By lab: % of domain pubs", "list[key(value)]", "Lab share within this domain"),
    ColumnSpec("By lab: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by lab within this domain"),
    ColumnSpec("By lab: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by lab"),
    ColumnSpec("Top 20 authors (name)", "list[str] (pipe)", "Top authors by pubs in this domain"),
    ColumnSpec("Top 20 authors (pubs)", "list[int] (pipe)", "Parallel list of pub counts for top authors"),
    ColumnSpec("Top 20 authors (Orcid)", "list[str] (pipe;semicolon)", "Parallel list of ORCIDs; sometimes two values per author"),
    ColumnSpec("Top 20 authors (ID)", "list[str] (pipe;semicolon)", "Parallel list of OpenAlex author IDs; sometimes two per author"),
    ColumnSpec("Top 20 authors (Average FWCI_FR)", "list[float] (pipe)", "Parallel list of avg FWCI_FR"),
    ColumnSpec("Top 20 authors (PPtop10% Count)", "list[int] (pipe)", "Parallel list of top10 counts"),
    ColumnSpec("Top 20 authors (PPtop1% Count)", "list[int] (pipe)", "Parallel list of top1 counts"),
    ColumnSpec("Top 20 authors (Is Lorraine)", "list[bool] (pipe)", "Parallel list: whether each author is UL"),
    ColumnSpec("Top 20 authors (Labs)", "list[list[str]] (pipe;semicolon)", "Parallel list of lab(s) per author; labs split by ';'"),
    ColumnSpec("Top 20 int partners (name)", "list[str] (pipe)", "Top international partners by co-pubs in this domain"),
    ColumnSpec("Top 20 int partners (type)", "list[str] (pipe)", "Parallel list: partner types"),
    ColumnSpec("Top 20 int partners (country)", "list[str] (pipe)", "Parallel list: partner countries"),
    ColumnSpec("Top 20 int partners (totals copubs in this domain)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 20 int partners (% of UL total copubs)", "list[float] (pipe)", "Parallel list: partner share vs UL copubs"),
    ColumnSpec("Top 20 FR partners (name)", "list[str] (pipe)", "Top French partners by co-pubs in this domain"),
    ColumnSpec("Top 20 FR partners (type)", "list[str] (pipe)", "Parallel list: types"),
    ColumnSpec("Top 20 FR partners (totals copubs in this domain)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 20 FR partners (% of UL total copubs)", "list[float] (pipe)", "Parallel list: share vs UL copubs"),
    ColumnSpec("See in OpenAlex", "url", "Convenience link to a pre-filtered OpenAlex query"),
]

# ---- ul_partners_indicators.parquet ---------------------------------------
UL_PARTNERS: List[ColumnSpec] = [
    ColumnSpec("Institution ID", "string", "OpenAlex institution ID (I...)"),
    ColumnSpec("Institution ROR", "string", "ROR ID"),
    ColumnSpec("Institution name", "string", "Partner organization name"),
    ColumnSpec("Institution type", "string", "Type (government, funder, university, facility, healthcare, ... )"),
    ColumnSpec("Country", "string", "Country name"),
    ColumnSpec("Copublications", "int", "Total co-publications with UL (2019-2023)"),
    ColumnSpec("Year distribution (2019-2023)", "list[int] (pipe)", "Per-year counts 2019|2020|2021|2022|2023"),
    ColumnSpec("% of UL total pubs", "float", "Share of UL publications co-authored with this partner"),
    ColumnSpec("Pubs LUE", "int", "Number of LUE co-publications with this partner"),
    ColumnSpec("% Pubs LUE (uni level)", "float", "Partner share relative to all LUE publications"),
    ColumnSpec("PPtop10%", "int", "Top-10% co-pubs count with this partner"),
    ColumnSpec("% PPtop10% (field level)", "float", "Share of partner's co-pubs that are top-10% within field"),
    ColumnSpec("% PPtop10% (uni level)", "float", "Share of UL top-10% co-pubs contributed by this partner"),
    ColumnSpec("PPtop1%", "int", "Top-1% co-pubs count with this partner"),
    ColumnSpec("% PPtop1% (field level)", "float", "Share of partner's co-pubs that are top-1% within field"),
    ColumnSpec("% PPtop1% (uni level)", "float", "Share of UL top-1% co-pubs contributed by this partner"),
    ColumnSpec("Avg FWCI_FR", "float", "Average FWCI using FR baseline"),
    ColumnSpec("FWCI_FR min", "float", "Minimum FWCI_FR among co-pubs with this partner"),
    ColumnSpec("FWCI_FR Q1", "float", "First quartile of FWCI_FR"),
    ColumnSpec("FWCI_FR Q2", "float", "Median FWCI_FR"),
    ColumnSpec("FWCI_FR Q3", "float", "Third quartile of FWCI_FR"),
    ColumnSpec("FWCI_FR max", "float", "Maximum FWCI_FR"),
    ColumnSpec("Lab coverage", "float", "Share of co-pubs linked to at least one UL lab"),
    ColumnSpec("By lab: count", "list[key(count)]", "Counts by UL lab for this partner"),
    ColumnSpec("By lab: % of total copubs", "list[key(value)]", "Lab share of partner co-pubs"),
    ColumnSpec("By lab: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by lab with this partner"),
    ColumnSpec("By lab: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by lab for this partner"),
    ColumnSpec("By lab: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by lab"),
    ColumnSpec("By lab: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by lab"),
    ColumnSpec("Top 10 authors (name)", "list[str] (pipe)", "Top UL authors co-publishing with this partner"),
    ColumnSpec("Top 10 authors (pubs)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 10 authors (Orcid)", "list[str] (pipe)", "Parallel list: ORCIDs"),
    ColumnSpec("Top 10 authors (ID)", "list[str] (pipe)", "Parallel list: OpenAlex author IDs"),
    ColumnSpec("Top 10 authors (Average FWCI_FR)", "list[float] (pipe)", "Parallel list: avg FWCI_FR"),
    ColumnSpec("Top 10 authors (PPtop10% Count)", "list[int] (pipe)", "Parallel list: top10 counts"),
    ColumnSpec("Top 10 authors (PPtop1% Count)", "list[int] (pipe)", "Parallel list: top1 counts"),
    ColumnSpec("Top 10 authors (Is Lorraine)", "list[bool] (pipe)", "Parallel list: is UL?"),
    ColumnSpec("Top 10 authors (Labs)", "list[list[str]] (pipe;semicolon)", "Parallel list of lab(s) per author; labs split by ';'"),
    ColumnSpec("By domain: count", "list[key(count)]", "Domain counts for this partner"),
    ColumnSpec("By domain: % of total copubs", "list[key(value)]", "Domain share of partner co-pubs"),
    ColumnSpec("By domain: LUE count", "list[key(count)]", "LUE-only domain counts"),
    ColumnSpec("By domain: LUE % of total copubs", "list[key(value)]", "LUE share within partner co-pubs"),
    ColumnSpec("By domain: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by domain"),
    ColumnSpec("By domain: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by domain"),
    ColumnSpec("By domain: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by domain"),
    ColumnSpec("By domain: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by domain"),
    ColumnSpec("By domain: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by domain"),
    ColumnSpec("By domain: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by domain"),
    ColumnSpec("By domain: TOP10 count", "list[key(count)]", "Top-10% counts by domain"),
    ColumnSpec("By domain: TOP10 % of total copubs", "list[key(value)]", "Top-10% share by domain"),
    ColumnSpec("By domain: TOP1 count", "list[key(count)]", "Top-1% counts by domain"),
    ColumnSpec("By domain: TOP1 % of total copubs", "list[key(value)]", "Top-1% share by domain"),
    ColumnSpec("By field: count", "list[key(count)]", "Field counts for this partner"),
    ColumnSpec("By field: % of total copubs", "list[key(value)]", "Field share of partner co-pubs"),
    ColumnSpec("By field: LUE count", "list[key(count)]", "LUE-only field counts"),
    ColumnSpec("By field: LUE % of total copubs", "list[key(value)]", "LUE share within partner co-pubs"),
    ColumnSpec("By field: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by field"),
    ColumnSpec("By field: TOP10 count", "list[key(count)]", "Top-10% counts by field"),
    ColumnSpec("By field: TOP10 % of total copubs", "list[key(value)]", "Top-10% share by field"),
    ColumnSpec("By field: TOP1 count", "list[key(count)]", "Top-1% counts by field"),
    ColumnSpec("By field: TOP1 % of total copubs", "list[key(value)]", "Top-1% share by field"),
    ColumnSpec("By subfield: count", "list[key(count)]", "Subfield counts for this partner"),
    ColumnSpec("By subfield: % of total copubs", "list[key(value)]", "Subfield share of partner co-pubs"),
    ColumnSpec("By subfield: LUE count", "list[key(count)]", "LUE-only subfield counts"),
    ColumnSpec("By subfield: LUE % of total copubs", "list[key(value)]", "LUE share within partner co-pubs"),
    ColumnSpec("By subfield: avg FWCI_FR", "list[key(value)]", "Average FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by subfield"),
    ColumnSpec("By subfield: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by subfield"),
    ColumnSpec("By subfield: TOP10 count", "list[key(count)]", "Top-10% counts by subfield"),
    ColumnSpec("By subfield: TOP10 % of total copubs", "list[key(value)]", "Top-10% share by subfield"),
    ColumnSpec("By subfield: TOP1 count", "list[key(count)]", "Top-1% counts by subfield"),
    ColumnSpec("By subfield: TOP1 % of total copubs", "list[key(value)]", "Top-1% share by subfield"),
    ColumnSpec("Total works 2019-2023", "int", "Total works for partner (redundant with Copublications but kept)"),
    ColumnSpec("Total works per field", "list[key(count)]", "Partner's totals by field across 2019-2023"),
    ColumnSpec("Total works per subfield", "list[key(count)]", "Partner's totals by subfield across 2019-2023"),
]

# ---- ul_units_indicators.parquet ------------------------------------------
UL_UNITS: List[ColumnSpec] = [
    ColumnSpec("ROR", "string", "Lab ROR ID"),
    ColumnSpec("OpenAlex ID", "string", "OpenAlex institution ID for the lab (I...)"),
    ColumnSpec("Unit Name", "string", "Laboratory / research unit name"),
    ColumnSpec("Department", "string", "Department or faculty (when available)"),
    ColumnSpec("Type", "string", "Institutional type (e.g., joint unit, lab)"),
    ColumnSpec("Pubs", "int", "Number of UL publications linked to this lab"),
    ColumnSpec("Year distribution (2019-2023)", "list[int] (pipe)", "Per-year counts 2019|2020|2021|2022|2023"),
    ColumnSpec("Type distribution (articles|chapters|books|reviews, 2019-2023)", "list[int] (pipe)", "Counts per type aggregated 2019-2023"),
    ColumnSpec("% Pubs (uni level)", "float", "Lab share of all UL publications"),
    ColumnSpec("Pubs LUE", "int", "Number of LUE publications for this lab"),
    ColumnSpec("% Pubs LUE (lab level)", "float", "LUE share within this lab"),
    ColumnSpec("% Pubs LUE (uni level)", "float", "This lab's LUE share relative to all LUE"),
    ColumnSpec("PPtop10%", "int", "Top-10% papers linked to this lab"),
    ColumnSpec("% PPtop10% (lab level)", "float", "Share of lab papers that are top-10%"),
    ColumnSpec("% PPtop10% (uni level)", "float", "Share of UL top-10% papers contributed by this lab"),
    ColumnSpec("PPtop1%", "int", "Top-1% papers linked to this lab"),
    ColumnSpec("% PPtop1% (lab level)", "float", "Share of lab papers that are top-1%"),
    ColumnSpec("% PPtop1% (uni level)", "float", "Share of UL top-1% papers contributed by this lab"),
    ColumnSpec("Avg FWCI (France)", "float", "Average FWCI using FR baseline"),
    ColumnSpec("Collab pubs (other labs)", "int", "Number of papers co-authored with another UL lab"),
    ColumnSpec("% collab w/ another internal lab", "float", "Share of lab's papers with intra-UL lab collaboration"),
    ColumnSpec("Collab labs (by ROR)", "list[key(count)]", "Key (lab ROR) with counts in parentheses"),
    ColumnSpec("Collab pubs (other structures)", "int", "Number of papers co-authored with other UL structures"),
    ColumnSpec("% collab w/ another internal structure", "float", "Share of lab's papers with other UL structures"),
    ColumnSpec("Collab other structures (by ROR)", "list[key(count)]", "Other UL structures (ROR) with counts"),
    ColumnSpec("% international", "float", "Share with at least one foreign org"),
    ColumnSpec("% industrial", "float", "Share with at least one company"),
    ColumnSpec("By field: counts", "list[key(count)]", "Field counts for this lab"),
    ColumnSpec("By field: LUE counts", "list[key(count)]", "LUE-only field counts for this lab"),
    ColumnSpec("By field: % of lab pubs", "list[key(value)]", "Field share within lab output"),
    ColumnSpec("By field: FWCI_FR min", "list[key(value)]", "Min FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q1", "list[key(value)]", "Q1 FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q2", "list[key(value)]", "Median FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR Q3", "list[key(value)]", "Q3 FWCI_FR by field"),
    ColumnSpec("By field: FWCI_FR max", "list[key(value)]", "Max FWCI_FR by field"),
    ColumnSpec("By field: TOP10 count", "list[key(count)]", "Top-10% counts by field"),
    ColumnSpec("By field: TOP10 % of lab pubs", "list[key(value)]", "Top-10% share by field"),
    ColumnSpec("By field: TOP1 count", "list[key(count)]", "Top-1% counts by field"),
    ColumnSpec("By field: TOP1 % of lab pubs", "list[key(value)]", "Top-1% share by field"),
    ColumnSpec("Top 10 authors (name)", "list[str] (pipe)", "Top authors by pubs in this lab"),
    ColumnSpec("Top 10 authors (ORCID)", "list[str] (pipe)", "Parallel list: ORCIDs"),
    ColumnSpec("Top 10 authors (ID)", "list[str] (pipe)", "Parallel list: OpenAlex author IDs"),
    ColumnSpec("Top 10 authors (pubs)", "list[int] (pipe)", "Parallel list: pub counts"),
    ColumnSpec("Top 10 authors (Average FWCI_FR)", "list[float] (pipe)", "Parallel list: avg FWCI_FR"),
    ColumnSpec("Top 10 authors (PPtop10% Count)", "list[int] (pipe)", "Parallel list: top10 counts"),
    ColumnSpec("Top 10 authors (PPtop1% Count)", "list[int] (pipe)", "Parallel list: top1 counts"),
    ColumnSpec("Top 10 authors (Is Lorraine)", "list[bool] (pipe)", "Parallel list: is UL?"),
    ColumnSpec("Top 10 authors (Other lab(s))", "list[list[str]] (pipe;semicolon)", "Other UL lab(s) per author; split by ';'"),
    ColumnSpec("Top 10 int partners (name)", "list[str] (pipe)", "Top international partners for this lab"),
    ColumnSpec("Top 10 int partners (type)", "list[str] (pipe)", "Parallel list: types"),
    ColumnSpec("Top 10 int partners (country)", "list[str] (pipe)", "Parallel list: countries"),
    ColumnSpec("Top 10 int partners (copubs with lab)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 10 int partners (% of UL copubs)", "list[float] (pipe)", "Parallel list: share vs UL"),
    ColumnSpec("Top 10 FR partners (name)", "list[str] (pipe)", "Top French partners for this lab"),
    ColumnSpec("Top 10 FR partners (type)", "list[str] (pipe)", "Parallel list: types"),
    ColumnSpec("Top 10 FR partners (copubs with lab)", "list[int] (pipe)", "Parallel list: counts"),
    ColumnSpec("Top 10 FR partners (% of UL copubs)", "list[float] (pipe)", "Parallel list: share vs UL"),
    ColumnSpec("See in OpenAlex", "url", "Convenience link to a pre-filtered OpenAlex query"),
]

# ---- all_topics.parquet ----------------------------------------------------
ALL_TOPICS: List[ColumnSpec] = [
    ColumnSpec("domain_id", "int", "OpenAlex domain code"),
    ColumnSpec("domain_name", "string", "Domain name"),
    ColumnSpec("field_id", "int", "OpenAlex field code"),
    ColumnSpec("field_name", "string", "Field name"),
    ColumnSpec("subfield_id", "int", "OpenAlex subfield code"),
    ColumnSpec("subfield_name", "string", "Subfield name"),
    ColumnSpec("topic_id", "string", "OpenAlex topic ID (T...)"),
    ColumnSpec("topic_name", "string", "Topic name"),
    ColumnSpec("keywords", "list[str] (pipe)", "Associated keywords"),
]

# Bundle all schemas in a single registry
SCHEMAS: Dict[str, List[ColumnSpec]] = {
    "pubs_final": PUBS_FINAL,
    "ul_authors_indicators": UL_AUTHORS,
    "ul_fields_indicators": UL_FIELDS,
    "ul_domains_indicators": UL_DOMAINS,
    "ul_partners_indicators": UL_PARTNERS,
    "ul_units_indicators": UL_UNITS,
    "all_topics": ALL_TOPICS,
}

# ---------------------------------------------------------------------------
# Column normalization helpers
# ---------------------------------------------------------------------------

_NORMALIZE_MAP: Dict[str, str] = {}

def normalize_column_name(name: str) -> str:
    """Return a snake_case, ascii-safe variant of a column name.
    Keeps a central alias map so you can reverse-look-up original names.
    Example: "Is_PPtop10%_(field)" -> "is_pptop10_field"
    """
    key = name
    s = name.lower()
    s = re.sub(r"\%", "pct", s)
    s = re.sub(r"[()\[\]]", "", s)
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    _NORMALIZE_MAP[s] = key
    return s

def original_column_name(normalized: str) -> Optional[str]:
    """If you need the original display name back."""
    return _NORMALIZE_MAP.get(normalized)

# ---------------------------------------------------------------------------
# Parsing primitives (reused across datasets)
# ---------------------------------------------------------------------------

_PIPE_SPLIT = re.compile(r"\s*\|\s*")
_SEMI_SPLIT = re.compile(r"\s*;\s*")
_IDX_PREFIX = re.compile(r"^\[\d+\]\s*")
_KEY_VAL = re.compile(r"\s*([^()]+?)\s*\(([^()]+)\)\s*")  # "key (value)"
_VAL_KEY = re.compile(r"\s*([^()]+)\s*\(([^()]+?)\)\s*")  # "value (key)" (rare)


def split_pipe(s: str) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [part.strip() for part in _PIPE_SPLIT.split(str(s).strip()) if part.strip()]


def split_semicolon(s: str) -> List[str]:
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return []
    return [part.strip() for part in _SEMI_SPLIT.split(str(s).strip()) if part.strip()]


def strip_bracket_index(prefix_item: str) -> str:
    """Remove leading "[1] ", "[23] " style indices from an item."""
    return _IDX_PREFIX.sub("", prefix_item).strip()


def parse_bracket_indexed_list(s: str) -> List[str]:
    """Parse strings that look like "[1] Alice | [2] Bob | [3] Carol".
    Returns ["Alice", "Bob", "Carol"]. Works for Authors, Institutions, etc.
    """
    return [strip_bracket_index(x) for x in split_pipe(s)]


def parse_key_value_items(s: str, *, value_cast: Callable[[str], float | int | str] = float) -> List[Tuple[str, float | int | str]]:
    """Parse "key (value) | key (value)" sequences.
    - key: often an ID (e.g., 1100, 03c4rpa03, country code, etc.) or a name
    - value: numeric (int or float) by default; use value_cast to customize
    Returns list of (key, value).
    """
    items = []
    for raw in split_pipe(s):
        m = _KEY_VAL.fullmatch(raw)
        if not m:
            # Try to be forgiving if the entire raw token is numeric
            if raw:
                items.append((raw, None))
            continue
        key, val = m.group(1).strip(), m.group(2).strip()
        try:
            val_cast = value_cast(val) if value_cast else val
        except Exception:
            try:
                val_cast = float(val)
            except Exception:
                val_cast = val
        items.append((key, val_cast))
    return items


def parse_value_key_items(s: str, *, value_cast: Callable[[str], float | int | str] = float) -> List[Tuple[str, float | int | str]]:
    """Parse "value (key) | value (key)" sequences (e.g., "901 (2025)").
    Returns list of (key, value).
    """
    items = []
    for raw in split_pipe(s):
        m = _VAL_KEY.fullmatch(raw)
        if not m:
            if raw:
                items.append((raw, None))
            continue
        val, key = m.group(1).strip(), m.group(2).strip()
        try:
            val_cast = value_cast(val) if value_cast else val
        except Exception:
            try:
                val_cast = float(val)
            except Exception:
                val_cast = val
        items.append((key, val_cast))
    return items


# ---------------------------------------------------------------------------
# DataFrame explode helpers
# ---------------------------------------------------------------------------

def explode_bracket_indexed_column(df: pd.DataFrame, column: str, *, out_col: Optional[str] = None) -> pd.DataFrame:
    """Explode a bracket-indexed + pipe-separated column into long format.
    Example (Authors -> author_name): one row per author per work.
    """
    out_col = out_col or normalize_column_name(column)
    temp = df[[column]].copy()
    temp[out_col] = temp[column].apply(parse_bracket_indexed_list)
    exploded = temp.explode(out_col, ignore_index=False).drop(columns=[column]).reset_index()
    return exploded


def explode_parallel_lists(df: pd.DataFrame, columns: List[str], *, out_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Explode several parallel pipe-separated columns together, preserving alignment.
    Useful for the many "Top N ..." blocks.
    """
    out_cols = out_cols or [normalize_column_name(c) for c in columns]
    tmp = df[columns].copy()
    for c, out_c in zip(columns, out_cols):
        tmp[out_c] = tmp[c].apply(split_pipe)
    # ensure equal lengths by padding with None
    max_len = tmp[out_cols[0]].apply(lambda x: len(x) if isinstance(x, list) else 0).max()
    for out_c in out_cols:
        tmp[out_c] = tmp[out_c].apply(lambda lst: (lst + [None] * (max_len - len(lst))) if isinstance(lst, list) else lst)
    # build DataFrame from lists and explode
    for out_c in out_cols:
        tmp[out_c] = tmp[out_c]
    tmp = tmp[out_cols]
    tmp = tmp.apply(pd.Series)
    long = tmp.stack().reset_index(level=1, drop=True).to_frame(name=out_cols[0])
    # The above gives only first column; instead, construct aligned rows
    rows = []
    for idx, row in df[columns].iterrows():
        lists = [split_pipe(row[c]) for c in columns]
        maxn = max(len(x) for x in lists)
        for i in range(maxn):
            values = [lists[j][i] if i < len(lists[j]) else None for j in range(len(columns))]
            rows.append((idx, *values))
    res = pd.DataFrame(rows, columns=["index", *out_cols]).set_index("index").reset_index()
    return res


def explode_key_value_column(df: pd.DataFrame, column: str, *, key_name: str = "key", value_name: str = "value", reverse: bool = False, value_cast: Callable[[str], float | int | str] = float) -> pd.DataFrame:
    """Explode a column formatted as "key (value) | key (value)" or the reverse.
    - reverse=True expects "value (key)" order (e.g., "901 (2025)")
    Returns columns: [index, key_name, value_name]
    """
    parser = parse_value_key_items if reverse else parse_key_value_items
    rows = []
    for idx, s in df[column].items():
        for k, v in parser(s, value_cast=value_cast):
            rows.append((idx, k, v))
    out = pd.DataFrame(rows, columns=["index", key_name, value_name]).set_index("index").reset_index()
    return out


def explode_nested_labs_list(df: pd.DataFrame, column: str, *, sep_outer: str = "|", sep_inner: str = ";", out_author_col: str = "author_pos", out_lab_col: str = "lab") -> pd.DataFrame:
    """Explode columns like "Top 10 authors (Labs)":
    - Outer separator is pipe: one item per author position
    - Inner separator is semicolon: multiple labs per author
    Outputs [index, author_pos (0-based), lab]
    """
    rows = []
    for idx, s in df[column].items():
        outer = split_pipe(s)
        for pos, item in enumerate(outer):
            for lab in split_semicolon(item):
                if lab:
                    rows.append((idx, pos, lab))
    return pd.DataFrame(rows, columns=["index", out_author_col, out_lab_col]).set_index("index").reset_index()


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

def schema_for(dataset_key: str) -> List[ColumnSpec]:
    return SCHEMAS[dataset_key]


def check_columns(df: pd.DataFrame, dataset_key: str, *, strict: bool = False) -> Dict[str, List[str]]:
    """Quick check of missing / unexpected columns for a given dataset.
    If strict=True, raises AssertionError on mismatch.
    """
    expected = [c.name for c in SCHEMAS[dataset_key]]
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    report = {"missing": missing, "extra": extra}
    if strict and (missing or extra):
        raise AssertionError(f"Schema mismatch for {dataset_key}: {report}")
    return report


# ---------------------------------------------------------------------------
# Convenience: specific parsers for common columns
# ---------------------------------------------------------------------------

# pubs_final specific
parse_authors = parse_bracket_indexed_list
parse_institutions = parse_bracket_indexed_list
parse_authors_ids = parse_bracket_indexed_list
parse_authors_orcid = parse_bracket_indexed_list
parse_institutions_ids = parse_bracket_indexed_list
parse_institutions_ror = parse_bracket_indexed_list


def parse_citations_per_year(s: str) -> List[Tuple[str, int]]:
    """Parse "count (year) | ..." into [(year, count), ...]."""
    return [(year, int(count)) for year, count in parse_value_key_items(s, value_cast=int)]


# generic helpers for year/type blocks

def parse_year_distribution(s: str, *, start_year: int = 2019, end_year: int = 2023) -> List[Tuple[int, int]]:
    """Parse "187 | 200 | 213 | 201 | 169" against a known year span.
    Returns [(2019, 187), ..., (2023, 169)]
    """
    vals = [int(x) for x in split_pipe(s)]
    years = list(range(start_year, end_year + 1))
    return list(zip(years, vals[: len(years)]))


def parse_type_distribution(s: str, *, labels: Iterable[str] = ("articles", "chapters", "books", "reviews")) -> List[Tuple[str, int]]:
    vals = [int(x) for x in split_pipe(s)]
    labs = list(labels)
    return list(zip(labs, vals[: len(labs)]))