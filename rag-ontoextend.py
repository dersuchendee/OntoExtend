from __future__ import annotations

import asyncio, base64, io, json, os, re, sys, zipfile, shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple, Optional, Dict
import aiofiles, aiohttp, faiss, numpy as np, openai  # type: ignore
from owlready2 import get_ontology, sync_reasoner_pellet  # type: ignore
from pydantic import BaseModel, Field
from rich import print
from tiktoken import get_encoding
import csv
import pandas as pd
from tqdm import tqdm
import time
import rdflib
from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
from copy import deepcopy
from collections import defaultdict

# === pricing you control (USD per 1K tokens) ===
# Prefer env vars so you don't hardcode changing prices.
def _p(env, default):
    v = os.getenv(env)
    return float(v) if v else default

PRICES_PER_1K = {
    # Chat/completions models (input vs output)
    "gpt-4o": {
        "input": _p("PRICE_GPT_4O_INPUT_PER_1K", None),   # e.g., 0.0025  (set in env)
        "output": _p("PRICE_GPT_4O_OUTPUT_PER_1K", None),  # e.g., 0.0100  (set in env)
    },
    # Embeddings models (input only)
    "text-embedding-3-small": {
        "input": _p("PRICE_TEXT_EMBEDDING_3_SMALL_PER_1K", 0.00002),  # $0.02 / 1M by default
    },
}




class TokenCostTracker:
    def __init__(self):
        # per-model counters
        self.totals = defaultdict(lambda: {"prompt": 0, "completion": 0, "embedding": 0})

    def snapshot(self):
        return deepcopy(self.totals)

    def diff(self, before, after):
        out = {}
        for m in set(before.keys()) | set(after.keys()):
            out[m] = {
                "prompt":  after[m]["prompt"]  - before.get(m, {}).get("prompt", 0),
                "completion": after[m]["completion"] - before.get(m, {}).get("completion", 0),
                "embedding": after[m]["embedding"] - before.get(m, {}).get("embedding", 0),
            }
        return out

    def note_chat(self, model: str, usage) -> None:
        # usage has prompt_tokens, completion_tokens, total_tokens
        if not usage:
            return
        self.totals[model]["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.totals[model]["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)

    def note_embed(self, model: str, usage, fallback_text: Optional[str] = None) -> None:
        if usage and getattr(usage, "total_tokens", None) is not None:
            self.totals[model]["embedding"] += int(usage.total_tokens)
        elif fallback_text is not None:
            # conservative fallback: count with tiktoken if API didn't return usage
            self.totals[model]["embedding"] += len(ENC.encode(fallback_text))

    def model_cost_usd(self, model: str, counts: Dict[str, int]) -> float:
        p = PRICES_PER_1K.get(model, {})
        cin  = (counts.get("prompt", 0) + counts.get("embedding", 0)) / 1000.0 * (p.get("input") or 0.0)
        cout = counts.get("completion", 0) / 1000.0 * (p.get("output") or 0.0)
        return cin + cout

    def totals_cost_usd(self) -> float:
        total = 0.0
        for m, c in self.totals.items():
            total += self.model_cost_usd(m, c)
        return total

token_tracker = TokenCostTracker()


ENC = get_encoding("cl100k_base")
CACHE_DIR = Path.home() / ".core_ontology_rag_cache"
CACHE_DIR.mkdir(exist_ok=True)

EMBED_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536  # embedding-3-small

openai_client = openai.AsyncOpenAI()

# CSV files for comprehensive logging
RESULTS_CSV = CACHE_DIR / "generation_runs.csv"
ELEMENTS_CSV = CACHE_DIR / "ontology_elements.csv"
PROCESSING_LOG_CSV = CACHE_DIR / "cq_processing_log.csv"

# Output files
COMBINED_ONTOLOGY_FILE = CACHE_DIR / "combined_ontology.ttl"
INDIVIDUAL_ONTOLOGIES_DIR = CACHE_DIR / "individual_ontologies"
INDIVIDUAL_ONTOLOGIES_DIR.mkdir(exist_ok=True)
TOKEN_COST_CSV = CACHE_DIR / "token_cost_log.csv"

def log_cost_row(cq_index: int, cq: str, deltas: Dict[str, Dict[str, int]]) -> None:
    TOKEN_COST_CSV.parent.mkdir(exist_ok=True)
    newfile = not TOKEN_COST_CSV.exists()
    with TOKEN_COST_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow([
                "timestamp", "cq_index", "cq",
                "model", "prompt_tokens", "completion_tokens", "embedding_tokens", "cost_usd"
            ])
        for model, counts in deltas.items():
            cost = token_tracker.model_cost_usd(model, counts)
            writer.writerow([
                datetime.now().isoformat(),
                cq_index,
                cq[:200] + ("..." if len(cq) > 200 else ""),
                model,
                counts.get("prompt", 0),
                counts.get("completion", 0),
                counts.get("embedding", 0),
                f"{cost:.6f}",
            ])
# ── Prefix collection ──────────────────────────────────────────────────────
STANDARD_PREFIXES = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

GLOBAL_PREFIX_BLOCK = ""  # filled at build_store(..)
# Add near your prefix utilities
PREFIX_NAME_SAFE = re.compile(r'^[A-Za-z_][A-Za-z0-9_\-]*$')

def sanitize_prefix_name(raw: Optional[str]) -> str:
    # Normalize, replace bad chars with '_'
    s = (raw or "").strip()
    s = re.sub(r'[^A-Za-z0-9_\-]', '_', s)

    # Must start with a letter or underscore; fall back to 'ns'
    if not s or not re.match(r'^[A-Za-z_]', s):
        s = f"ns_{s}" if s else "ns"

    # Collapse repeats like '__' → '_'
    s = re.sub(r'_+', '_', s)
    return s


def collect_prefixes_from_files(ontology_files: List[str]) -> Dict[str, str]:
    """Return {namespace_iri: prefix} using each file's declared prefixes, de-duplicated & sanitized."""
    ns_to_prefix: Dict[str, str] = {}
    used: set[str] = set(STANDARD_PREFIXES.keys())  # reserve standard names like rdf, rdfs, ...

    for f in ontology_files:
        g = Graph()
        try:
            g.parse(f)
        except Exception:
            continue

        for pref, ns in g.namespaces():
            ns = str(ns)
            # Skip XML and empty namespace
            if not ns or (pref and pref.lower() == "xml"):
                continue
            # Skip standard namespaces (we add them ourselves)
            if ns in STANDARD_PREFIXES.values():
                continue

            # Sanitize the proposed prefix
            p = sanitize_prefix_name(pref)

            # Ensure uniqueness: ex, ex2, ex3, ...
            while p in used:
                m = re.search(r'(\d+)$', p)
                if m:
                    p = re.sub(r'\d+$', str(int(m.group(1)) + 1), p)
                else:
                    p = f"{p}2"

            ns_to_prefix[ns] = p
            used.add(p)

    return ns_to_prefix


def build_prefix_block(ontology_files: List[str]) -> str:
    """Render @prefix lines using collected prefixes + your base ':' + standards."""
    ns_to_prefix = collect_prefixes_from_files(ontology_files)
    lines = [
        '@prefix : <http://www.example.org/ontology#> .',
        '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
        '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
        '@prefix owl: <http://www.w3.org/2002/07/owl#> .',
        '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
    ]
    # Sort by prefix for stability
    for ns, pref in sorted(ns_to_prefix.items(), key=lambda kv: kv[1].lower()):
        lines.append(f"@prefix {pref}: <{ns}> .")
    return "\n".join(lines) + "\n"



# ── Data Models ────────────────────────────────────────────────────────────

class OntologyElement(BaseModel):
    """Represents an element (class, property, etc.) from the reference ontologies."""
    uri: str
    element_type: str  # 'class', 'object_property', 'data_property', 'annotation_property'
    source_ontology: str  # filename/path of the source ontology
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: Optional[List[str]] = Field(default_factory=list)
    range: Optional[List[str]] = Field(default_factory=list)
    super_classes: Optional[List[str]] = Field(default_factory=list)
    sub_classes: Optional[List[str]] = Field(default_factory=list)

    def get_searchable_text(self) -> str:
        """Get text representation for embedding and search."""
        parts = []

        # Add the local name (URI fragment)
        local_name = self.uri.split('#')[-1].split('/')[-1]
        parts.append(local_name)

        if self.label:
            parts.append(self.label)
        if self.comment:
            parts.append(self.comment)

        # Add type information
        parts.append(f"Type: {self.element_type.replace('_', ' ')}")

        # Add source ontology info
        ontology_name = Path(self.source_ontology).stem
        parts.append(f"From: {ontology_name}")

        # Add domain/range info for properties
        if self.domain:
            domain_names = [d.split('#')[-1].split('/')[-1] for d in self.domain]
            parts.append(f"Domain: {', '.join(domain_names)}")
        if self.range:
            range_names = [r.split('#')[-1].split('/')[-1] for r in self.range]
            parts.append(f"Range: {', '.join(range_names)}")

        return " | ".join(parts)

    def to_turtle_fragment(self) -> str:
        """Convert element to a Turtle fragment for inclusion in generated ontologies."""
        local_name = self.uri.split('#')[-1].split('/')[-1]

        lines = [f":{local_name}"]

        # Add type declaration
        if self.element_type == 'class':
            lines.append("    a owl:Class")
        elif self.element_type == 'object_property':
            lines.append("    a owl:ObjectProperty")
        elif self.element_type == 'data_property':
            lines.append("    a owl:DatatypeProperty")
        elif self.element_type == 'annotation_property':
            lines.append("    a owl:AnnotationProperty")

        # Add label and comment
        if self.label:
            lines.append(f'    rdfs:label "{self.label}"')
        if self.comment:
            lines.append(f'    rdfs:comment "{self.comment}"')

        # Add domain and range for properties
        if self.domain:
            for d in self.domain:
                domain_name = d.split('#')[-1].split('/')[-1]
                lines.append(f'    rdfs:domain :{domain_name}')
        if self.range:
            for r in self.range:
                range_name = r.split('#')[-1].split('/')[-1]
                lines.append(f'    rdfs:range :{range_name}')

        # Add superclass relationships
        if self.super_classes:
            for sc in self.super_classes:
                super_name = sc.split('#')[-1].split('/')[-1]
                lines.append(f'    rdfs:subClassOf :{super_name}')

        return " ;\n".join(lines) + " ."


# ── Multiple Ontologies Processing ────────────────────────────────────────

def extract_elements_from_multiple_ontologies(ontology_files: List[str]) -> List[OntologyElement]:
    """Extract elements from multiple reference ontologies."""
    print(f"[blue]Loading {len(ontology_files)} reference ontologies...[/blue]")

    all_elements = []

    for ontology_file in ontology_files:
        if not Path(ontology_file).exists():
            print(f"[red]Ontology file not found: {ontology_file}[/red]")
            continue

        print(f"[cyan]  Processing: {Path(ontology_file).name}[/cyan]")

        try:
            elements = extract_ontology_elements(ontology_file)
            all_elements.extend(elements)
            print(f"[green] Extracted {len(elements)} elements[/green]")
        except Exception as e:
            print(f"[red]Error processing {ontology_file}: {e}[/red]")
            continue

    print(f"[green]✅ Total extracted {len(all_elements)} elements from {len(ontology_files)} ontologies[/green]")

    # Log elements to CSV
    log_ontology_elements(all_elements)

    return all_elements


def extract_ontology_elements(ontology_file: str) -> List[OntologyElement]:
    """Extract classes, properties and their metadata from a single ontology."""

    g = Graph()
    try:
        g.parse(ontology_file)
    except Exception as e:
        print(f"[red]❌ Error loading ontology {ontology_file}: {e}[/red]")
        raise

    elements = []

    # Extract classes
    classes = list(g.subjects(RDF.type, OWL.Class))
    for cls in classes:
        if isinstance(cls, URIRef):
            element = extract_class_info(g, cls, ontology_file)
            if element:
                elements.append(element)

    # Extract object properties
    object_props = list(g.subjects(RDF.type, OWL.ObjectProperty))
    for prop in object_props:
        if isinstance(prop, URIRef):
            element = extract_property_info(g, prop, 'object_property', ontology_file)
            if element:
                elements.append(element)

    # Extract data properties
    data_props = list(g.subjects(RDF.type, OWL.DatatypeProperty))
    for prop in data_props:
        if isinstance(prop, URIRef):
            element = extract_property_info(g, prop, 'data_property', ontology_file)
            if element:
                elements.append(element)

    # Extract annotation properties
    annotation_props = list(g.subjects(RDF.type, OWL.AnnotationProperty))
    for prop in annotation_props:
        if isinstance(prop, URIRef):
            element = extract_property_info(g, prop, 'annotation_property', ontology_file)
            if element:
                elements.append(element)

    return elements


def extract_class_info(g: Graph, cls: URIRef, source_ontology: str) -> Optional[OntologyElement]:
    """Extract information about a class."""
    try:
        # Get label and comment
        label = get_literal_value(g, cls, RDFS.label)
        comment = get_literal_value(g, cls, RDFS.comment)

        # Get superclasses
        super_classes = [str(sc) for sc in g.objects(cls, RDFS.subClassOf)
                         if isinstance(sc, URIRef)]

        # Get subclasses
        sub_classes = [str(sc) for sc in g.subjects(RDFS.subClassOf, cls)
                       if isinstance(sc, URIRef)]

        return OntologyElement(
            uri=str(cls),
            element_type='class',
            source_ontology=source_ontology,
            label=label,
            comment=comment,
            super_classes=super_classes,
            sub_classes=sub_classes
        )
    except Exception as e:
        print(f"[yellow] Error extracting class {cls}: {e}[/yellow]")
        return None


def extract_property_info(g: Graph, prop: URIRef, prop_type: str, source_ontology: str) -> Optional[OntologyElement]:
    """Extract information about a property."""
    try:
        # Get label and comment
        label = get_literal_value(g, prop, RDFS.label)
        comment = get_literal_value(g, prop, RDFS.comment)

        # Get domain and range
        domain = [str(d) for d in g.objects(prop, RDFS.domain)
                  if isinstance(d, URIRef)]
        range_vals = [str(r) for r in g.objects(prop, RDFS.range)
                      if isinstance(r, URIRef)]

        return OntologyElement(
            uri=str(prop),
            element_type=prop_type,
            source_ontology=source_ontology,
            label=label,
            comment=comment,
            domain=domain,
            range=range_vals
        )
    except Exception as e:
        print(f"[yellow]Error extracting property {prop}: {e}[/yellow]")
        return None


def get_literal_value(g: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
    """Get the first literal value for a given subject-predicate pair."""
    for obj in g.objects(subject, predicate):
        if isinstance(obj, Literal):
            return str(obj)
    return None


# ── Vector Store ────────────────────────────────────────────────────────────

class VectorStore:
    def __init__(self, dim: int):
        self.idx = faiss.IndexFlatIP(dim)
        self.elements = []

    def add(self, vec, element):
        self.idx.add(vec)
        self.elements.append(element)

    def search(self, vec, k=10):
        if self.idx.ntotal == 0:
            return []
        D, I = self.idx.search(vec, k)
        return [(self.elements[i], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1]


_STORE = None


async def embed(text: str) -> np.ndarray:
    r = await openai_client.embeddings.create(model=EMBED_MODEL, input=text)
    # Track embedding tokens (prefer API usage; fallback to tiktoken)
    token_tracker.note_embed(EMBED_MODEL, getattr(r, "usage", None), fallback_text=text)
    return np.asarray(r.data[0].embedding, dtype="float32")[None, :]


async def build_store(ontology_files: List[str]) -> VectorStore:
    """Build vector store from multiple reference ontology elements."""
    global _STORE
    if _STORE:
        return _STORE

    print("[yellow]Building vector store from reference ontologies...[/yellow]")
    _STORE = VectorStore(VECTOR_DIM)

    # Extract elements from all ontologies
    elements = extract_elements_from_multiple_ontologies(ontology_files)

    # Embed each element
    for element in elements:
        searchable_text = element.get_searchable_text()
        embedding = await embed(searchable_text)
        _STORE.add(embedding, element)
    # Build a global prefix block for prompts and headers
    global GLOBAL_PREFIX_BLOCK
    GLOBAL_PREFIX_BLOCK = build_prefix_block(ontology_files)

    print(f"[green]Indexed {len(_STORE.elements)} ontology elements from {len(ontology_files)} ontologies[/green]")
    return _STORE


# ── Retrieval ──────────────────────────────────────────────────────────────

async def retrieve_relevant_elements(cq: str, k: int = 20) -> Tuple[List[OntologyElement], Dict[str, int]]:
    """Retrieve relevant ontology elements for a competency question."""
    if not _STORE:
        raise ValueError("Vector store not initialized. Call build_store first.")

    qvec = await embed(cq)
    results = _STORE.search(qvec, k)

    # Extract elements and sort by relevance
    relevant_elements = [element for element, score in results]

    # Count elements by source ontology for statistics
    source_counts = {}
    for element in relevant_elements:
        ontology_name = Path(element.source_ontology).stem
        source_counts[ontology_name] = source_counts.get(ontology_name, 0) + 1

    print(f"[cyan]Retrieved {len(relevant_elements)} relevant elements from {len(source_counts)} ontologies[/cyan]")
    for source, count in source_counts.items():
        print(f"[dim]  - {source}: {count} elements[/dim]")

    return relevant_elements, source_counts


# ── LLM Orchestration ──────────────────────────────────────────────────────

LLM_MODEL = "gpt-4o"

SYS_PROMPT = ("You are an ontology engineer. Reuse elements from the provided core ontology when possible. "
              "If you must create new elements, append '# GENERATED' as a comment. "
              "Return only Turtle/TTL syntax inside one markdown code block.")

USER_TMPL = """

You are a helpful assistant designed to generate ontologies. You receive a Competency Question (CQ) and an Ontology Story (OS). \n
Based on CQ, which is a requirement for the ontology, and OS, which tells you what the context of the ontology is, your task is generating one ontology (O). The goal is to generate O that models the CQ properly. This means there is a way to write a SPARQL query to extract the answer to this CQ in O.  \n
Reuse the relevant ontology elements provided in the RELEVANT ONTOLOGY ELEMENTS section below whenever possible. These elements come from multiple reference ontologies. Only create new elements if absolutely necessary. In any case, add labels and comments. \n
Use the following prefixes: \n
{prefix_block}\n

Don't put any A-Box (instances) in the ontology and just generate the OWL file using Turtle syntax. Include the entities mentioned in the CQ. Remember to use restrictions when the CQ implies it. The output should be self-contained without any errors. Outside of the code box don't put any comment.\n
Instructions:\n
1. Analyze the CQ to understand what concepts and relationships are needed
2. Map the required concepts to classes/properties from the RELEVANT ONTOLOGY ELEMENTS above
3. Prefer elements with higher semantic similarity to the CQ concepts
4. If something required is missing, create it under the : namespace and mark with '# GENERATED'
5. Output only the TBox ontology in Turtle syntax (no instances/ABox)
6. Include restrictions when the CQ implies them (e.g., cardinality, value restrictions)
7. Ensure the output is syntactically correct and self-contained
Competency Question: "{cq}" \n
RELEVANT ONTOLOGY ELEMENTS (from {num_ontologies} reference ontologies):
{relevant_elements}

"""



async def ask_llm(prompt: str) -> str:
    """Send prompt to LLM and return response."""
    try:
        response = await openai_client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": prompt}
            ]
        )
        # Track chat tokens
        token_tracker.note_chat(LLM_MODEL, getattr(response, "usage", None))
        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"[red]LLM Error: {e}[/red]")
        raise


async def generate_ontology_fragment(cq: str, relevant_elements: List[OntologyElement],
                                     source_counts: Dict[str, int]) -> str:
    """Generate ontology fragment for a competency question using relevant elements."""

    # Format relevant elements for the prompt, grouped by source ontology
    elements_by_source = {}
    for element in relevant_elements:
        source_name = Path(element.source_ontology).stem
        if source_name not in elements_by_source:
            elements_by_source[source_name] = []
        elements_by_source[source_name].append(element)

    # Create formatted text showing elements organized by source
    relevant_elements_text = []
    for source_name, elements in elements_by_source.items():
        relevant_elements_text.append(f"\n## From {source_name} ontology:")
        for element in elements:
            relevant_elements_text.append(f"- {element.get_searchable_text()}")

    relevant_elements_formatted = "\n".join(relevant_elements_text)

    # Create the prompt
    prompt = USER_TMPL.format(
        num_ontologies=len(source_counts),
        relevant_elements=relevant_elements_formatted,
        cq=cq,
        prefix_block=(GLOBAL_PREFIX_BLOCK or
                      '@prefix : <http://www.example.org/ontology#> .\n'
                      '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n'
                      '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n'
                      '@prefix owl: <http://www.w3.org/2002/07/owl#> .\n'
                      '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n')
    )

    # Get LLM response
    response = await ask_llm(prompt)

    # Extract Turtle code block
    ttl_match = re.search(r"```(?:ttl|turtle)?\n(.*?)```", response, re.DOTALL)
    if ttl_match:
        return ttl_match.group(1).strip()
    else:
        # If no code block found, return the whole response
        return response.strip()


# ── Validation and Combination ─────────────────────────────────────────────

def validate_turtle(ttl_content: str) -> bool:
    """Validate Turtle syntax using rdflib."""
    try:
        g = Graph()
        g.parse(data=ttl_content, format="turtle")
        return True
    except Exception as e:
        print(f"[yellow]Turtle validation failed: {e}[/yellow]")
        return False


def combine_ontology_fragments(
    fragments: List[Tuple[str, str]],
    ontology_files: List[str]
) -> str:
    """Combine individual fragments into one TTL using prefixes from the sources."""
    prefix_block = GLOBAL_PREFIX_BLOCK or build_prefix_block(ontology_files)

    header = (
        f"{prefix_block}\n"
        f"# Combined ontology generated from competency questions\n"
        f"# Reference ontologies: {', '.join(Path(f).name for f in ontology_files)}\n"
        f"# Generated on: {datetime.now().isoformat()}\n"
        f"# Total fragments: {len(fragments)}\n\n"
        "<http://www.example.org/ontology> a owl:Ontology .\n\n"
    )

    body_parts: List[str] = []
    for i, (cq, fragment) in enumerate(fragments, 1):
        body_parts.append(
            f"# ── Fragment {i}: {cq[:80]}{'...' if len(cq) > 80 else ''} ──\n"
        )
        body_parts.append(remove_prefixes_from_fragment(fragment).strip())
        body_parts.append("")  # blank line between fragments

    return header + "\n".join(body_parts)



def remove_prefixes_from_fragment(fragment: str) -> str:
    """Remove prefix declarations from a fragment since they'll be in the header."""
    lines = fragment.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith('@prefix') or stripped.startswith('PREFIX'):
            continue  # Skip prefix declarations
        cleaned_lines.append(line)

    return '\n'.join(cleaned_lines)


# ── CSV Logging Functions ──────────────────────────────────────────────────

def log_ontology_elements(elements: List[OntologyElement]) -> None:
    """Log ontology elements to CSV."""
    ELEMENTS_CSV.parent.mkdir(exist_ok=True)

    with ELEMENTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "uri", "element_type", "source_ontology", "label", "comment",
            "domain", "range", "super_classes", "sub_classes"
        ])

        timestamp = datetime.now().isoformat()
        for element in elements:
            writer.writerow([
                timestamp,
                element.uri,
                element.element_type,
                Path(element.source_ontology).name,
                element.label or "",
                element.comment or "",
                ";".join(element.domain or []),
                ";".join(element.range or []),
                ";".join(element.super_classes or []),
                ";".join(element.sub_classes or [])
            ])


def log_processing_result(cq_index: int, cq: str, success: bool,
                          elements_used: int, source_ontologies: Dict[str, int],
                          processing_time: float, error: str = None) -> None:
    """Log CQ processing results."""
    PROCESSING_LOG_CSV.parent.mkdir(exist_ok=True)
    newfile = not PROCESSING_LOG_CSV.exists()

    with PROCESSING_LOG_CSV.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow([
                "timestamp", "cq_index", "cq", "success", "elements_used",
                "source_ontologies_used", "processing_time_seconds", "error"
            ])

        # Format source ontologies as "onto1:count1;onto2:count2"
        sources_str = ";".join([f"{Path(k).stem}:{v}" for k, v in source_ontologies.items()])

        writer.writerow([
            datetime.now().isoformat(),
            cq_index,
            cq[:200] + "..." if len(cq) > 200 else cq,
            success,
            elements_used,
            sources_str,
            f"{processing_time:.2f}",
            error or ""
        ])


def log_final_results(total_cqs: int, successful: int, failed: int,
                      total_time: float, output_file: str) -> None:
    """Log final processing results."""
    RESULTS_CSV.parent.mkdir(exist_ok=True)

    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "total_cqs", "successful", "failed", "success_rate_percent",
            "total_time_minutes", "avg_time_per_cq_seconds", "output_file"
        ])

        success_rate = (successful / total_cqs * 100) if total_cqs > 0 else 0
        avg_time = (total_time / total_cqs) if total_cqs > 0 else 0

        writer.writerow([
            datetime.now().isoformat(),
            total_cqs,
            successful,
            failed,
            f"{success_rate:.1f}",
            f"{total_time / 60:.2f}",
            f"{avg_time:.2f}",
            output_file
        ])


# ── Main Processing Function ───────────────────────────────────────────────

async def process_cq_dataset(csv_file: str, ontology_files: List[str],
                             limit: int = None) -> None:
    """Process all CQs from a CSV dataset and generate combined ontology."""

    print(f"[bold blue]Processing CQ dataset: {csv_file}[/bold blue]")
    print(f"[bold blue]Using {len(ontology_files)} reference ontologies:[/bold blue]")
    for i, onto_file in enumerate(ontology_files, 1):
        print(f"[cyan]  {i}. {Path(onto_file).name}[/cyan]")

    # Validate inputs
    if not Path(csv_file).exists():
        print(f"[red]CSV file not found: {csv_file}[/red]")
        return

    missing_ontologies = [f for f in ontology_files if not Path(f).exists()]
    if missing_ontologies:
        print(f"[red]Ontology files not found: {missing_ontologies}[/red]")
        return

    # Load CSV
    try:
        df = pd.read_csv(csv_file)
        print(f"[green]Loaded {len(df)} rows from CSV[/green]")
    except Exception as e:
        print(f"[red]Error reading CSV: {e}[/red]")
        return

    # Validate CQ column
    if 'CQ' not in df.columns:
        print(f"[red]'CQ' column not found in CSV. Available columns: {list(df.columns)}[/red]")
        return

    # Apply limit if specified
    if limit:
        df = df.head(limit)
        print(f"[blue]Processing limited to first {len(df)} rows[/blue]")

    # Build vector store from all ontologies
    await build_store(ontology_files)

    # Process each CQ
    fragments = []
    successful = 0
    failed = 0
    start_time = time.time()

    pbar = tqdm(total=len(df), desc="Processing CQs")

    for idx, row in df.iterrows():
        cq = str(row['CQ']).strip()
        before = token_tracker.snapshot()

        if not cq or cq.lower() in ['nan', 'none', '']:
            print(f"[yellow]Row {idx}: Empty CQ, skipping[/yellow]")
            log_processing_result(idx, cq, False, 0, {}, 0, "Empty CQ")
            failed += 1
            pbar.update(1)
            continue

        pbar.set_description(f"Processing CQ {idx}")

        cq_start_time = time.time()
        try:
            # Retrieve relevant elements from all ontologies
            relevant_elements, source_counts = await retrieve_relevant_elements(cq)

            # Generate ontology fragment
            fragment = await generate_ontology_fragment(cq, relevant_elements, source_counts)

            # Validate generated fragment
            if validate_turtle(fragment):
                fragments.append((cq, fragment))

                # Save individual fragment
                individual_file = INDIVIDUAL_ONTOLOGIES_DIR / f"cq_{idx}.ttl"
                individual_file.write_text(fragment, encoding="utf-8")

                processing_time = time.time() - cq_start_time
                log_processing_result(idx, cq, True, len(relevant_elements), source_counts, processing_time)
                successful += 1

                print(f"[green]CQ {idx} processed successfully[/green]")
            else:
                print(f"[red]CQ {idx}: Generated invalid Turtle[/red]")
                log_processing_result(idx, cq, False, len(relevant_elements), source_counts,
                                      time.time() - cq_start_time, "Invalid Turtle")
                failed += 1
                after = token_tracker.snapshot()
                deltas = token_tracker.diff(before, after)
                log_cost_row(idx, cq, deltas)

        except Exception as e:
            processing_time = time.time() - cq_start_time
            error_msg = str(e)
            print(f"[red]CQ {idx} failed: {error_msg}[/red]")
            log_processing_result(idx, cq, False, 0, {}, processing_time, error_msg)
            failed += 1
            after = token_tracker.snapshot()
            deltas = token_tracker.diff(before, after)
            log_cost_row(idx, cq, deltas)

        pbar.update(1)

        # Small delay to avoid overwhelming the API
        await asyncio.sleep(0.1)

    pbar.close()

    # Combine all fragments
    if fragments:
        print(f"[yellow]Combining {len(fragments)} ontology fragments...[/yellow]")
        combined_ontology = combine_ontology_fragments(fragments, ontology_files)

        # Save combined ontology
        COMBINED_ONTOLOGY_FILE.write_text(combined_ontology, encoding="utf-8")
        print(f"[green]Combined ontology saved to: {COMBINED_ONTOLOGY_FILE}[/green]")

        # Validate combined ontology
        if validate_turtle(combined_ontology):
            print("[green]Combined ontology is syntactically valid[/green]")
        else:
            print("[yellow]Combined ontology has syntax issues[/yellow]")

    # Final statistics
    total_time = time.time() - start_time
    log_final_results(len(df), successful, failed, total_time, str(COMBINED_ONTOLOGY_FILE))

    print(f"\n[bold green]Processing Complete![/bold green]")
    print(f"Total CQs: {len(df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(successful / len(df) * 100):.1f}%")
    print(f"[bold green]💲 Estimated total API cost (USD): {token_tracker.totals_cost_usd():.6f}[/bold green]")

    #print(


import argparse


def ensure_api_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("[bold red]OPENAI_API_KEY is not set[/bold red]")
        sys.exit(1)


async def run_single_cq(cq: str, ontology_files: List[str]) -> None:
    ensure_api_key()

    # Build the vector store once
    await build_store(ontology_files)

    # Retrieve, generate, validate
    relevant_elements, source_counts = await retrieve_relevant_elements(cq)
    fragment = await generate_ontology_fragment(cq, relevant_elements, source_counts)

    if validate_turtle(fragment):
        INDIVIDUAL_ONTOLOGIES_DIR.mkdir(exist_ok=True)
        out_path = INDIVIDUAL_ONTOLOGIES_DIR / "single_cq.ttl"
        out_path.write_text(fragment, encoding="utf-8")
        print(f"[green]Ontology fragment saved to: {out_path}[/green]")
        print("\n[bold green]Ontology fragment[/bold green]\n")
        print(fragment)
    else:
        print("[bold yellow]Generated fragment failed Turtle validation[/bold yellow]")
        print(fragment)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core ontology RAG for CQ → TTL")
    sub = p.add_subparsers(dest="cmd")

    # Single CQ
    p_single = sub.add_parser("cq", help="Generate ontology for a single CQ")
    p_single.add_argument("--cq", required=True, help="Competency Question text")
    p_single.add_argument("--onto", nargs="+", help="One or more reference ontology files (.ttl/.owl)")
    # Single CQ
    p_single.add_argument("--onto-dir", nargs="+", help="One or more directories to scan for .ttl/.owl")

    # Batch CSV

    # Batch CSV
    p_batch = sub.add_parser("csv", help="Process a CSV with a CQ column")
    p_batch.add_argument("--file", required=True, help="CSV file with column 'CQ'")
    p_batch.add_argument("--onto", nargs="+", help="One or more reference ontology files (.ttl/.owl)")
    p_batch.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    p_batch.add_argument("--onto-dir", nargs="+", help="One or more directories to scan for .ttl/.owl")


    return p.parse_args()

from collections import OrderedDict

def expand_onto_args(files: Optional[List[str]], dirs: Optional[List[str]]) -> List[str]:
    """Build a flat list of ontology files from --onto file list and/or --onto-dir folders."""
    candidates: List[str] = []
    if files:
        candidates.extend(files)
    if dirs:
        for d in dirs:
            p = Path(d)
            if not p.exists() or not p.is_dir():
                print(f"[yellow]Not a directory (skipped): {d}[/yellow]")
                continue
            candidates.extend(str(f) for f in p.rglob("*.ttl"))
            candidates.extend(str(f) for f in p.rglob("*.owl"))

    # de-duplicate while preserving order
    seen = set()
    result = []
    for f in candidates:
        if f not in seen:
            result.append(f); seen.add(f)

    if not result:
        print("[red]No ontology files found from --onto/--onto-dir[/red]")
        sys.exit(1)
    return result


if __name__ == "__main__":
    args = parse_args()
    if not args.cmd:
        print("[bold red]Usage:[/bold red] python rag_ontology.py cq --cq \"...\" --onto core.ttl more.ttl")
        print("         python rag_ontology.py csv --file dataset.csv --onto core.ttl more.ttl [--limit 50]")
        sys.exit(1)

    if args.cmd == "cq":
        ontofiles = expand_onto_args(args.onto, args.onto_dir)
        asyncio.run(run_single_cq(args.cq, ontofiles))
    elif args.cmd == "csv":
        ontofiles = expand_onto_args(args.onto, args.onto_dir)
        asyncio.run(process_cq_dataset(args.file, ontofiles, args.limit))


#>python rag-ontoextend.py csv --file .\cqs.csv --onto-dir .\coreontologies --limit 3

