from __future__ import annotations
import argparse
import asyncio
import csv
import os
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import faiss
import numpy as np
import openai
import pandas as pd
from pydantic import BaseModel, Field
from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef
from rich import print
from tiktoken import get_encoding

# ==========================
# Configuration & Constants
# ==========================

def _p(env: str, default):
    v = os.getenv(env)
    return float(v) if v else default

PRICES_PER_1K: Dict[str, Dict[str, Optional[float]]] = {
    "gpt-4o": {  # Chat/completions models (input vs output)
        "input": _p("PRICE_GPT_4O_INPUT_PER_1K", None),
        "output": _p("PRICE_GPT_4O_OUTPUT_PER_1K", None),
    },
    "text-embedding-3-small": {  # Embeddings (input only)
        "input": _p("PRICE_TEXT_EMBEDDING_3_SMALL_PER_1K", 0.00002),
    },
    "text-embedding-3-large": {
        "input": _p("PRICE_TEXT_EMBEDDING_3_LARGE_PER_1K", None),
    },
    "text-embedding-ada-002": {
        "input": _p("PRICE_TEXT_EMBEDDING_ADA_002_PER_1K", None),
    },
    "Qwen3-Embedding-8B": {  # for OpenAI-compatible endpoints if we use one...
        "input": _p("PRICE_QWEN3_EMBEDDING_8B_PER_1K", None),
    },
}

ENC = get_encoding("cl100k_base")

STANDARD_PREFIXES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}

LLM_MODEL = "gpt-4o"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"

SYS_PROMPT = (
    "You are an ontology engineer. Reuse elements from the provided core ontology when possible. "
    "If you must create new elements, append '# GENERATED' as a comment. "
    "Return only Turtle/TTL syntax inside one markdown code block."
)

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


# ==========================
# Paths & IO
# ==========================

@dataclass
class Paths:
    cache_dir: Path = Path.home() / ".core_ontology_rag_cache"

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / "individual_ontologies").mkdir(parents=True, exist_ok=True)

    @property
    def results_csv(self) -> Path:
        return self.cache_dir / "generation_runs.csv"

    @property
    def elements_csv(self) -> Path:
        return self.cache_dir / "ontology_elements.csv"

    @property
    def processing_log_csv(self) -> Path:
        return self.cache_dir / "cq_processing_log.csv"

    @property
    def token_cost_csv(self) -> Path:
        return self.cache_dir / "token_cost_log.csv"

    @property
    def combined_ttl(self) -> Path:
        return self.cache_dir / "combined_ontology.ttl"

    @property
    def individual_dir(self) -> Path:
        return self.cache_dir / "individual_ontologies"

    @property
    def retrieval_log_csv(self) -> Path:
        return self.cache_dir / "retrieval_log.csv"


# ==========================
# Cost Tracking & Logging
# ==========================

class TokenCostTracker:
    def __init__(self) -> None:
        self.totals: Dict[str, Dict[str, int]] = {}

    def _ensure(self, model: str) -> None:
        if model not in self.totals:
            self.totals[model] = {"prompt": 0, "completion": 0, "embedding": 0}

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        return deepcopy(self.totals)

    def diff(self, before: Dict[str, Dict[str, int]], after: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for m in set(before.keys()) | set(after.keys()):
            out[m] = {
                "prompt": after.get(m, {}).get("prompt", 0) - before.get(m, {}).get("prompt", 0),
                "completion": after.get(m, {}).get("completion", 0) - before.get(m, {}).get("completion", 0),
                "embedding": after.get(m, {}).get("embedding", 0) - before.get(m, {}).get("embedding", 0),
            }
        return out

    def note_chat(self, model: str, usage: Any) -> None:
        if not usage:
            return
        self._ensure(model)
        self.totals[model]["prompt"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        self.totals[model]["completion"] += int(getattr(usage, "completion_tokens", 0) or 0)

    def note_embed(self, model: str, usage: Any, fallback_text: Optional[str] = None) -> None:
        self._ensure(model)
        if usage and getattr(usage, "total_tokens", None) is not None:
            self.totals[model]["embedding"] += int(usage.total_tokens)
        elif fallback_text is not None:
            self.totals[model]["embedding"] += len(ENC.encode(fallback_text))

    def model_cost_usd(self, model: str, counts: Dict[str, int]) -> float:
        p = PRICES_PER_1K.get(model, {})
        cin = (counts.get("prompt", 0) + counts.get("embedding", 0)) / 1000.0 * (p.get("input") or 0.0)
        cout = counts.get("completion", 0) / 1000.0 * (p.get("output") or 0.0)
        return cin + cout

    def totals_cost_usd(self) -> float:
        return sum(self.model_cost_usd(m, c) for m, c in self.totals.items())


class RunLogger:
    def __init__(self, paths: Paths, tracker: TokenCostTracker) -> None:
        self.paths = paths
        self.tracker = tracker

    def log_elements(self, elements: List["OntologyElement"]) -> None:
        self.paths.elements_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.elements_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "uri", "element_type", "source_ontology", "label", "comment",
                "domain", "range", "super_classes", "sub_classes",
            ])
            ts = datetime.now().isoformat()
            for e in elements:
                w.writerow([
                    ts,
                    e.uri,
                    e.element_type,
                    Path(e.source_ontology).name,
                    e.label or "",
                    e.comment or "",
                    ";".join(e.domain or []),
                    ";".join(e.range or []),
                    ";".join(e.super_classes or []),
                    ";".join(e.sub_classes or []),
                ])

    def log_processing_result(
        self,
        cq_index: int,
        cq: str,
        success: bool,
        elements_used: int,
        source_ontologies: Dict[str, int],
        processing_time: float,
        error: Optional[str] = None,
    ) -> None:
        self.paths.processing_log_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.processing_log_csv.exists()
        with self.paths.processing_log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp", "cq_index", "cq", "success", "elements_used",
                    "source_ontologies_used", "processing_time_seconds", "error",
                ])
            sources_str = ";".join([f"{Path(k).stem}:{v}" for k, v in source_ontologies.items()])
            w.writerow([
                datetime.now().isoformat(),
                cq_index,
                cq[:200] + "..." if len(cq) > 200 else cq,
                success,
                elements_used,
                sources_str,
                f"{processing_time:.2f}",
                error or "",
            ])

    def log_final_results(
        self,
        total_cqs: int,
        successful: int,
        failed: int,
        total_time: float,
        output_file: str,
    ) -> None:
        self.paths.results_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.results_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp", "total_cqs", "successful", "failed", "success_rate_percent",
                "total_time_minutes", "avg_time_per_cq_seconds", "output_file",
            ])
            success_rate = (successful / total_cqs * 100) if total_cqs > 0 else 0
            avg_time = (total_time / total_cqs) if total_time > 0 and total_cqs > 0 else 0
            w.writerow([
                datetime.now().isoformat(),
                total_cqs,
                successful,
                failed,
                f"{success_rate:.1f}",
                f"{total_time / 60:.2f}",
                f"{avg_time:.2f}",
                output_file,
            ])

    def log_cost_row(self, cq_index: int, cq: str, deltas: Dict[str, Dict[str, int]]) -> None:
        self.paths.token_cost_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.token_cost_csv.exists()
        with self.paths.token_cost_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp", "cq_index", "cq", "model",
                    "prompt_tokens", "completion_tokens", "embedding_tokens", "cost_usd",
                ])
            for model, counts in deltas.items():
                cost = self.tracker.model_cost_usd(model, counts)
                w.writerow([
                    datetime.now().isoformat(),
                    cq_index,
                    cq[:200] + ("..." if len(cq) > 200 else ""),
                    model,
                    counts.get("prompt", 0),
                    counts.get("completion", 0),
                    counts.get("embedding", 0),
                    f"{cost:.6f}",
                ])

    def log_retrieval(
        self,
        cq_index: int,
        cq: str,
        model: str,
        threshold: float,
        topN: int,
        pool_size: int,
        rows: List[Dict[str, Any]],
    ) -> None:
        """
        rows: list of dicts with keys:
          uri, element_type, label, source_ontology, score, selected (bool), parent_included (bool)
        """
        self.paths.retrieval_log_csv.parent.mkdir(parents=True, exist_ok=True)
        newfile = not self.paths.retrieval_log_csv.exists()
        with self.paths.retrieval_log_csv.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if newfile:
                w.writerow([
                    "timestamp","cq_index","cq","model","threshold","topN","pool_size",
                    "uri","element_type","label","source_ontology","score","selected","parent_included",
                ])
            ts = datetime.now().isoformat()
            cq_short = cq[:200] + ("..." if len(cq) > 200 else "")
            for r in rows:
                w.writerow([
                    ts, cq_index, cq_short, model, f"{threshold:.6f}", topN, pool_size,
                    r.get("uri",""), r.get("element_type",""), r.get("label",""),
                    Path(r.get("source_ontology","")).name,
                    f"{float(r.get('score',0.0)):.6f}",
                    bool(r.get("selected", False)),
                    bool(r.get("parent_included", False)),
                ])


# ==========================
# Ontology Elements
# ==========================

class OntologyElement(BaseModel):
    uri: str
    element_type: str  # 'class', 'object_property', 'data_property', 'annotation_property'
    source_ontology: str
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: Optional[List[str]] = Field(default_factory=list)
    range: Optional[List[str]] = Field(default_factory=list)
    super_classes: Optional[List[str]] = Field(default_factory=list)
    sub_classes: Optional[List[str]] = Field(default_factory=list)

    def get_searchable_text(self) -> str:
        parts: List[str] = []
        local_name = self.uri.split('#')[-1].split('/')[-1]
        parts.append(local_name)
        if self.label:
            parts.append(self.label)
        if self.comment:
            parts.append(self.comment)
        parts.append(f"Type: {self.element_type.replace('_', ' ')}")
        ontology_name = Path(self.source_ontology).stem
        parts.append(f"From: {ontology_name}")
        if self.domain:
            domain_names = [d.split('#')[-1].split('/')[-1] for d in self.domain]
            parts.append(f"Domain: {', '.join(domain_names)}")
        if self.range:
            range_names = [r.split('#')[-1].split('/')[-1] for r in self.range]
            parts.append(f"Range: {', '.join(range_names)}")
        return " | ".join(parts)

    def get_embedding_text(self, include_comment: bool = True, format: str = "pipe") -> str:
        items: List[str] = []
        items.append(self.uri)
        items.append(self.label or "")
        if include_comment:
            items.append(self.comment or "")
        items.append(self.element_type.replace("_", " "))
        items.append(",".join(self.domain or []))
        items.append(",".join(self.range or []))
        if format == "newline":
            return "\n".join(items).strip()
        return " | ".join(items).strip()


# ==========================
# Prefix Management
# ==========================

class PrefixManager:
    def __init__(self) -> None:
        pass

    @staticmethod
    def sanitize_prefix_name(raw: Optional[str]) -> str:
        s = (raw or "").strip()
        s = re.sub(r"[^A-Za-z0-9_\-]", "_", s)
        if not s or not re.match(r"^[A-Za-z_]", s):
            s = f"ns_{s}" if s else "ns"
        s = re.sub(r"_+", "_", s)
        return s

    def collect_prefixes_from_files(self, ontology_files: List[str]) -> Dict[str, str]:
        ns_to_prefix: Dict[str, str] = {}
        used: set[str] = set(STANDARD_PREFIXES.keys())
        for f in ontology_files:
            g = Graph()
            try:
                g.parse(f)
            except Exception:
                continue
            for pref, ns in g.namespaces():
                ns = str(ns)
                if not ns or (pref and str(pref).lower() == "xml"):
                    continue
                if ns in STANDARD_PREFIXES.values():
                    continue
                p = self.sanitize_prefix_name(pref)
                while p in used:
                    m = re.search(r"(\d+)$", p)
                    p = re.sub(r"\d+$", str(int(m.group(1)) + 1), p) if m else f"{p}2"
                ns_to_prefix[ns] = p
                used.add(p)
        return ns_to_prefix

    def build_prefix_block(self, ontology_files: List[str]) -> str:
        ns_to_prefix = self.collect_prefixes_from_files(ontology_files)
        lines = [
            '@prefix : <http://www.example.org/ontology#> .',
            '@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .',
            '@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .',
            '@prefix owl: <http://www.w3.org/2002/07/owl#> .',
            '@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .',
        ]
        for ns, pref in sorted(ns_to_prefix.items(), key=lambda kv: kv[1].lower()):
            lines.append(f"@prefix {pref}: <{ns}> .")
        return "\n".join(lines) + "\n"


# ==========================
# Extraction
# ==========================

class OntologyExtractor:
    def __init__(self, logger: RunLogger) -> None:
        self.logger = logger

    def extract_from_files(self, ontology_files: List[str]) -> List[OntologyElement]:
        print(f"[blue]Loading {len(ontology_files)} reference ontologies...[/blue]")
        all_elements: List[OntologyElement] = []
        for ontology_file in ontology_files:
            p = Path(ontology_file)
            if not p.exists():
                print(f"[red]Ontology file not found: {ontology_file}[/red]")
                continue
            print(f"[cyan]  Processing: {p.name}[/cyan]")
            try:
                elements = self._extract_single(str(p))
                all_elements.extend(elements)
                print(f"[green] Extracted {len(elements)} elements[/green]")
            except Exception as e:
                print(f"[red]Error processing {ontology_file}: {e}[/red]")
                continue
        print(f"[green] Total extracted {len(all_elements)} elements from {len(ontology_files)} ontologies[/green]")
        self.logger.log_elements(all_elements)
        return all_elements

    def _extract_single(self, ontology_file: str) -> List[OntologyElement]:
        g = Graph()
        g.parse(ontology_file)
        elements: List[OntologyElement] = []
        for cls in list(g.subjects(RDF.type, OWL.Class)):
            if isinstance(cls, URIRef):
                el = self._extract_class_info(g, cls, ontology_file)
                if el:
                    elements.append(el)
        for prop in list(g.subjects(RDF.type, OWL.ObjectProperty)):
            if isinstance(prop, URIRef):
                el = self._extract_property_info(g, prop, 'object_property', ontology_file)
                if el:
                    elements.append(el)
        for prop in list(g.subjects(RDF.type, OWL.DatatypeProperty)):
            if isinstance(prop, URIRef):
                el = self._extract_property_info(g, prop, 'data_property', ontology_file)
                if el:
                    elements.append(el)
        for prop in list(g.subjects(RDF.type, OWL.AnnotationProperty)):
            if isinstance(prop, URIRef):
                el = self._extract_property_info(g, prop, 'annotation_property', ontology_file)
                if el:
                    elements.append(el)
        return elements

    @staticmethod
    def _extract_class_info(g: Graph, cls: URIRef, source_ontology: str) -> Optional[OntologyElement]:
        try:
            label = OntologyExtractor._get_literal_value(g, cls, RDFS.label)
            comment = OntologyExtractor._get_literal_value(g, cls, RDFS.comment)
            super_classes = [str(sc) for sc in g.objects(cls, RDFS.subClassOf) if isinstance(sc, URIRef)]
            sub_classes = [str(sc) for sc in g.subjects(RDFS.subClassOf, cls) if isinstance(sc, URIRef)]
            return OntologyElement(
                uri=str(cls),
                element_type='class',
                source_ontology=source_ontology,
                label=label,
                comment=comment,
                super_classes=super_classes,
                sub_classes=sub_classes,
            )
        except Exception as e:
            print(f"[yellow] Error extracting class {cls}: {e}[/yellow]")
            return None

    @staticmethod
    def _extract_property_info(g: Graph, prop: URIRef, prop_type: str, source_ontology: str) -> Optional[OntologyElement]:
        try:
            label = OntologyExtractor._get_literal_value(g, prop, RDFS.label)
            comment = OntologyExtractor._get_literal_value(g, prop, RDFS.comment)
            domain = [str(d) for d in g.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
            range_vals = [str(r) for r in g.objects(prop, RDFS.range) if isinstance(r, URIRef)]
            return OntologyElement(
                uri=str(prop),
                element_type=prop_type,
                source_ontology=source_ontology,
                label=label,
                comment=comment,
                domain=domain,
                range=range_vals,
            )
        except Exception as e:
            print(f"[yellow]Error extracting property {prop}: {e}[/yellow]")
            return None

    @staticmethod
    def _get_literal_value(g: Graph, subject: URIRef, predicate: URIRef) -> Optional[str]:
        for obj in g.objects(subject, predicate):
            if isinstance(obj, Literal):
                return str(obj)
        return None


# ==========================
# Embedding & Vector Store
# ==========================

class OpenAIEmbedder:
    def __init__(self, model: str, tracker: TokenCostTracker, normalize: bool = True, batch_size: int = 64) -> None:
        self.model = model
        self.tracker = tracker
        self.normalize = normalize
        self.batch_size = max(1, batch_size)
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = openai.AsyncOpenAI(base_url=base_url) if base_url else openai.AsyncOpenAI()

    def _l2norm(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / n

    async def embed_one(self, text: str) -> np.ndarray:
        r = await self.client.embeddings.create(model=self.model, input=text)
        self.tracker.note_embed(self.model, getattr(r, "usage", None), fallback_text=text)
        vec = np.asarray(r.data[0].embedding, dtype="float32")[None, :]
        return self._l2norm(vec)

    async def embed_many(self, texts: List[str]) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            r = await self.client.embeddings.create(model=self.model, input=batch)
            self.tracker.note_embed(self.model, getattr(r, "usage", None))
            vecs = np.asarray([d.embedding for d in r.data], dtype="float32")
            vecs = self._l2norm(vecs)
            out.extend([v[None, :] for v in vecs])
        return out


class FaissVectorStore:
    def __init__(self, inner_product: bool = True) -> None:
        self.idx: Optional[faiss.IndexFlat] = None
        self.inner_product = inner_product
        self.dim: Optional[int] = None
        self.elements: List[OntologyElement] = []
        self.uri_to_element: Dict[str, OntologyElement] = {}

    def _ensure_index(self, dim: int) -> None:
        if self.idx is None:
            self.dim = dim
            self.idx = faiss.IndexFlatIP(dim) if self.inner_product else faiss.IndexFlatL2(dim)

    def add(self, vec: np.ndarray, element: OntologyElement) -> None:
        self._ensure_index(vec.shape[1])
        self.idx.add(vec)  # type: ignore
        self.elements.append(element)
        self.uri_to_element.setdefault(element.uri, element)

    def search(self, vec: np.ndarray, k: int = 10) -> List[Tuple[OntologyElement, float]]:
        if self.idx is None or self.idx.ntotal == 0:
            return []
        D, I = self.idx.search(vec, k)  # type: ignore
        pairs: List[Tuple[OntologyElement, float]] = []
        for j, i in enumerate(I[0]):
            if i == -1:
                continue
            el = self.elements[int(i)]
            pairs.append((el, float(D[0][j])))
        return pairs


# ==========================
# Prompting & LLM
# ==========================

class PromptBuilder:
    def __init__(self, prefix_block: str) -> None:
        self.prefix_block = prefix_block

    @staticmethod
    def _format_relevant_elements(elements: List[OntologyElement]) -> Tuple[str, int]:
        elements_by_source: Dict[str, List[OntologyElement]] = {}
        for e in elements:
            src = Path(e.source_ontology).stem
            elements_by_source.setdefault(src, []).append(e)
        parts: List[str] = []
        for source, els in elements_by_source.items():
            parts.append(f"\n## From {source} ontology:")
            for e in els:
                parts.append(f"- {e.get_searchable_text()}")
        return "\n".join(parts), len(elements_by_source)

    def build(self, cq: str, elements: List[OntologyElement]) -> str:
        relevant_text, num_ontologies = self._format_relevant_elements(elements)
        return USER_TMPL.format(
            num_ontologies=num_ontologies,
            relevant_elements=relevant_text,
            cq=cq,
            prefix_block=self.prefix_block,
        )


class LLMService:
    def __init__(self, model: str, tracker: TokenCostTracker) -> None:
        self.model = model
        self.tracker = tracker
        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = openai.AsyncOpenAI(base_url=base_url) if base_url else openai.AsyncOpenAI()

    async def complete(self, prompt: str) -> str:
        try:
            resp = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": prompt}],
            )
            self.tracker.note_chat(self.model, getattr(resp, "usage", None))
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[red]LLM Error: {e}[/red]")
            raise


# ==========================
# Validation & Combination
# ==========================

class TurtleValidator:
    @staticmethod
    def validate_ttl(ttl_content: str) -> bool:
        try:
            g = Graph()
            g.parse(data=ttl_content, format="turtle")
            return True
        except Exception as e:
            print(f"[yellow]Turtle validation failed: {e}[/yellow]")
            return False


class OntologyCombiner:
    @staticmethod
    def remove_prefixes_from_fragment(fragment: str) -> str:
        lines = fragment.split('\n')
        cleaned: List[str] = []
        for line in lines:
            s = line.strip()
            if s.startswith('@prefix') or s.startswith('PREFIX'):
                continue
            cleaned.append(line)
        return '\n'.join(cleaned)

    @staticmethod
    def combine(fragments: List[Tuple[str, str]], ontology_files: List[str], prefix_block: str) -> str:
        header = (
            f"{prefix_block}\n"
            f"# Combined ontology generated from competency questions\n"
            f"# Reference ontologies: {', '.join(Path(f).name for f in ontology_files)}\n"
            f"# Generated on: {datetime.now().isoformat()}\n"
            f"# Total fragments: {len(fragments)}\n\n"
            "<http://www.example.org/ontology> a owl:Ontology .\n\n"
        )
        body: List[str] = []
        for i, (cq, fragment) in enumerate(fragments, 1):
            body.append(f"# ── Fragment {i}: {cq[:80]}{'...' if len(cq) > 80 else ''} ──\n")
            body.append(OntologyCombiner.remove_prefixes_from_fragment(fragment).strip())
            body.append("")
        return header + "\n".join(body)


# ==========================
# RAG Application
# ==========================

class OntologyRAG:
    def __init__(
        self,
        paths: Paths,
        embed_model: str = DEFAULT_EMBED_MODEL,
        normalize_vectors: bool = True,
        batch_size: int = 64,
        embed_format: str = "pipe",
        include_comment_in_embed: bool = True,
        avg_top_n: int = 20,
        search_k: int = 200,
        include_parents: bool = True,
        show_retrieval_scores: bool = False,
        retrieval_only: bool = False,
        log_retrieval: bool = False,
    ) -> None:
        self.paths = paths
        self.tracker = TokenCostTracker()
        self.logger = RunLogger(paths, self.tracker)
        self.extractor = OntologyExtractor(self.logger)
        self.prefix_mgr = PrefixManager()
        self.embedder = OpenAIEmbedder(embed_model, self.tracker, normalize=normalize_vectors, batch_size=batch_size)
        self.store = FaissVectorStore(inner_product=True)  # cosine if normalized
        self.llm = LLMService(LLM_MODEL, self.tracker)

        self.embed_model = embed_model
        self.embed_format = embed_format
        self.include_comment_in_embed = include_comment_in_embed
        self.avg_top_n = max(1, avg_top_n)
        self.search_k = max(self.avg_top_n, search_k)
        self.include_parents = include_parents
        self.show_retrieval_scores = show_retrieval_scores
        self.retrieval_only = retrieval_only
        self.log_retrieval = log_retrieval

        self.uri_to_element: Dict[str, OntologyElement] = {}

    @staticmethod
    def ensure_api_key() -> None:
        if not os.getenv("OPENAI_API_KEY"):
            print("[bold red]OPENAI_API_KEY is not set[/bold red]")
            sys.exit(1)

    async def _build_store(self, ontology_files: List[str]) -> str:
        print("[yellow]Building vector store from reference ontologies...[/yellow]")
        elements = self.extractor.extract_from_files(ontology_files)
        texts = [e.get_embedding_text(include_comment=self.include_comment_in_embed, format=self.embed_format) for e in elements]
        vecs = await self.embedder.embed_many(texts)
        for v, e in zip(vecs, elements):
            self.store.add(v, e)
            self.uri_to_element.setdefault(e.uri, e)
        prefix_block = self.prefix_mgr.build_prefix_block(ontology_files)
        print(f"[green]Indexed {len(self.store.elements)} ontology elements from {len(ontology_files)} ontologies[/green]")
        return prefix_block

    def _include_parent_classes_meta(
        self, elements_with_scores: List[Tuple[OntologyElement, float]]
    ) -> List[Tuple[OntologyElement, float, bool]]:
        """Return list where each item is (element, score, parent_included_flag)."""
        if not self.include_parents:
            return [(el, s, False) for el, s in elements_with_scores]

        seen: set[str] = set()
        out: List[Tuple[OntologyElement, float, bool]] = []
        for el, s in elements_with_scores:
            if el.uri not in seen:
                out.append((el, s, False))
                seen.add(el.uri)
            if el.element_type == "class":
                for parent_uri in el.super_classes or []:
                    parent = self.store.uri_to_element.get(parent_uri)
                    if parent and parent.uri not in seen:
                        out.append((parent, s - 1e-6, True))
                        seen.add(parent.uri)
        out.sort(key=lambda t: t[1], reverse=True)
        return out

    async def _retrieve_relevant_dynamic(
        self, cq: str
    ) -> Tuple[List[OntologyElement], Dict[str, Any], List[Tuple[OntologyElement, float, bool]], List[Tuple[OntologyElement, float]]]:
        """
        Returns:
          - elements_sorted: selected elements (dedup, order by score)
          - stats: dict with threshold, etc.
          - selected_meta: list of (element, score, parent_included?)
          - pool: top-K pool (element, score)
        """
        qvec = await self.embedder.embed_one(cq)
        pool = self.store.search(qvec, self.search_k)
        if not pool:
            stats = {"selected": 0, "pool": 0, "threshold": None, "top1": None, "topN": self.avg_top_n, "model": self.embed_model}
            return [], stats, [], []

        top_n = pool[: self.avg_top_n]
        avg_top = float(np.mean([s for _, s in top_n])) if top_n else pool[0][1]

        selected = [(el, s) for (el, s) in pool if s >= avg_top]
        selected_meta = self._include_parent_classes_meta(selected)

        # dedup while preserving order
        seen_uris: set[str] = set()
        elements_sorted: List[OntologyElement] = []
        for el, _, _ in selected_meta:
            if el.uri not in seen_uris:
                seen_uris.add(el.uri)
                elements_sorted.append(el)

        stats = {
            "selected": len(elements_sorted),
            "pool": len(pool),
            "threshold": avg_top,
            "top1": pool[0][1],
            "topN": self.avg_top_n,
            "model": self.embed_model,
        }
        return elements_sorted, stats, selected_meta, pool

    async def _generate_fragment(self, cq: str, relevant: List[OntologyElement], prefix_block: str) -> str:
        prompt = PromptBuilder(prefix_block).build(cq, relevant)
        response = await self.llm.complete(prompt)
        m = re.search(r"```(?:ttl|turtle)?\n(.*?)```", response, re.DOTALL)
        return (m.group(1).strip() if m else response.strip())

    def _print_retrieval_stats(self, cq_idx: int, stats: Dict[str, Any], preview: List[Tuple[OntologyElement, float]] | None = None) -> None:
        print(
            f"[blue]Retrieval stats (CQ {cq_idx}): pool={stats.get('pool')} "
            f"selected={stats.get('selected')} threshold(avg top-{stats.get('topN')}): {stats.get('threshold'):.4f} "
            f"top1={stats.get('top1'):.4f} model={stats.get('model')}[/blue]"
        )
        if self.show_retrieval_scores and preview:
            print("[magenta]Top matches (uri :: score):[/magenta]")
            for el, s in preview[:20]:
                print(f"  {el.uri} :: {s:.4f}")

    def _print_selected_elements(self, selected_meta: List[Tuple[OntologyElement, float, bool]]) -> None:
        print("\n[bold green]Selected elements (after dynamic thresholding)[/bold green]")
        for rank, (el, s, parent_included) in enumerate(selected_meta, start=1):
            tag = "(parent)" if parent_included else ""
            label = f" | label: {el.label}" if el.label else ""
            print(f"{rank:>3}. {s:.4f} {tag} [{el.element_type}] {el.uri}{label}  ← {Path(el.source_ontology).name}")
            if el.element_type != "class" and (el.domain or el.range):
                print(f"     domain={el.domain}  range={el.range}")

    def _log_retrieval_rows(
        self,
        cq_index: int,
        cq: str,
        model: str,
        threshold: float,
        topN: int,
        pool: List[Tuple[OntologyElement, float]],
        selected_meta: List[Tuple[OntologyElement, float, bool]],
    ) -> None:
        selected_set = {el.uri for (el, _, _) in selected_meta}
        parent_added_set = {el.uri for (el, _, is_parent) in selected_meta if is_parent}

        rows: List[Dict[str, Any]] = []
        for el, s in pool:
            rows.append({
                "uri": el.uri,
                "element_type": el.element_type,
                "label": el.label or "",
                "source_ontology": el.source_ontology,
                "score": s,
                "selected": el.uri in selected_set,
                "parent_included": el.uri in parent_added_set,
            })
        pool_uris = {el.uri for el, _ in pool}
        for el, s, is_parent in selected_meta:
            if el.uri not in pool_uris:
                rows.append({
                    "uri": el.uri,
                    "element_type": el.element_type,
                    "label": el.label or "",
                    "source_ontology": el.source_ontology,
                    "score": s,
                    "selected": True,
                    "parent_included": is_parent,
                })
        self.logger.log_retrieval(cq_index, cq, model, threshold, topN, len(pool), rows)

    async def run_single_cq(self, cq: str, ontology_files: List[str]) -> None:
        self.ensure_api_key()
        _ = await self._build_store(ontology_files)
        before = self.tracker.snapshot()

        relevant, stats, selected_meta, pool = await self._retrieve_relevant_dynamic(cq)
        self._print_retrieval_stats(-1, stats, preview=pool)
        self._print_selected_elements(selected_meta)

        if self.log_retrieval or self.retrieval_only:
            self._log_retrieval_rows(-1, cq, stats["model"], stats["threshold"], stats["topN"], pool, selected_meta)
            print(f"[green]Retrieval details logged to: {self.paths.retrieval_log_csv}[/green]")

        if self.retrieval_only:
            after = self.tracker.snapshot()
            self.logger.log_cost_row(-1, cq, self.tracker.diff(before, after))
            print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")
            return  # <-- stop here, no LLM / TTL

        # continue to LLM only if not retrieval-only
        prefix_block = self.prefix_mgr.build_prefix_block(ontology_files)
        fragment = await self._generate_fragment(cq, relevant, prefix_block)
        if TurtleValidator.validate_ttl(fragment):
            out_path = self.paths.individual_dir / "single_cq.ttl"
            out_path.write_text(fragment, encoding="utf-8")
            print(f"[green]Ontology fragment saved to: {out_path}[/green]")
            print("\n[bold green]Ontology fragment[/bold green]\n")
            print(fragment)
        else:
            print("[bold yellow]Generated fragment failed Turtle validation[/bold yellow]")
            print(fragment)
        after = self.tracker.snapshot()
        self.logger.log_cost_row(-1, cq, self.tracker.diff(before, after))
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")

    async def run_csv(self, csv_file: str, ontology_files: List[str], limit: Optional[int] = None) -> None:
        self.ensure_api_key()
        print(f"[bold blue]Processing CQ dataset: {csv_file}[/bold blue]")
        print(f"[bold blue]Using {len(ontology_files)} reference ontologies:[/bold blue]")
        for i, onto_file in enumerate(ontology_files, 1):
            print(f"[cyan]  {i}. {Path(onto_file).name}[/cyan]")

        if not Path(csv_file).exists():
            print(f"[red]CSV file not found: {csv_file}[/red]")
            return
        missing = [f for f in ontology_files if not Path(f).exists()]
        if missing:
            print(f"[red]Ontology files not found: {missing}[/red]")
            return

        try:
            df = pd.read_csv(csv_file)
            print(f"[green]Loaded {len(df)} rows from CSV[/green]")
        except Exception as e:
            print(f"[red]Error reading CSV: {e}[/red]")
            return

        if 'CQ' not in df.columns:
            print(f"[red]'CQ' column not found in CSV. Available columns: {list(df.columns)}[/red]")
            return

        if limit:
            df = df.head(limit)
            print(f"[blue]Processing limited to first {len(df)} rows[/blue]")

        _ = await self._build_store(ontology_files)

        fragments: List[Tuple[str, str]] = []
        successful = 0
        failed = 0
        start_time = time.time()

        from tqdm import tqdm  # local import
        pbar = tqdm(total=len(df), desc="Processing CQs")

        for idx, row in df.iterrows():
            cq = str(row['CQ']).strip()
            before = self.tracker.snapshot()

            if not cq or cq.lower() in ['nan', 'none', '']:
                print(f"[yellow]Row {idx}: Empty CQ, skipping[/yellow]")
                self.logger.log_processing_result(idx, cq, False, 0, {}, 0, "Empty CQ")
                failed += 1
                pbar.update(1)
                continue

            pbar.set_description(f"Processing CQ {idx}")
            cq_start = time.time()

            try:
                relevant, stats, selected_meta, pool = await self._retrieve_relevant_dynamic(cq)
                if self.show_retrieval_scores:
                    self._print_retrieval_stats(idx, stats, preview=pool[:10])
                self._print_selected_elements(selected_meta)

                if self.log_retrieval or self.retrieval_only:
                    self._log_retrieval_rows(idx, cq, stats["model"], stats["threshold"], stats["topN"], pool, selected_meta)
                    print(f"[green]Logged retrieval (CQ {idx}) → {self.paths.retrieval_log_csv}[/green]")

                if self.retrieval_only:
                    self.logger.log_processing_result(idx, cq, True, len(relevant), {}, time.time() - cq_start)
                    successful += 1
                else:
                    prefix_block = self.prefix_mgr.build_prefix_block(ontology_files)
                    fragment = await self._generate_fragment(cq, relevant, prefix_block)
                    if TurtleValidator.validate_ttl(fragment):
                        fragments.append((cq, fragment))
                        out_file = self.paths.individual_dir / f"cq_{idx}.ttl"
                        out_file.write_text(fragment, encoding="utf-8")
                        self.logger.log_processing_result(idx, cq, True, len(relevant), {}, time.time() - cq_start)
                        successful += 1
                        print(f"[green]CQ {idx} processed successfully[/green]")
                    else:
                        print(f"[red]CQ {idx}: Generated invalid Turtle[/red]")
                        self.logger.log_processing_result(idx, cq, False, len(relevant), {}, time.time() - cq_start, "Invalid Turtle")
                        failed += 1

            except Exception as e:
                print(f"[red]CQ {idx} failed: {e}[/red]")
                self.logger.log_processing_result(idx, cq, False, 0, {}, time.time() - cq_start, str(e))
                failed += 1
            finally:
                after = self.tracker.snapshot()
                self.logger.log_cost_row(idx, cq, self.tracker.diff(before, after))
                pbar.update(1)
                await asyncio.sleep(0.1)

        pbar.close()

        if not self.retrieval_only and fragments:
            print(f"[yellow]Combining {len(fragments)} ontology fragments...[/yellow]")
            prefix_block = self.prefix_mgr.build_prefix_block(ontology_files)
            combined = OntologyCombiner.combine(fragments, ontology_files, prefix_block)
            self.paths.combined_ttl.write_text(combined, encoding="utf-8")
            print(f"[green]Combined ontology saved to: {self.paths.combined_ttl}[/green]")
            if TurtleValidator.validate_ttl(combined):
                print("[green]Combined ontology is syntactically valid[/green]")
            else:
                print("[yellow]Combined ontology has syntax issues[/yellow]")

        total_time = time.time() - start_time
        self.logger.log_final_results(len(df), successful, failed, total_time, str(self.paths.combined_ttl))

        print(f"\n[bold green]Processing Complete![/bold green]")
        print(f"Total CQs: {len(df)}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {(successful / len(df) * 100):.1f}%")
        print(f"[bold green]💲 Estimated total API cost (USD): {self.tracker.totals_cost_usd():.6f}[/bold green]")


# ==========================
# CLI
# ==========================

def expand_onto_args(files: Optional[List[str]], dirs: Optional[List[str]]) -> List[str]:
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
    seen = set()
    result: List[str] = []
    for f in candidates:
        if f not in seen:
            result.append(f)
            seen.add(f)
    if not result:
        print("[red]No ontology files found from --onto/--onto-dir[/red]")
        sys.exit(1)
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Core ontology RAG for CQ → TTL (with dynamic retrieval & controls)")
    sub = p.add_subparsers(dest="cmd")

    def add_commonopts(ap: argparse.ArgumentParser) -> None:
        ap.add_argument("--onto", nargs="+", help="One or more reference ontology files (.ttl/.owl)")
        ap.add_argument("--onto-dir", nargs="+", help="One or more directories to scan for .ttl/.owl")
        ap.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL,
                        help="Embedding model: text-embedding-3-small|text-embedding-3-large|text-embedding-ada-002|Qwen3-Embedding-8B (via OpenAI-compatible endpoint)")
        ap.add_argument("--embed-format", choices=["pipe", "newline"], default="pipe",
                        help="Embedding text format: 'pipe' or 'newline'")
        ap.add_argument("--no-comment-in-embed", action="store_true",
                        help="Exclude rdfs:comment from the embedding text")
        ap.add_argument("--avg-top-n", type=int, default=20,
                        help="Average the top-N scores; select all results >= that average")
        ap.add_argument("--search-k", type=int, default=200,
                        help="Size of the initial FAISS pool before thresholding")
        ap.add_argument("--no-include-parents", action="store_true",
                        help="Do not automatically include parent classes")
        ap.add_argument("--show-retrieval-scores", action="store_true",
                        help="Print a preview of top matches with scores")
        ap.add_argument("--retrieval-only", action="store_true",
                        help="Only run retrieval (no LLM, no TTL output)")
        ap.add_argument("--log-retrieval", action="store_true",
                        help="Write retrieval details (scores, threshold) to retrieval_log.csv")

    p_single = sub.add_parser("cq", help="Inspect retrieval or generate ontology for a single CQ")
    p_single.add_argument("--cq", required=True, help="Competency Question text")
    add_commonopts(p_single)

    p_batch = sub.add_parser("csv", help="Process a CSV with a CQ column")
    p_batch.add_argument("--file", required=True, help="CSV file with column 'CQ'")
    p_batch.add_argument("--limit", type=int, default=None, help="Process only first N rows")
    add_commonopts(p_batch)

    return p.parse_args()


async def _main_async(args: argparse.Namespace) -> None:
    paths = Paths()
    app = OntologyRAG(
        paths,
        embed_model=getattr(args, "embed_model", DEFAULT_EMBED_MODEL),
        embed_format=getattr(args, "embed_format", "pipe"),
        include_comment_in_embed=not getattr(args, "no_comment_in_embed", False),
        avg_top_n=getattr(args, "avg_top_n", 20),
        search_k=getattr(args, "search_k", 200),
        include_parents=not getattr(args, "no_include_parents", False),
        show_retrieval_scores=getattr(args, "show_retrieval_scores", False),
        retrieval_only=getattr(args, "retrieval_only", False),
        log_retrieval=getattr(args, "log_retrieval", False),
    )

    if not args.cmd:
        print("[bold red]Usage:[/bold red] python tidied_rag_ontology.py cq --cq \"...\" --onto core.ttl more.ttl [--retrieval-only] [--log-retrieval]")
        print("         python tidied_rag_ontology.py csv --file dataset.csv --onto core.ttl more.ttl [--limit 50] [--retrieval-only] [--log-retrieval]")
        sys.exit(1)

    if args.cmd == "cq":
        ontofiles = expand_onto_args(args.onto, args.onto_dir)
        await app.run_single_cq(args.cq, ontofiles)
    elif args.cmd == "csv":
        ontofiles = expand_onto_args(args.onto, args.onto_dir)
        await app.run_csv(args.file, ontofiles, limit=args.limit)


def main() -> None:
    args = parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()


'''
# single CQ
python rag-for-testing.py cq --cq "Which materials does a fibre contain?" --onto-dir ./coreontologies  --retrieval-only --log-retrieval --embed-format newline --no-comment-in-embed  --avg-top-n 20 --search-k 200 --embed-model text-embedding-3-small --show-retrieval-scores

# CSV
python tidied_rag_ontology.py csv --file cqs.csv --onto-dir ./coreontologies \
  --retrieval-only --log-retrieval --avg-top-n 20 --search-k 200


'''