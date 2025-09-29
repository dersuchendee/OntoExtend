from __future__ import annotations
import os
import re
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import faiss
from rdflib import Graph, Literal, OWL, RDF, RDFS, URIRef


try:
    from call_LLM_API import call_LLM_API
except Exception:
    try:
        from Components.CallLLMUnit import call_LLM_API
    except Exception:
        call_LLM_API = None


# ==========================
# Configuration
# ==========================

SYS_PROMPT = (
    "You are an ontology engineer. Reuse elements from the provided core ontology when possible. "
    "If you must create new elements, append '# GENERATED' as a comment. "
    "Return only Turtle/TTL syntax inside one markdown code block."
)

USER_TMPL = """

You are a helpful assistant designed to generate ontologies. You receive a Competency Question (CQ) and an Ontology Story (OS). 
Based on CQ, which is a requirement for the ontology, and OS, which tells you what the context of the ontology is, your task is generating one ontology (O). The goal is to generate O that models the CQ properly. This means there is a way to write a SPARQL query to extract the answer to this CQ in O.  
Reuse the relevant ontology elements provided in the RELEVANT ONTOLOGY ELEMENTS section below whenever possible. These elements come from multiple reference ontologies. Only create new elements if absolutely necessary. In any case, add labels and comments. 
Use the following prefixes: 
{prefix_block}

Do not include A-Box instances. Generate only the OWL TBox in Turtle syntax. Outside of the code box do not add commentary.
Instructions:
1. Analyze the CQ to understand what concepts and relationships are needed
2. Map the required concepts to classes and properties from RELEVANT ONTOLOGY ELEMENTS
3. Prefer elements with higher semantic similarity to the CQ concepts
4. If something required is missing, create it under the : namespace and mark with '# GENERATED'
5. Output only the TBox ontology in Turtle syntax
6. Include restrictions implied by the CQ
7. Ensure the output is syntactically correct and self-contained

Competency Question: "{cq}"

RELEVANT ONTOLOGY ELEMENTS (from {num_ontologies} reference ontologies):
{relevant_elements}

"""

STANDARD_PREFIXES: Dict[str, str] = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


# ==========================
# Utilities and data
# ==========================

@dataclass
class OntologyElement:
    uri: str
    element_type: str
    source_ontology: str
    label: Optional[str] = None
    comment: Optional[str] = None
    domain: List[str] = field(default_factory=list)
    range: List[str] = field(default_factory=list)
    super_classes: List[str] = field(default_factory=list)
    sub_classes: List[str] = field(default_factory=list)

    def get_searchable_text(self) -> str:
        parts: List[str] = []
        local_name = self.uri.split('#')[-1].split('/')[-1]
        parts.append(local_name)
        if self.label:
            parts.append(self.label)
        if self.comment:
            parts.append(self.comment)
        parts.append(f"Type: {self.element_type.replace('_', ' ')}")
        parts.append(f"From: {Path(self.source_ontology).stem}")
        if self.domain:
            dnames = [d.split('#')[-1].split('/')[-1] for d in self.domain]
            parts.append(f"Domain: {', '.join(dnames)}")
        if self.range:
            rnames = [r.split('#')[-1].split('/')[-1] for r in self.range]
            parts.append(f"Range: {', '.join(rnames)}")
        return " | ".join(parts)

    def get_embedding_text(self, include_comment: bool = True, fmt: str = "pipe") -> str:
        items: List[str] = [
            self.uri,
            self.label or "",
            self.comment or "" if include_comment else "",
            self.element_type.replace("_", " "),
            ",".join(self.domain or []),
            ",".join(self.range or []),
        ]
        if fmt == "newline":
            return "\n".join(items).strip()
        return " | ".join(items).strip()


class OntologyExtractor:
    def extract_from_files(self, ontology_files: List[str]) -> List[OntologyElement]:
        elements: List[OntologyElement] = []
        for ontology_file in ontology_files:
            p = Path(ontology_file)
            if not p.exists():
                continue
            g = Graph()
            try:
                g.parse(str(p))
            except Exception:
                continue
            elements.extend(self._extract_graph(g, str(p)))
        return elements

    def _extract_graph(self, g: Graph, source_ontology: str) -> List[OntologyElement]:
        elements: List[OntologyElement] = []
        for cls in list(g.subjects(RDF.type, OWL.Class)):
            if isinstance(cls, URIRef):
                elements.append(self._extract_class_info(g, cls, source_ontology))
        for prop in list(g.subjects(RDF.type, OWL.ObjectProperty)):
            if isinstance(prop, URIRef):
                elements.append(self._extract_property_info(g, prop, "object_property", source_ontology))
        for prop in list(g.subjects(RDF.type, OWL.DatatypeProperty)):
            if isinstance(prop, URIRef):
                elements.append(self._extract_property_info(g, prop, "data_property", source_ontology))
        for prop in list(g.subjects(RDF.type, OWL.AnnotationProperty)):
            if isinstance(prop, URIRef):
                elements.append(self._extract_property_info(g, prop, "annotation_property", source_ontology))
        return elements

    @staticmethod
    def _lit(g: Graph, s: URIRef, p: URIRef) -> Optional[str]:
        for obj in g.objects(s, p):
            if isinstance(obj, Literal):
                return str(obj)
        return None

    def _extract_class_info(self, g: Graph, cls: URIRef, source_ontology: str) -> OntologyElement:
        label = self._lit(g, cls, RDFS.label)
        comment = self._lit(g, cls, RDFS.comment)
        super_classes = [str(sc) for sc in g.objects(cls, RDFS.subClassOf) if isinstance(sc, URIRef)]
        sub_classes = [str(sc) for sc in g.subjects(RDFS.subClassOf, cls) if isinstance(sc, URIRef)]
        return OntologyElement(
            uri=str(cls),
            element_type="class",
            source_ontology=source_ontology,
            label=label,
            comment=comment,
            super_classes=super_classes,
            sub_classes=sub_classes,
        )

    def _extract_property_info(self, g: Graph, prop: URIRef, ptype: str, source_ontology: str) -> OntologyElement:
        label = self._lit(g, prop, RDFS.label)
        comment = self._lit(g, prop, RDFS.comment)
        domain = [str(d) for d in g.objects(prop, RDFS.domain) if isinstance(d, URIRef)]
        range_vals = [str(r) for r in g.objects(prop, RDFS.range) if isinstance(r, URIRef)]
        return OntologyElement(
            uri=str(prop),
            element_type=ptype,
            source_ontology=source_ontology,
            label=label,
            comment=comment,
            domain=domain,
            range=range_vals,
        )


class PrefixManager:
    def collect_prefixes_from_files(self, ontology_files: List[str]) -> Dict[str, str]:
        ns_to_prefix: Dict[str, str] = {}
        used = set(STANDARD_PREFIXES.keys())
        for f in ontology_files:
            g = Graph()
            try:
                g.parse(f)
            except Exception:
                continue
            for pref, ns in g.namespaces():
                ns = str(ns)
                if not ns:
                    continue
                if str(pref).lower() == "xml":
                    continue
                if ns in STANDARD_PREFIXES.values():
                    continue
                p = self._sanitize_prefix_name(str(pref))
                while p in used:
                    p = self._bump_suffix(p)
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

    @staticmethod
    def _sanitize_prefix_name(raw: str) -> str:
        import re as _re
        s = (raw or "").strip()
        s = _re.sub(r"[^A-Za-z0-9_\-]", "_", s)
        if not s or not _re.match(r"^[A-Za-z_]", s):
            s = f"ns_{s}" if s else "ns"
        s = _re.sub(r"_+", "_", s)
        return s

    @staticmethod
    def _bump_suffix(s: str) -> str:
        import re as _re
        m = _re.search(r"(\d+)$", s)
        return _re.sub(r"\d+$", str(int(m.group(1)) + 1), s) if m else f"{s}2"


class PromptBuilder:
    def __init__(self, prefix_block: str) -> None:
        self.prefix_block = prefix_block

    @staticmethod
    def _format_relevant(elements: List[OntologyElement]) -> Tuple[str, int]:
        from collections import defaultdict
        by_src: Dict[str, List[OntologyElement]] = defaultdict(list)
        for e in elements:
            by_src[Path(e.source_ontology).stem].append(e)
        parts: List[str] = []
        for source, els in by_src.items():
            parts.append(f"\n## From {source} ontology:")
            for e in els:
                parts.append(f"- {e.get_searchable_text()}")
        return "\n".join(parts), len(by_src)

    def build(self, cq: str, elements: List[OntologyElement]) -> str:
        relevant_text, num_ontologies = self._format_relevant(elements)
        return USER_TMPL.format(
            num_ontologies=num_ontologies,
            relevant_elements=relevant_text,
            cq=cq,
            prefix_block=self.prefix_block,
        )


class TurtleValidator:
    @staticmethod
    def validate(ttl_content: str) -> bool:
        try:
            g = Graph()
            g.parse(data=ttl_content, format="turtle")
            return True
        except Exception:
            return False


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

    def add(self, vecs: np.ndarray, elements: List[OntologyElement]) -> None:
        self._ensure_index(vecs.shape[1])
        self.idx.add(vecs.astype("float32"))  # type: ignore
        self.elements.extend(elements)
        for e in elements:
            self.uri_to_element[e.uri] = e

    def search(self, qvec: np.ndarray, k: int) -> List[Tuple[OntologyElement, float]]:
        if self.idx is None or self.idx.ntotal == 0:
            return []
        D, I = self.idx.search(qvec.astype("float32"), k)
        out: List[Tuple[OntologyElement, float]] = []
        for j, i in enumerate(I[0]):
            if i == -1:
                continue
            out.append((self.elements[int(i)], float(D[0][j])))
        return out


# ==========================
# Embedding caller
# ==========================

class EmbeddingCaller:
    """
    Provider-agnostic embeddings without touching call_LLM_API.
    Supports OpenAI and Azure OpenAI via environment or API_inputs.
    """

    def __init__(self, api: Optional[str] = None, model: Optional[str] = None, api_inputs: Optional[Dict[str, Any]] = None,
                 normalize: bool = True, batch_size: int = 128) -> None:
        self.api = api or os.getenv("RAG_EMBED_API", "OpenAI")
        self.model = model or os.getenv("RAG_EMBED_MODEL", "text-embedding-3-small")
        self.api_inputs = api_inputs or {}
        self.normalize = normalize
        self.batch_size = max(1, int(batch_size))

        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        if self.api == "OpenAI":
            import openai
            self._client = openai
        elif self.api in {"Azure", "LiU_Azure", "AzureOpenAI"}:
            from openai import AzureOpenAI
            endpoint = self.api_inputs.get("endpoint") or os.getenv("AZURE_OPENAI_ENDPOINT") or os.getenv("endpoint")
            api_version = self.api_inputs.get("api_version") or os.getenv("AZURE_OPENAI_API_VERSION") or os.getenv("api_version")
            api_key = self.api_inputs.get("api_key") or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("API_KEY")
            if not endpoint or not api_version or not api_key:
                raise RuntimeError("Azure embeddings require endpoint, api_version, and api_key")
            self._client = AzureOpenAI(azure_endpoint=endpoint, api_version=api_version, api_key=api_key)
        else:
            raise RuntimeError(f"Unsupported embedding API: {self.api}")

    def _l2norm(self, arr: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return arr
        n = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
        return arr / n

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vecs: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            if self.api == "OpenAI":
                # openai.Embeddings.create returns dict-like
                r = self._client.Embeddings.create(model=self.model, input=batch)
                emb = np.asarray([d["embedding"] for d in r["data"]], dtype="float32")
            else:
                # AzureOpenAI client
                r = self._client.embeddings.create(model=self.model, input=batch)
                emb = np.asarray([d.embedding for d in r.data], dtype="float32")
            vecs.append(emb)
        out = np.vstack(vecs) if vecs else np.zeros((0, 1536), dtype="float32")
        return self._l2norm(out)

    def embed_one(self, text: str) -> np.ndarray:
        arr = self.embed_texts([text])
        return arr[0:1, :]


# ==========================
# RAG caller
# ==========================

def _expand_onto_args(files: Optional[List[str]], dirs: Optional[List[str]]) -> List[str]:
    candidates: List[str] = []
    if files:
        candidates.extend(files)
    if dirs:
        for d in dirs:
            p = Path(d)
            if p.exists() and p.is_dir():
                candidates.extend(str(f) for f in p.rglob("*.ttl"))
                candidates.extend(str(f) for f in p.rglob("*.owl"))
    seen = set()
    result: List[str] = []
    for f in candidates:
        if f not in seen:
            result.append(f)
            seen.add(f)
    return result


class RAGCaller:
    """
    Retrieval augmented ontology generator.
    Keeps call_LLM_API completely separate. You may pass API and LLM at call time or rely on env defaults.
    """

    def __init__(
        self,
        ontology_files: Optional[List[str]] = None,
        ontology_dirs: Optional[List[str]] = None,
        embed_api: Optional[str] = None,
        embed_model: Optional[str] = None,
        embed_api_inputs: Optional[Dict[str, Any]] = None,
        include_comment_in_embed: bool = True,
        embed_format: str = "pipe",
        include_parents: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        env_files = os.getenv("CORE_ONTO_FILES")
        env_dirs = os.getenv("CORE_ONTO_DIRS")

        files_from_env = [s for s in (env_files.split(";") if env_files else []) if s.strip()]
        dirs_from_env = [s for s in (env_dirs.split(";") if env_dirs else []) if s.strip()]
        if ontology_files is None and files_from_env:
            ontology_files = files_from_env
        if ontology_dirs is None and dirs_from_env:
            ontology_dirs = dirs_from_env
        if ontology_files is None and ontology_dirs is None and Path("./coreontologies").exists():
            ontology_dirs = ["./coreontologies"]

        self.ontology_files = _expand_onto_args(ontology_files, ontology_dirs)
        if not self.ontology_files:
            raise RuntimeError("No ontology files found. Set CORE_ONTO_DIRS or pass ontology_dirs/files to RAGCaller.")

        self.include_comment_in_embed = include_comment_in_embed
        self.embed_format = embed_format
        self.include_parents = include_parents

        self.prefix_mgr = PrefixManager()
        self.extractor = OntologyExtractor()
        self.store = FaissVectorStore(inner_product=True)

        self.embedder = EmbeddingCaller(api=embed_api, model=embed_model, api_inputs=embed_api_inputs)

        self.cache_dir = Path(cache_dir or os.getenv("RAG_CACHE_DIR", str(Path.home() / ".core_ontology_rag_cache")))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._built = False
        self._elements: List[OntologyElement] = []

    def _build_store(self) -> None:
        if self._built:
            return
        elements = self.extractor.extract_from_files(self.ontology_files)
        if not elements:
            raise RuntimeError("No elements extracted from ontology files.")
        texts = [e.get_embedding_text(include_comment=self.include_comment_in_embed, fmt=self.embed_format) for e in elements]
        vecs = self.embedder.embed_texts(texts)
        self.store.add(vecs, elements)
        self._elements = elements
        self._built = True

    def _include_parent_classes_meta(
        self, elements_with_scores: List[Tuple[OntologyElement, float]]
    ) -> List[Tuple[OntologyElement, float, bool]]:
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

    def _dynamic_select(self, pool: List[Tuple[OntologyElement, float]], topN: int) -> List[OntologyElement]:
        if not pool:
            return []
        top_n = pool[: topN]
        avg_top = float(np.mean([s for _, s in top_n])) if top_n else pool[0][1]
        selected = [(el, s) for (el, s) in pool if s >= avg_top]
        selected_meta = self._include_parent_classes_meta(selected)

        seen: set[str] = set()
        ordered: List[OntologyElement] = []
        for el, _, _ in selected_meta:
            if el.uri not in seen:
                seen.add(el.uri)
                ordered.append(el)
        return ordered

    def _build_prompt(self, cq: str, elements: List[OntologyElement]) -> str:
        prefix_block = self.prefix_mgr.build_prefix_block(self.ontology_files)
        return PromptBuilder(prefix_block).build(cq, elements)

    def CallRAG(
        self,
        CQ: str,
        topK: int = 20,
        searchK: int = 200,
        retrieval_only: bool = False,
        API: Optional[str] = None,
        API_inputs: Optional[Dict[str, Any]] = None,
        LLM: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Returns a TTL fragment string if retrieval_only is False.
        If retrieval_only is True, returns a JSON string with retrieval details and selected element summaries.
        """

        if call_LLM_API is None and not retrieval_only:
            raise RuntimeError("call_LLM_API is not importable. Pass it via your environment or adjust imports.")

        self._build_store()

        qvec = self.embedder.embed_one(CQ)
        # Ensure searchK within index bounds
        k = min(max(1, searchK), int(self.store.idx.ntotal) if self.store.idx is not None else 1)
        pool = self.store.search(qvec, k)

        selected = self._dynamic_select(pool, topN=max(1, topK))

        if retrieval_only:
            payload = {
                "cq": CQ,
                "pool_size": len(pool),
                "selected": len(selected),
                "selected_elements": [
                    {
                        "uri": e.uri,
                        "type": e.element_type,
                        "label": e.label,
                        "source": Path(e.source_ontology).name,
                        "domain": e.domain,
                        "range": e.range,
                    }
                    for e in selected
                ],
            }
            return json.dumps(payload, ensure_ascii=False, indent=2)

        prompt = self._build_prompt(CQ, selected)

        api = API or os.getenv("RAG_LLM_API", "OpenAI")
        model = LLM or os.getenv("RAG_LLM_MODEL", "GPT-5")
        inputs = API_inputs or {}


        prompt_to_send = (system_prompt or SYS_PROMPT) + "\n" + prompt

        text = call_LLM_API(api, inputs, model, prompt_to_send)  # external, not modified here

        # Extract code block if present
        m = re.search(r"```(?:ttl|turtle)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        ttl = m.group(1).strip() if m else text.strip()

        if not TurtleValidator.validate(ttl):
            repair_prompt = (
                "Fix only Turtle syntax errors. Return the corrected TTL in one code block. "
                "Do not change semantics.\n\n" + ttl
            )
            repaired = call_LLM_API(api, inputs, model, repair_prompt)
            m2 = re.search(r"```(?:ttl|turtle)?\s*(.*?)```", repaired, flags=re.DOTALL | re.IGNORECASE)
            ttl2 = m2.group(1).strip() if m2 else repaired.strip()
            if TurtleValidator.validate(ttl2):
                return ttl2
            return ttl

        return ttl
