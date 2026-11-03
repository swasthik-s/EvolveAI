"""Persistent source-grounded memory; graph edges encode co-occurrence only."""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

TOKEN_RE=re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
STOP=set("the and for are was were with that this from have has had not but you your they their them its into about after before when where which what who how why can could would should will shall may might must does did than then there here also been being each other more most some such only over under between through because while within without onto upon these those all any one two three four five first second new use used using make made many much very".split())

def tokens(text):
    return [x.lower() for x in TOKEN_RE.findall(text) if x.lower() not in STOP]

def concepts(text,limit=12):
    words=tokens(text); uni=Counter(w for w in words if len(w)>=4)
    bi=Counter(a+" "+b for a,b in zip(words,words[1:]) if a!=b)
    ranked=sorted([(k,v,1) for k,v in uni.items()]+[(k,v,2) for k,v in bi.items()],key=lambda z:(z[1]*z[2],z[2],z[0]),reverse=True)
    out=[]
    for term,_,_ in ranked:
        if term not in out: out.append(term)
        if len(out)>=limit: break
    return out

class SemanticMemory:
    def __init__(self,directory):
        self.directory=Path(directory); self.directory.mkdir(parents=True,exist_ok=True)
        self.docs_path=self.directory/"semantic_documents.jsonl"
        self.graph_path=self.directory/"knowledge_graph.json"
        self.manifest_path=self.directory/"semantic_memory_manifest.json"
        self.documents={}; self.nodes={}; self.edges={}; self._load()

    def _load(self):
        if self.docs_path.exists():
            for line in self.docs_path.read_text(encoding="utf-8").splitlines():
                try:
                    d=json.loads(line); self.documents[d["document_id"]]=d
                except (ValueError,KeyError): pass
        if self.graph_path.exists():
            try:
                g=json.loads(self.graph_path.read_text(encoding="utf-8"))
                self.nodes=g.get("nodes",{}); self.edges=g.get("edges",{})
            except (ValueError,OSError): pass

    def add_document(self,text,source_ids=None,domain="unknown",source_event=None):
        text=re.sub(r"\\s+"," ",str(text)).strip()
        if len(text)<20: return False
        sources=sorted(set(source_ids or []))
        did=hashlib.sha256((domain+"|"+",".join(sources)+"|"+text).encode()).hexdigest()
        if did in self.documents:
            self.documents[did]["observations"]+=1; return False
        cs=concepts(text); now=datetime.now(timezone.utc).isoformat()
        self.documents[did]={"document_id":did,"text":text,"domain":domain,"source_ids":sources,
          "source_event":source_event,"concepts":cs,"observations":1,"created_at_utc":now,
          "extraction":"heuristic frequency-ranked terms and adjacent bigrams"}
        for term in cs:
            node=self.nodes.setdefault(term,{"concept":term,"document_count":0,"domains":{},"source_ids":[],"first_seen_utc":now})
            node["document_count"]+=1; node["domains"][domain]=node["domains"].get(domain,0)+1
            node["source_ids"]=sorted(set(node["source_ids"])|set(sources))
        for i,a0 in enumerate(cs):
            for b0 in cs[i+1:]:
                a,b=sorted((a0,b0)); eid=hashlib.sha256((a+"\\0"+b).encode()).hexdigest()
                edge=self.edges.setdefault(eid,{"source":a,"target":b,"relation":"co_occurs_in_source","weight":0,"evidence_document_ids":[]})
                edge["weight"]+=1
                if len(edge["evidence_document_ids"])<100: edge["evidence_document_ids"].append(did)
        return True

    def ingest_episodic_jsonl(self,path,max_events=None):
        path=Path(path)
        if not path.is_file(): raise FileNotFoundError(path)
        added=skipped=examples=events=0
        with path.open(encoding="utf-8") as f:
            for i,line in enumerate(f):
                if max_events is not None and i>=max_events: break
                events+=1
                try: event=json.loads(line)
                except ValueError: skipped+=1; continue
                for sample in event.get("examples",[]):
                    examples+=1
                    ok=self.add_document(sample,event.get("source_ids",[]),event.get("domain","unknown"),
                      {"timestamp_utc":event.get("timestamp_utc"),"step":event.get("step")})
                    if ok: added+=1
                    else: skipped+=1
        self.save()
        return {"events_read":events,"examples_seen":examples,"documents_added":added,
          "duplicates_or_rejected":skipped,**self.stats()}

    def save(self):
        tmp=self.docs_path.with_suffix(".tmp")
        with tmp.open("w",encoding="utf-8") as f:
            for d in self.documents.values(): f.write(json.dumps(d,ensure_ascii=False)+"\\n")
        tmp.replace(self.docs_path)
        graph={"schema_version":1,"edge_semantics":"co-occurrence only; not verified semantic or causal relations","nodes":self.nodes,"edges":self.edges}
        tmp=self.graph_path.with_suffix(".tmp"); tmp.write_text(json.dumps(graph,ensure_ascii=False),encoding="utf-8"); tmp.replace(self.graph_path)
        manifest={"schema_version":1,**self.stats(),"retrieval":"BM25-style lexical ranking; no embeddings or LLM",
          "limitations":["Concepts are heuristic candidates.","Edges indicate source co-occurrence only.","Review source evidence before treating statements as facts."]}
        self.manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    def search(self,query,top_k=5):
        q=tokens(query)
        if not q:return []
        n=max(1,len(self.documents)); docs={k:tokens(d["text"]) for k,d in self.documents.items()}
        df=Counter(t for arr in docs.values() for t in set(arr)); avg=sum(map(len,docs.values()))/n or 1
        hits=[]
        for did,arr in docs.items():
            tf=Counter(arr); score=0; norm=len(arr)/avg
            for term in q:
                if tf[term]:
                    idf=math.log(1+(n-df[term]+.5)/(df[term]+.5))
                    score+=idf*(tf[term]*2.5)/(tf[term]+1.5*(1-.75+.75*norm))
            if score:
                d=self.documents[did]
                hits.append({"score":round(score,5),"document_id":did,"domain":d["domain"],
                  "source_ids":d["source_ids"],"text":d["text"],"concepts":d["concepts"]})
        return sorted(hits,key=lambda h:h["score"],reverse=True)[:max(1,top_k)]

    def related_concepts(self,concept,limit=10):
        key=concept.strip().lower(); out=[]
        for e in self.edges.values():
            if e["source"]==key: out.append({"concept":e["target"],"weight":e["weight"],"relation":e["relation"]})
            elif e["target"]==key: out.append({"concept":e["source"],"weight":e["weight"],"relation":e["relation"]})
        return sorted(out,key=lambda x:x["weight"],reverse=True)[:limit]

    def stats(self):
        domains=Counter(d["domain"] for d in self.documents.values())
        return {"documents":len(self.documents),"concept_nodes":len(self.nodes),"cooccurrence_edges":len(self.edges),
          "documents_by_domain":dict(domains),"storage_directory":str(self.directory)}
