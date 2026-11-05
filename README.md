# KiberJog-RAG

Magyar nyelvű kérdés-válasz rendszer a NIS2-höz, a DORA-hoz és a 2024. évi LXIX. törvényhez (cikk/§-szintű hivatkozással).
BGE-M3 + BM25 hibrid keresés Qdrantban, bge-reranker-v2-m3, Qwen3.5-9B (llama.cpp, GGUF Q8_0).

## Futtatás
1. Források: `data/raw/` (EUR-Lex XHTML: `publications.europa.eu/resource/celex/32022L2555` és `32022R2554`, `Accept: application/xhtml+xml`, `Accept-Language: hun`; njt.hu: `https://njt.hu/jogszabaly/2024-69-00-00` -> `kibertv.html`).
2. `pip install -r requirements.txt`; Qdrant: `docker run -p 6333:6333 qdrant/qdrant`.
3. Qwen3.5-9B: `unsloth/Qwen3.5-9B-GGUF` (`Qwen3.5-9B-Q8_0.gguf`) a `models/` mappába, majd
   `docker run --gpus all -p 127.0.0.1:8191:8080 -v $PWD/models:/models <llama.cpp-server-image> -m /models/Qwen3.5-9B-Q8_0.gguf -ngl 99 -c 98304 -np 3 -fa on --jinja`
   (a 8191-es port az `src/rag.py` `LLM_URL` változója). A bíráló/kérdésgeneráló nagy modell a `src/run_eval.py` `JUDGE` címén várt.
4. `python src/ingest.py` -> `python -c "import sys;sys.path.insert(0,'src');from rag import *;[build_index(v) for v in 'ABC']"`
5. `python src/gen_testset.py` (nyers kérdések; a válogatott lista: `eval/testset.jsonl`)
6. `python src/run_eval.py` (retr, gen, judge) -> `python src/report.py` -> `python src/build_doc.py`

Eredmények: `results/summary.json`, ábrák: `figs/`. Dokumentáció és prezentáció: `docs/`.
