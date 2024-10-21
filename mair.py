from typing import Dict, Tuple
from datasets import load_dataset
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import faiss
import pytrec_eval



def trec_eval(qrels: Dict[str, Dict[str, int]],
              results: Dict[str, Dict[str, float]],
              k_values: Tuple[int] = (10, 50, 100, 200, 1000)) -> Dict[str, float]:
    ndcg, _map, recall = {}, {}, {}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string})
    scores = evaluator.evaluate(results)

    for query_id in scores:
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]

    def _normalize(m: dict) -> dict:
        return {k: round(v / len(scores), 5) for k, v in m.items()}

    ndcg = _normalize(ndcg)
    _map = _normalize(_map)
    recall = _normalize(recall)

    all_metrics = {}
    for mt in [ndcg, _map, recall]:
        all_metrics.update(mt)

    return all_metrics


def print_results(output_dict, metrics=['NDCG@1', 'NDCG@5', 'NDCG@10']):
    task_results = defaultdict(list)
    for k, v in output_dict.items():
        v = v[-1]
        task_results[v['task']].append(v)
    table_data = []
    avg_score = [0 for _ in range(len(metrics))]
    avg_size = [0 for _ in range(len(metrics))]
    for task in task_results:
        line = [task]
        for i, metric in enumerate(metrics):
            score = [x['eval_results'][metric] * x['size'] for x in task_results[task]]
            size = [x['size'] for x in task_results[task]]
            avg_score[i] += sum(score)
            avg_size[i] += sum(size)
            score = sum(score) / sum(size)
            score = score * 100
            line.append(f"{score:.2f}")
        table_data.append(line)
    line = ['Avg']
    for i, metric in enumerate(metrics):
        score = avg_score[i] / avg_size[i]
        score = score * 100
        line.append(f"{score:.2f}")
    table_data.append(line)

    headers = ["Data"] + metrics

    try:
        from tabulate import tabulate
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    except ModuleNotFoundError:
        column_widths = [max(len(str(item)) for item in column) for column in zip(headers, *table_data)]
        header_row = " | ".join(f"{headers[i]:^{column_widths[i]}}" for i in range(len(headers)))
        print(f"| {header_row} |")
        separator_row = "-+-".join('-' * column_widths[i] for i in range(len(headers)))
        print(f"{separator_row}")
        for row in table_data:
            row_str = " | ".join(f"{row[i]:^{column_widths[i]}}" for i in range(len(row)))
            print(f"| {row_str} |")


def eval_embedding(model, tasks, instruct=True):
    output_dict = defaultdict(list)
    for task in tasks:
        if task in output_dict:
            continue
        data = load_dataset('MAIR-Bench/MAIR-Queries', task)
        docs = load_dataset('MAIR-Bench/MAIR-Docs', task)
        for split in data:
            doc_split = 'docs' if split == 'queries' else split.replace('_queries', '_docs')
            doc_content = [item['doc'] for item in docs[doc_split]]
            doc_embedding = model.encode(doc_content, batch_size=32, show_progress_bar=True, max_length=2048)
            doc_embedding = np.asarray(doc_embedding, dtype=np.float32)

            dim = doc_embedding.shape[1]
            index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
            index.add(doc_embedding)

            query_embedding = []
            for item in data[split]:
                if instruct:
                    query_embedding.append(model.encode(item['query'], prompt=item['instruction']))
                else:
                    query_embedding.append(model.encode(item['query']))
            query_embedding = np.asarray(query_embedding, dtype=np.float32)
            distance, rank = index.search(query_embedding, 100)

            qrels = {}
            for item in data[split]:
                qrels[item['qid']] = {str(x['id']): int(x['score']) for x in item['labels']}
            results = {}
            for item, rk, ds in zip(data[split], rank, distance):
                results[item['qid']] = {}
                for r, d in zip(rk, ds):
                    results[item['qid']][str(docs[doc_split][int(r)]['id'])] = float(d)
            eval_results = trec_eval(qrels, results, k_values=(1, 5, 10, 100))
            output_dict[task + '/' + split].append(
                {'task': task, 'split': split, 'eval_results': eval_results, 'size': len(data[split]),
                 'results': results})
            print(task + '/' + split, eval_results)
    print_results(output_dict)
    return output_dict


def eval_rerank(model, tasks, instruct=True, first_stage=None):
    if first_stage is None:
        first_stage = load_dataset('MAIR-Bench/MAIR-Results-text-embedding-3-small')['train']
    
    output_dict = defaultdict(list)
    for task in tasks:
        if task in output_dict:
            continue
        data = load_dataset('MAIR-Bench/MAIR-Queries', task)
        docs = load_dataset('MAIR-Bench/MAIR-Docs', task)
        for split in data:
            doc_split = 'docs' if split == 'queries' else split.replace('_queries', '_docs')
            try:
                results = first_stage[task + '/' + split][-1]['results']
            except:
                results = first_stage[task + '/' + split][-1][-1]['results']
            query_data = {item['qid']: item for item in data[split]}
            doc_data = {item['id']: item for item in docs[doc_split]}
            new_results = {}
            for qid in tqdm(results):
                new_results[qid] = {}
                candidates = []
                for doc_id in results[qid]:
                    candidates.append(doc_data[doc_id])
                candidates = candidates[:100]

                query = query_data[qid]['query']
                if instruct:
                    try:  # try to input instruction as prompt
                        rankings = model.rank(query, [x['doc'] for x in candidates], prompt=query_data[qid]['instruction'])
                    except:  #
                        query = f"Instruct: {query_data[qid]['instruction']}\nQuery: {query}"
                        rankings = model.rank(query, [x['doc'] for x in candidates])
                else:
                    rankings = model.rank(query, [x['doc'] for x in candidates])

                for ranking in rankings:
                    doc_id = candidates[ranking['corpus_id']]['id']
                    new_results[qid][doc_id] = float(ranking['score'])

            qrels = {}
            for item in data[split]:
                qrels[item['qid']] = {str(x['id']): int(x['score']) for x in item['labels']}
            eval_results = trec_eval(qrels, new_results, k_values=(1, 5, 10, 100))
            output_dict[task + '/' + split].append(
                {'task': task, 'split': split, 'eval_results': eval_results, 'size': len(data[split]),
                 'results': new_results})
            print(task + '/' + split, eval_results)
    print_results(output_dict)
    return output_dict



def eval_bm25(tasks, instruct=True):
    import bm25s
    output_dict = defaultdict(list)
    for task in tasks:
        if task in output_dict:
            continue
        data = load_dataset('MAIR-Bench/MAIR-Queries', task)
        docs = load_dataset('MAIR-Bench/MAIR-Docs', task)
        for split in data:
            doc_split = 'docs' if split == 'queries' else split.replace('_queries', '_docs')
            doc_content = [item['doc'] for item in docs[doc_split]]
            doc_ids = [item['id'] for item in docs[doc_split]]
            corpus_tokens = bm25s.tokenize(doc_content, stopwords="en")
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)

            results = {}
            for item in data[split]:
                query = item['query']
                if instruct:
                    query = item['instruction'] + ' ' + query
                query_tokens = bm25s.tokenize(query)
                if len(query_tokens.vocab) == 0:
                    query_tokens = bm25s.tokenize('NONE', stopwords=[])
                hits, scores = retriever.retrieve(query_tokens, corpus=doc_ids, k=min(100, len(doc_ids)))
                results[item['qid']] = {}
                for i in range(len(hits[0])):
                    results[item['qid']][hits[0, i]] = float(scores[0, i])

            qrels = {}
            for item in data[split]:
                qrels[item['qid']] = {str(x['id']): int(x['score']) for x in item['labels']}

            eval_results = trec_eval(qrels, results, k_values=(1, 5, 10, 100))
            output_dict[task + '/' + split].append(
                {'task': task, 'split': split, 'eval_results': eval_results, 'size': len(data[split]),
                 'results': results})
            print(task + '/' + split, eval_results)
    print_results(output_dict)
    return output_dict


