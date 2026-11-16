import datasets
import transformers.data.metrics.squad_metrics as squad_metrics


def _score_one(doc, prediction):
    reference = doc["answers"]["text"][0]
    f1_sum = squad_metrics.compute_f1(reference, prediction)
    exact_match = squad_metrics.compute_exact(reference, prediction)
    return {"f1": f1_sum, "exact_match": exact_match}


def process_results(doc, results):
    # results[0] holds the K sampled generations; average their scores
    scored = [_score_one(doc, prediction) for prediction in results[0]]
    return {key: sum(s[key] for s in scored) / len(scored) for key in scored[0]}


def process_docs(dataset: datasets.Dataset):
    def _helper(doc):
        doc["title"] = doc["context"].strip().split("\n")[0].strip()
        doc["passage"] = "\n".join(doc["context"].strip().split("\n")[1:]).strip()
        doc["question"] = " ".join(doc["question"].strip().split())
        return doc

    return dataset.map(_helper)
