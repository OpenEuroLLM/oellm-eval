import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Join the sentences of each source article into one document.

    FLORES-200 takes several sentences from each Wikinews, Wikijunior or
    Wikivoyage article. Scoring them per article gives each sentence the ones
    before it as context, as in running text, instead of scoring every
    sentence cold. Articles keep the order in which they first appear, and
    sentences their order within the article.
    """
    articles: dict[str, list[str]] = {}
    for url, sentence in zip(dataset["URL"], dataset["sentence"], strict=True):
        articles.setdefault(url, []).append(sentence.strip())
    return datasets.Dataset.from_dict(
        {
            "URL": list(articles),
            "text": [" ".join(sentences) for sentences in articles.values()],
        }
    )
