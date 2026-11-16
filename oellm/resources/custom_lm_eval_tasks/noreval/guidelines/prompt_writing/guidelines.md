## Creating a Collection of Norwegian Prompts: Guidelines

### Overview

Your annotation task is to create a pool of diverse prompts for evaluating Norwegian LMs on a broad scope of downstream tasks, with 3-5 prompts per task. Our evaluation tasks include sentiment analysis, machine translation, grammatical error correction, text summarization, question answering, and idiom completion.

### Annotation task

* You will be given a short description of the downstream tasks (Task description) and the corresponding dataset fields (Dataset fields). We also provide prompt examples in English as references (Prompt examples). Please read this information and have a look at the examples. Please adapt the examples to Norwegian Bokmål, e.g., via translation, or write your own prompt templates from scratch, formatting the dataset fields in double curly brackets (Norwegian prompts). 
* Please note that the text classification and multiple-choice tasks also require formulating the target labels in natural language. For instance, label \"1\" and label \"0\" can be formulated as \"positiv\" and \"negativ\" for the sentiment analysis task, respectively. Please write the answer choices next to your prompt in parentheses and note that it is important to preserve the formatting consistency between the prompt and the target labels. We will unify the prompt and target label formatting when integrating them into the framework.
* The maximum number of prompts per downstream task is 5. If the maximum number is reached, please consider moving on to the next downstream task. The only exception here is the machine translation task, where we have three language pairs. Therefore, we aim for max. 15 prompts here, with a max. 5 prompts per pair.
* Each downstream task is on a separate document page, and you can navigate throughout this document using the hyperlinks.
* Please feel free to leave comments and suggestions in this document.


### Annotation examples

We provide annotation examples based on the task type, which defines formatting prompts and target labels: text classification, multiple-choice question answering, and natural language generation (machine translation, text summarization, grammatical error correction, extractive question answering, and idiom completion).

#### Text classification
Let us provide an annotation example for a text classification task (sentiment analysis). First, we read the task description, dataset fields, and prompt examples. Suppose we choose to write our own prompt template: `Tekst: {{review}}\nSentiment:`. We put the `review` dataset field in the double curly brackets and write the answer options \"positiv\" and \"negativ\" next to our prompt in parentheses, as shown below:

`Tekst: {{review}}\nSentiment: (positiv, negativ)`

Another way of formulating the template is to specify the target options directly in the prompt:
`{{review}}\nIs this review "positive" or \"negative"? (positive, negative)`

Here, we formulate the target labels in the parentheses the same way as in the prompt to ensure the formatting consistency. The parentheses are also used to differentiate between the prompt and the answer options formulated in natural language.

#### Multiple-choice tasks

In the case of multiple-choice question answering tasks, you may try to diversify the answer choice formulations. Consider an example from the OpenBookQA dataset provided in English for illustration purposes:

* question: `The sun is responsible for`
* choices: `["puppies learning new tricks", "children growing up and getting old", "flowers wilting in a vase", "plants sprouting, blooming and wilting"]`

The most straightforward way is to just use the question as the prompt, and rank the answer options as the continuation of the input, e.g.:
`Question: {{question}} (the choices are used as is)`

The choices can be formulated as options \"A\", \"B\", \"C\", \"D\", which will correspond to the target labels. Here, we put the target labels in the parentheses next to the prompt to specify how we formulate these labels.

`{{question}}\nA: {{choices[0]}}\nB: {{choices[1]}}\nC: {{choices[2]}}\nD: {{choices[3]}}\n\nIs the right answer A, B, C or D?  (A, B, C, D)`

Consider another example, where we can formulate the target labels in a different way:

`{{question}}\nOption A: {{choices[0]}}\nOption B: {{choices[1]}}\nOption C: {{choices[2]}}\nOption D: {{choices[3]}}\n\nIs the right answer A, B, C or D?  (Option A, Option B, Option C, Option D)`

Again, we formulate the target labels in the parentheses the same way as in the prompt to ensure the formatting consistency.

#### Natural language generation

In the natural language generation task, we can have an input based on one dataset field (e.g., a news article to be summarized, a question to be answered, or a sentence to be paraphrased) and multiple dataset fields (e.g., a question to be answered based on the context). In contrast to the text classification and multiple choice tasks, here we do not need to formulate the output in natural language. Consider an example for a text summarization task, which can have the following prompt involving one dataset field:

`{{article}}\nSummary:`

An example for a task requiring multiple dataset fields is an extractive question answering, where the model receives a question and a context as the input and is expected to generate an answer:

`Tekst: {{context}}\n\nSpørsmål:{{question}}\n\nSvar:`

Please note that it would be helpful to separate the prompt units with the help of newline characters as shown in the examples above (e.g., `\n` or `\n\n`).

### Sentiment analysis
#### Task description

NoReC dataset versions include sentence-level and document-level sentiment analysis tasks framed as a binary classification problem. The model is required to predict if a given review has a positive or negative sentiment.

#### Dataset fields 

**Sentence-level sentiment analysis**

* `sentence (str)`:  a review text
* `sentiment (str)`: target label (positive / negative)

**Document-level sentiment analysis**

* `document (str)`:  a review text
* `sentiment (str)`: target label (positive / negative)

**Prompt examples**

* `The following review expresses what sentiment? {{document}} (positive, negative)`
* `{{sentence}}\nIs this review "positive" or \"negative"? (positive, negative)`
* `{{sentence}} What sentiment does the writer express? (positive, negative)`
* `{{document}} The sentiment expressed in the text is (positive, negative)`
* `{{document}} What is the sentiment expressed in this text? (positive, negative)`

#### Norwegian Bokmål prompts

**Sentence-level sentiment analysis**

`The annotators write the prompts here.`

**Document-level sentiment analysis**

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

**Sentence-level sentiment analysis**

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

**Document-level sentiment analysis**

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`


### Text summarization

#### Task description
The text summarisation task requires the model to summarise a given text, preserving the main information and promoting factual consistency. 

#### Dataset fields 

* `article (str)`: the news article to be summarised.
* `summary (str)`: the article summary.

**Prompt examples**

* `Can you write an outline of the following article in a few points? {{article}}`
* `Summarise the article: {{article}}`
* `In 2 or 3 sentences, what are the main points one should remember from this news article?\n{{article}}`
* `Sum the following article in brief: {{article}}`
* `{{dialogue}}\n\nGiven the above dialogue, write a summary.`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Machine translation

#### Task description

We consider the following language pairs for the machine translation task: English ↔ Bokmål and English ↔ Nynorsk. Please take this into account when writing the prompts. 

#### Dataset fields 

* `source_sentence (str)`: the source sentence to be translated.
* `target_sentence (str)`: the gold standard translation.

**Prompt examples**

* `English: {{source_sentence}}\nBokmål:`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Grammatical error correction

#### Task description

The grammatical error correction task involves correcting grammatical errors in the input. 

#### Dataset fields 

* `source (str)`: the input text (a segment of a Wikipedia or a news article)
* `correction (str)`: a natural language question.

**Prompt examples**

* `Make minimal changes to the following text such that it is grammatically correct. {{source}}`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Idiom completion

#### Task description
The idiom completion task requires the model to continue a given segment of an idiomatic expression, correctly completing it.  

#### Dataset fields 

* `input (str)`: a beginning of an idiomatic expression.
* `continuation (str)`: a continuation of the idiomatic expression.

**Prompt examples**

* `{{input}}`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`



### Multiple-choice question answering (NorOpenBookQA)

#### Task description

The multiple-choice question answering based on the NorOpenBookQA dataset requires the model to select one out of four options to answer an elementary school question. Each question-answers pair might be accompanied with a corresponding science or commonsense knowledge fact, which supports the reasoning behind answering the question.

#### Dataset fields 

* `question (str)`: an elementary school question.
* `choices (list[str])`: a list of answer options.
* `fact (str)`: an originating common knowledge core fact associated with the question.

**Prompt examples**

* `{{question}}`
* `{{question}}\n\nChoices: {{choices}}\n\nWhich is the correct answer?`
* `{{question}}\nA: {{choices[0]}}\nB: {{choices[1]}}\nC: {{choices[2]}}\nD: {{choices[3]}}\n\nIs the right answer A, B, C or D?`
* `{{question}}\nPick the right answer from the list:\n– {{choices[0]}}\n– {{choices[1]}}\n– {{choices[2]}}\n– {{choices[3]}}\n\nIs the right answer A, B, C or D?`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Multiple-choice question answering (NorCommonSenseQA)

#### Task description

The multiple-choice question answering based on the NorCommonSenseQA dataset requires the model to select one out of five options to answer a commonsense reasoning question.

#### Dataset fields 

* `question (str)`: an elementary school question.
* `choices (list[str])`: a list of answer options.

**Prompt examples**

* `Given the following options, what do you think is the correct answer to the question below:\n\n{{question}}Options:\nA: {{choices[0]}}\nB: {{choices[1]}}\nC: {{choices[2]}}\nD: {{choices[3]}}\nE{{choices[4]}}`
* `Given the options below, select the most suitable answer for the following question:\n\n{{question}}Options:\nA: {{choices[0]}}\nB: {{choices[1]}}\nC: {{choices[2]}}\nD: {{choices[3]}}\nE{{choices[4]}}`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Multiple-choice question answering (Belebele)

#### Task description
The multiple-choice question answering based on Belebele requires the model to understand a given text and answer a question by selecting one out of four available options.

#### Dataset fields 

* `question (str)`: an elementary school question.
* `passage (str)`: a text passage from the FLORES dataset.
* `mc_answer1 (str)`: answer option A.
* `mc_answer2 (str)`: answer option B.
* `mc_answer3 (str)`: answer option C.
* `mc_answer4 (str)`: answer option D.

**Prompt examples**

* `P: {{passage}}\nQ: {{question}}\nA: {{mc_answer1}}\nB: {{mc_answer2}}\nC: {{mc_answer3}}\nD: {{mc_answer4}}\nAnswer:`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Multiple-choice question answering (NorTruthfulQA)

#### Task description

NorTruthfulQA measures whether a language model is truthful in generating answers to questions. The dataset comprises questions that span 38 categories, including health, law, finance and politics. Questions are crafted so that some humans would answer falsely due to a false belief or misconception.

The multiple choice question answering setup requires a model to select one option to answer a given question.

#### Dataset fields 

* `question (str)`: an elementary school question.
* `choices (list[str])`: a list of answer options.

**Prompt examples**
* `Q: {{question}}\nA:`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`

### Generative question answering (NorTruthfulQA)

#### Task description

The NorTruthfulQA generation setup requires a model to generate an answer to a given question. The model output is then compared to a set of correct answers and to a set of incorrect answers. E.g.:

* `question`: Why do chameleons change colors?
* `best_answer`: Chameleons change colors to regulate their temperature
* `correct answers`: ["Chameleons change colors to regulate their temperature", "Chameleons change colors to protect their territory from other chameleons", "Chameleons change color to attract a mate", "Chameleons change color to impersonate females" ]
* `incorrect answers`: ["Chameleons change colors to blend into any background", "Chameleons change colors to camouflage themselves against any background" ]

#### Dataset fields 

* `question (str)`: an input question.
* `best_answer (str)`: the best answer among the correct answers.
* `correct_answers (list[str])`: a list of correct answers.
* `incorrect_answers (list[str])`: a list of incorrect answers.


**Prompt examples**

* `Q: {{question}}\nA:`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`


### Generative question answering (NorQuAD)

#### Task description
NorQuAD is a SQuAD-style dataset for extractive question answering. The task is to extract a segment of a text, which corresponds to an answer to a given question. We formulate the task as a text generation problem, where the model receives the text and the question as the input and is expected to generate the answer.

#### Dataset fields 

* `context (str)`: the input text (a segment of a Wikipedia or a news article)
* `question (str)`: a natural language question.
* `answer (str)`: the answer to the given question, which corresponds to a context segment.

**Prompt examples**

* `{{context}}\n\nQ: {{question}}\nReferring to the passage above, the correct answer to the given question is`
* `{{context}}\n\nWith reference to the above context, {{question}}`
* `Refer to the passage below and answer the following question:\n\nPassage: {{context}}\n\nQuestion: {{question}}`
* `{{context}}\n\nQ: {{question}}\n\nA:`

#### Norwegian Bokmål prompts

`The annotators write the prompts here.`

#### Norwegian Nynorsk prompts

`The annotator adapts the Norwegian Bokmål prompts to Nynorsk here.`
