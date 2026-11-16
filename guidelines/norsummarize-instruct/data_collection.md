## Data collection guidelines

### Overview

Our annotation is run in iterations, and each iteration includes the following stages:
1. [Training](./data_collection.md): you practice to perform the annotation task for a small number of examples and get a feedback from the annotation curators.
2. [Annotation](./data_collection.md): you create prompt-response pairs from scratch by carefully following the guidelines.
3. [Peer-reviewing](./peer_reviewing.md): you judge the quality of the prompt-response pairs created by another annotators and make necessary edits.

These guidelines describe the training and annotation stages for creative abstractive summarization. 

You can always access the guidelines for each iteration in our GitHub repository. 

Your training, annotation, and peer-reviewing submissions will be distributed and collected via your private GitHub repositories.

### Annotation procedure

A step-by-step high-level annotation procedure is summarized as follows:

1. You create your private GitHub repository and grant access to the annotation curators. 
2. You perform a training task, where you create 2 prompt-response pairs from scratch.
3. We collect your training submission, check it, and share our feedback with you.
4. You perform the annotation task, where you create 25 prompt-response pairs from scratch.
5. We collect your annotation submission, prepare data for the peer-reviewing stage, and push it to your private GitHub repository.
6. You perform the peer-reviewing task as described [here](./peer_reviewing.md). 


### Definitions

**What is abstractive summarization?**

There are two main approaches to the summarization task: extraction and abstraction. 
Extractive summarization involves combining the input text segments into a summary, while abstraction involves generating novel text segments based on the information in the input text.

**What is a prompt-response pair?**

A prompt-response pair contains two key components: (1) a user prompt illustrating the user intent and (2) a response expected from a language model (LM). 

Below is an example of a prompt-response pair for the abstractive summarization task. 
Usually, a text to be summarized (```context```) is included in the prompt; during our annotation project, we are separating it from the prompt for our convenience.

|```prompt``` | ```context``` | ```response``` |
|---|---|---|
Jeg trenger et sammendrag av følgende dokument.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. Magnus Moan gikk seg opp fra 28.- til sjetteplass. Verdenscupleder Koivuranta var best også i hoppbakken og hadde 25 sekunders ledelse til David Zauner før langrennet. Moan mislyktes i bakken og var bare nummer 28 etter hoppingen. Han var to minutter bak Koivuranta, men gikk et sterkt langrenn og endte til slutt som nummer seks. I mål var han 44,3 sekunder bak den finske seierherren. Björn Kircheisen fra Tyskland ble nummer to. Han var 25,9 sekunder bak Koivuranta. Franske Jason Lamy Chappuis fulgte på tredjeplass. Jan Schmid ble nest beste norske på åttendeplass, mens Mikko Kokslien fulgte på 17.-plass.|Anssi Koivuranta fra Finland vant verdenscuprennet i kombinert i Granåsen på søndag. Norske Magnus Moan gikk opp 22 plasser og endte på 6. plass. Moan hadde et dårlig hopp, og endte på 28. plass etter hoppingen. Til slutt endte han 44,3 sekunder bak den finske seierherren. Nest beste nordmann var Jan Schmid, som endte på 8. plass.|

### Annotation task

#### Your task

Please do the following:

* Write a ```prompt``` for the abstractive summarization task. Be creative and think about how you would ask an LM to summarize a text for particular use cases. You can think about the response format (e.g., a bulleted or an enumerated list), the response length (e.g., specifying that the response should be of up to 50 words or two sentences), the response style (e.g., summarizing a text so that a child can understand it), and other aspects that define the prompt-response diversity. 
* Select a ```context``` that will be summarized by you. Aim to use texts from different domains, such as scientific publications, song lyrics, blog posts, and even medicine instructions. It is important to use sources published under open licenses, so you are asked to employ the [list](#recommended-sources-for-contexts) provided in these guidelines below. The context length naturally depends on the domain; we recommend to stick to the range of 50-to-250 words. However, if you feel you can produce high-quality and diverse summaries for longer contexts counting up to 500 words, you are welcome 😊.
* Write a ```response``` to the ```prompt``` and ```context```. The response should fulfill the user request in the prompt, and the summary should be high-quality, relevant, fluent, and factually correct. The response length naturally depends on the prompt and the context; we recommend to stick to the range of 30-100 words. Please do not overthink the length and keep it natural 😊. Please avoid extracting and combining parts of the context and rephrase, rewrite, and reformulate them instead. Think about a response you would ideally want to get from an LM. 
* If you think it might be important for your reviewer to know any helpful information at the peer-reviewing stage, you can use the ```comment``` field. In particular, you should always provide an URL to the context, but you can also specify other details for the reviewer to double-check.
* Double-check your ```prompt```, ```context```, and ```response```. Please pay attention to grammar, style, and misspellings. A tip is to read your prompt and response aloud to check if there are any inconsistencies.

Some annotation steps described above are interconnected, and their order is not strict. 
E.g., you might want to first find and read a ```context``` that helps diversify your examples, and then write the corresponding ```prompt``` and ```response```. 
Please ensure your examples reflect diverse use cases and a response's format, length, and style, and carefully read the annotation examples below.

#### How to submit your task?

##### Task submission

1. Please use any editing tool or application of your convenience to create the examples. 
2. Your examples should be submitted in the form of a tab-separated dataframe file (i.e., a table with the ```"\t"``` column separator) with the following columns: ```prompt```, ```context```, ```response```, and ```comment```. Please refer to [a submission file example](./submission_sample.tsv) and see an example below:

|```prompt``` | ```context``` | ```response``` | ```comment``` |
|---|---|---|---|
|Dette er en dummy prompt. | Dette er en dummy kontekst. | Dette er en dummy eksempelsvar. | Dette er en dummy kommentar.|
|Dette er en annen dummy prompt. | Dette er en annen dummy kontekst. | Dette er en annen dummy eksempelsvar. ||

3. Please use the following filename format and pay attention to the filenames during submission:
* Training: save your examples as ```summarization_training.tsv```.
* Annotation: save your examples as ```summarization_submission.tsv```.


### Annotation examples

Below, we provide annotation examples based on publicly available instruction-tuning datasets for English.

|```prompt``` | ```context``` | ```response``` | ```comment``` |
|---|---|---|---|
|I want this shortened into a sentence or two: | When Mr. Lorry had finished his breakfast, he went out for a stroll on the beach. The little narrow, crooked town of Dover hid itself away from the beach, and ran its head into the chalk cliffs, like a marine ostrich. The beach was a desert of heaps of sea and stones tumbling wildly about, and the sea did what it liked, and what it liked was destruction. It thundered at the town, and thundered at the cliffs, and brought the coast down, madly. The air among the houses was of so strong a piscatory flavour that one might have supposed sick fish went up to be dipped in it, as sick people went down to be dipped in the sea. A little fishing was done in the port, and a quantity of strolling about by night, and looking seaward: particularly at those times when the tide made, and was near flood. Small tradesmen, who did no business whatever, sometimes unaccountably realised large fortunes, and it was remarkable that nobody in the neighbourhood could endure a lamplighter. As the day declined into the afternoon, and the air, which had been at intervals clear enough to allow the French coast to be seen, became again charged with mist and vapour, Mr. Lorry’s thoughts seemed to cloud too. When it was dark, and he sat before the coffee-room fire, awaiting his dinner as he had awaited his breakfast, his mind was busily digging, digging, digging, in the live red coals. | After his breakfast, Mr. Lorry went for a walk on the beach. He seemed troubled later on when he was waiting for his dinner. | URL |
|Please summarize this content providing me with the key highlights in a bulleted list. | Add These Disease-Fighting Foods to Your Diet to Live a Longer Life While there are many factors—like environment, family history, and age—that can affect how long you live, there’s no shortage of research to back up the very strong connection between diet and longevity. “Fruits and vegetables contain compounds that reduce the risk of heart disease and stroke, some cancers, chronic respiratory diseases, diabetes, and obesity, which is one of the reasons health experts are constantly trying to encourage people to eat more of them,” says Frances Largeman-Roth, RDN, author of Eating in Color. But fruits and veggies aren’t the only foods you should be nourishing your body with to play your best defense against diseases. Here is a laundry list of the most nutritious foods to add to your diet to improve your longevity and protect your health. Legumes Beans, peas, lentils, peanuts, and chickpeas are excellent sources of fiber and plant-based protein to stabilize blood sugar and keep cravings at bay. They also help nourish a healthy microbiome. “A healthy gut microbiome is known to help regulate inflammation, lower blood lipids (cholesterol) and regulate immune function,” explains Suzanne Dixon, RD, a dietitian with The Mesothelioma Center at Asbestos.com. She recommends including at least five serving of legumes in your diet weekly. | • Research shows that diet can have a great impact on how long you live and your overall health.  • Fruits and vegetables are known to reduce the risk of various health conditions and diseases.  • While they are helpful, fruits and vegetables are not the only foods that can help you fight against diseases. • Legumes such as beans, peas, and lentils are great sources of fiber and plant-based protein which can stabilize blood sugar and curb your cravings. | URL |
|Summarize why Fall Guys is so popular in just a couple of sentences. Here is the text to draw your summary from. |Another factor in the game’s favor is its accessibility. Players only need to know three controls — running, jumping and grabbing — meaning it’s much easier for newcomers to learn how to play. This, combined with consumers’ search for escapism during the coronavirus pandemic, has helped broaden the game’s appeal. “Fall Guys has really flourished in this environment because, at its core, it is a simple, family-friendly game,” Louise Shorthouse, games industry analyst at Omdia, told CNBC. “Unlike traditional, violent battle royale titles, Fall Guys appeals to all ages, and is approved by parents.” Then there’s the virality of Fall Guys on streaming platforms like Amazon’s  Twitch and Google’s  YouTube, as well as social media. The game was watched more than Fortnite across Twitch, YouTube and Facebook last month, according to esports intelligence firm Stream Hatchet, racking up 118.5 million hours of watchtime versus Fortnite’s 106.3 million between Aug. 4 and Aug. 28. |Fall Guys' success can largely be attributed to the simplistic, family-friendly style of gameplay that doesn't feature complex controls or mechanics. The time of release is also a significant factor in the game's popularity, with the game serving as a welcome relief from stress caused by the coronavirus pandemic, broadening its scope of influence. The game has also been massively popular on streaming sites such as Twitch and YouTube, as well as social media sites like Facebook. | URL |
|Explain to me what the main subject of this article is: | California officials are investigating the disappearance of 30 tons (61,000 pounds) of ammonium nitrate from a railcar crossing the Midwest. Ammonium nitrate, which is used as fertilizer, can also be used to create explosives and was a key component in the homemade bomb used in the 1995 attack on the Murrah Federal Building in Oklahoma City. The chemical was shipped by train from Cheyenne, Wyoming last month but when it arrived in the Mojave Desert two weeks later, its holding car was empty and the nitrate was gone. Dyno Nobel, the company responsible for shipping the ammonium nitrate is a leader in commercial explosives and filed a report with the federal National Response Center (NRC) on May 10 and an investigation is currently underway. The Federal Railroad Administration, the California Public Utilities Commission, Union Pacific, and Dyno Nobel are looking into the disappearance, and the railcar is being transported back to Wyoming to undergo a thorough inspection. It is unclear how or when the chemical disappeared during transit, but Dyno Nobel told KQED News, “The railcar was sealed when it left the Cheyenne facility, and the seals were still intact when it arrived in Saltdale. The initial assessment is that a leak through the bottom gate on the railcar may have developed in transit.” |This article covers an ongoing investigation into the disappearance of 30 tons of ammonium nitrate that was being transported from Wyoming to the Mojave Desert by railcar. While mainly used as a fertilizer this substance can also be used as an explosive which raises security concerns.  Several agencies, including the California Public Utilities Commission, are investigating.| URL |
|From the given paragraph, write some points on Church. | A church, church building or church house is a building used for Christian worship services and other Christian religious activities. The earliest identified Christian church is a house church founded between 233 and 256. From the 11th through the 14th centuries, there was a wave of church construction in Western Europe. Sometimes, the word church is used by analogy for the buildings of other religions. Church is also used to describe the Christian religious community as a whole, or a body or an assembly of Christian believers around the world. |1. A church, church building, or church house is a structure used for Christian religious activities such as worship ceremonies. 2. A home church that was established between 233 and 256 is the earliest known Christian congregation. 3. In Western Europe, there was a wave of church building from the 11th through the 14th centuries. 4. The word "church" is occasionally used as an analogy for the structures of other religions. 5. The word "church" is also used to refer to the global body of Christian believers as well as the Christian religious community as a whole.' |URL|


### Recommended sources for contexts

* [Norwegian Wikipedia](https://no.wikipedia.org/wiki/Portal:Forside)
* [Norwegian Colossal Corpus](https://huggingface.co/datasets/NbAiLab/NCC) (compressed files, one document per line):
    * [Openly available Norwegian books](https://data.hplt-project.org/NCC/books_no.txt.gz)
    * [Openly available Norwegian newspapers](https://data.hplt-project.org/NCC/newspaper_ocr_no.txt.gz) (caution: 4.4 GB of data, 9 million articles)
    * [Openly available public documents from Språkbanken: maalfrid_fylkesmannen](https://data.hplt-project.org/NCC/maalfrid_fylkesmannen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_regjeringen](https://data.hplt-project.org/NCC/maalfrid_regjeringen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_uio](https://data.hplt-project.org/NCC/maalfrid_uio_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_ssb](https://data.hplt-project.org/NCC/maalfrid_ssb_no.txt.gz)
