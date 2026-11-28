## Peer-reviewing guidelines 

### Overview

Our annotation is run in iterations, and each iteration includes the following stages:
1. [Training](./data_collection.md): you practice to perform the annotation task for a small number of examples and get a feedback from the annotation curators.
2. [Annotation](./data_collection.md): you create prompt-response pairs from scratch by carefully following the guidelines.
3. [Peer-reviewing](./peer_reviewing.md): you judge the quality of the prompt-response pairs created by other annotators and make necessary edits.

These guidelines describe the peer-reviewing stage for creative abstractive summarization. 

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

### Annotation task

#### Your task

Please do the following:
* Carefully read each given example created by other annotators (```prompt```, ```context```, ```response```, and ```comment```).
* Check out the ```comment```. If the ```context```'s source does *not* belong to the [list of recommended resources](#recommended-sources-for-contexts) or might have potential licensing issues based on the information provided in the URLs, please label the example as ```S``` (stands for ```source```) in the ```label``` column. Proceed to the next steps described below.
* Judge the overall quality of the example, paying special attention to:
    * the *relevance* of ```response``` to ```prompt``` & ```context```. Does the response complete the user request and correspond to the intended format, length, style, and other properties specified in the ```prompt```? 
    * the *factual consistency* between ```response```, ```prompt```, and ```context```. Does the ```response``` contain only statements that are entailed by ```context```? Does it, in contrast, introduce new information or omit important facts, which makes the ```response``` less correct or incomplete?
    * the *natural flow* of each component. Do ```prompt```, ```context```, and ```response``` have any formatting, capitalization, grammar, spelling, and style issues?
    * the *originality* of the text is low. Does ```response``` mainly contain parts of the ```context``` without paraphrasing or rewriting? 
    * any other issues that reduce the quality.
* If you find any insignificant issues, please edit the ```prompt```, ```context```, and ```response``` in *the same* columns. Recall that everyone has their own style of writing; please make the changes only if you feel they improve the overall quality of the example.
* If the overall quality of the example is unacceptable (e.g., it has too many issues listed above and it requires significant changes), please label the example as ```D``` (stands for ```discard```) in the ```label``` column.
* If the ```response``` is more *extractive* (i.e., created by combining the ```response``` segments into a summary) than *abstractive* (i.e., created by generating novel text segments based on the information in ```response```), please make necessary edits to make it *abstractive* via paraphrasing and rewriting. However, if it requires significant changes, please label the example as ```D``` (stands for ```discard```) in the ```label``` column.
* Double-check the ```prompt```, ```context```, and ```response```. A tip is to read the example aloud to check for  inconsistencies.
* Ensure you have labelled the example as ```S``` or ```D``` if needed. If the example should have both labels, assign ```S, D``` in the ```label``` column. Note that if the example is good or you have made some changes, you *do not need* to assign any label.


#### How to submit your task?


##### Task submission

1. Please use any editing tool or application of your convenience to peer-review the examples.
2. We provide the examples for peer-reviewing in the form of a tab-separated dataframe file (i.e., a table with the ```"\t"``` column separator), which can be accessed as ```summarization_{uio_username}.tsv``` (```uio_username``` is your UiO GitHub username). The dataframe has the following columns: ```prompt```, ```context```, ```response```, ```comment```, and ```label```. 
3. Please submit your peer-reviewed and edited examples in the form of a *separate* tab-separated dataframe file with the same column names. Please refer to [a submission file example](./peer_review_submission_sample.tsv) and see an example below:

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
|Dette er en dummy prompt. | Dette er en dummy kontekst. | Dette er en dummy eksempelsvar. | Dette er en dummy kommentar.| D |
|Dette er en annen dummy prompt. | Dette er en annen dummy kontekst. | Dette er en annen dummy eksempelsvar. | |

4. Please use the following filename format and save your reviewed examples as ```summarization_review.tsv```
5. Recall that at this final stage you will have four dataframes in your repository:
* Training: ```summarization_training.tsv```.
* Annotation: ```summarization_submission.tsv```.
* Peer-reviewing: ```summarization_{uio_username}.tsv``` and ```summarization_review.tsv```. Please note that the ```summarization_{uio_username}.tsv``` should remain unchanged, while ```summarization_review.tsv``` will contain your reviewed examples with the neccessary edits and labels.


### Annotation examples

### Relevance

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer denne teksten med maks 3 setninger.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. Magnus Moan gikk seg opp fra 28.- til sjetteplass. Verdenscupleder Koivuranta var best også i hoppbakken og hadde 25 sekunders ledelse til David Zauner før langrennet. Moan mislyktes i bakken og var bare nummer 28 etter hoppingen. Han var to minutter bak Koivuranta, men gikk et sterkt langrenn og endte til slutt som nummer seks. I mål var han 44,3 sekunder bak den finske seierherren. Björn Kircheisen fra Tyskland ble nummer to. Han var 25,9 sekunder bak Koivuranta. Franske Jason Lamy Chappuis fulgte på tredjeplass. Jan Schmid ble nest beste norske på åttendeplass, mens Mikko Kokslien fulgte på 17.-plass.|Anssi Koivuranta fra Finland vant verdenscuprennet i kombinert i Granåsen på søndag. Norske Magnus Moan gikk opp 22 plasser og endte på 6. plass. Moan hadde et dårlig hopp, og endte på 28. plass etter hoppingen. Til slutt endte han 44,3 sekunder bak den finske seierherren. Nest beste nordmann var Jan Schmid, som endte på 8. plass.|||

**Brief explanation**: Response does not conform to the instruction provided in the prompt, it should be maximum 3 sentences long. You would shorten it by saving the edited version in the corresponding column (```response```).


### Factual consistency

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer denne teksten med maks 3 setninger.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. Magnus Moan gikk seg opp fra 28.- til sjetteplass. Verdenscupleder Koivuranta var best også i hoppbakken og hadde 25 sekunders ledelse til David Zauner før langrennet. Moan mislyktes i bakken og var bare nummer 28 etter hoppingen. Han var to minutter bak Koivuranta, men gikk et sterkt langrenn og endte til slutt som nummer seks. I mål var han 44,3 sekunder bak den finske seierherren. Björn Kircheisen fra Tyskland ble nummer to. Han var 25,9 sekunder bak Koivuranta. Franske Jason Lamy Chappuis fulgte på tredjeplass. Jan Schmid ble nest beste norske på åttendeplass, mens Mikko Kokslien fulgte på 17.-plass.|Anssi Koivuranta fra Finland vant verdenscuprennet i kombinert i Granåsen på søndag. Norske Magnus Moan gikk opp 22 plasser og endte på 6. plass. Moan hadde et dårlig hopp, og endte på 30. plass etter hoppingen.|||

**Brief explanation**: Response contains a factual error (Moan got 28th not 30th place). This error can be fixed, and you would make the edits in the corresponding column (```response```).

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer denne teksten med maks 3 setninger.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. Magnus Moan gikk seg opp fra 28.- til sjetteplass. Verdenscupleder Koivuranta var best også i hoppbakken og hadde 25 sekunders ledelse til David Zauner før langrennet. Moan mislyktes i bakken og var bare nummer 28 etter hoppingen. Han var to minutter bak Koivuranta, men gikk et sterkt langrenn og endte til slutt som nummer seks. I mål var han 44,3 sekunder bak den finske seierherren. Björn Kircheisen fra Tyskland ble nummer to. Han var 25,9 sekunder bak Koivuranta. Franske Jason Lamy Chappuis fulgte på tredjeplass. Jan Schmid ble nest beste norske på åttendeplass, mens Mikko Kokslien fulgte på 17.-plass.|Anssi Koivuranta fra Finland vant verdenscuprennet i kombinert i Granåsen på søndag.||D|

**Brief explanation**: Response omits a central piece of information since it has no mention of Moan, the Norwegian jumper in focus in the article. We can consider editing the response in the ```response``` column or discarding this example by assigning the label ```D``` in the ```label``` column.

### Natural flow

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer denne teksten med maks 3 setninger.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. |Anssi Koivuranta fra Fniland vant verdenscuprennet i kombnert i Granåsen på søndag. |||

**Brief explanation**: Contains spelling errors, which can be corrected.


### Low originality (extractive summaries)

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer denne teksten med 2 setninger.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. |Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. Moan gikk opp 22 plasser.||D|

**Brief explanation**: Summary only copies parts of the source text instead of reformulating, should be rephrased. We can consider editing the response in the ```response``` column or discarding this example by assigning the label ```D``` in the ```label``` column.


|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer denne teksten med 2 setninger.|Moan gikk opp 22 plasser. Endte på sjetteplass. Finske Anssi Koivuranta tok en suveren seier i verdenscuprennet i kombinert i Granåsen søndag. |Finske Anssi Koivuranta vant suverent i verdenscuprennet i kombinert i Granåsen søndag. Moan gikk opp 22 plasser.||D|

**Brief explanation**: Parts of the summary copies from the source text instead of reformulating, should be rephrased.

### Source under a restrictive license

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` |
|---|---|---|---|---|
Oppsummer hovedbudskapet i sangteksten med noen få setninger.|[Vers 1: Chirag] De sender meg på minneseremoni, for å stå på Nobel Jeg er det største forbildet som går på to ben Jeg får lyst på MDMA, og å gå på bordell Hver gang dere kaller meg for rollemodell Ingen her har leid meg til toppen av fjell Jeg runker og vasker denne kroppen her selv, yes Mitt legeme, mine valg, yes Og du vil'ke gå i detalj, nei Hva du synes er irrelevant Så lenge solen skinner inn i tomme slott i mitt lille land Midt i blant Canada Goose og au pairer Går pappa på jobb i vernesko og Blåkläder Det er her gåselever'n bare vokser på trærne Og vi payer toppskatt til vi dæver [Chorus: Chirag] Og sola skinner inn i store slott som ingen bor i Og sola skinner inn i store slott som ingen bor i Og sola skinner inn i de store slott som ingen bor i Og sola skinner inn i store slott som ingen bor i [Vers 2: Magdi] Jeg får stå på Nobel Jeg får gå på Senkveld Og jeg får lyst til å putte cannabis i boller fra Shell Hver gang dere kaller meg for rollemodell Men jeg er stolt av meg selv For Bård er våken om kveld’n Og bumper låter jeg skrev fra jeg var 15 til nå Og Bård er 15 i morgen Bård skjønner at det er mennesker i sang Slips skjønner bare penger eller pang Penger, penger, penger, penger, pang, pang, pang Ring en annen hvis du trenger mannekenger med no' sang Jeg er’ke han, "Dans neger, negerdans Kast glans, mine venner, dere selger", da vi vant Var jeg varmt, jeg er’ke relevant lenger, eller hva? For det er kaldt i mitt lille land [Chorus: Chirag] Og sola skinner inn i store slott som ingen bor i Og sola skinner inn i store slott som ingen bor i Og sola skinner inn i de store slott som ingen bor i Og sola skinner inn i store slott som ingen bor i [Bro: Magdi] Dine hus, hotell, slott brenner Dine hus, hotell, slott brenner Dine hus, hotell, slott brenner, som ingen bor i Dine hus, hotell, slott brenner Dine hus, hotell, slott brenner Dine hus, hotell, slott brenner, som ingen bor i [Chorus: Chirag] Og sola skinner inn i store slott som ingen bor i Og sola skinner inn i store slott som ingen bor i Og sola skinner inn i de store slott som ingen bor i Og sola skinner inn i store slott som ingen bor i |Sangteksten omhandler problematikken med å bli omtalt som en rollemodell. Chirag og Magdi uttrykker i teksten at de er ukomfortable med denne merkelappen, ved å hinte til selvdestruktiv oppførsel, i håp om å bli kvitt den. I refrenget brukes en linje fra sangen "Tusen Etasjer Høi" laget av bandet Dumdum Boys for å formidle at rollemodeller ikke er ekte personer, men heller et produkt av oppbygde fantasier basert på det man ser utenifra.|https://genius.com/Karpe-hus-hotell-slott-brenner-lyrics|S|

**Brief explanation**: The context is borrowed from the genius.com website, and their [terms of use](https://genius.com/static/terms) explicitly prohibit re-distribution of their data. We assign label ```S```to this example in the ```label``` column.


### Recommended sources for contexts

* [Norwegian Wikipedia](https://no.wikipedia.org/wiki/Portal:Forside)
* [Norwegian Colossal Corpus](https://huggingface.co/datasets/NbAiLab/NCC) (compressed files, one document per line):
    * [Openly available Norwegian books](https://data.hplt-project.org/NCC/books_no.txt.gz)
    * [Openly available Norwegian newspapers](https://data.hplt-project.org/NCC/newspaper_ocr_no.txt.gz) (caution: 4.4 GB of data, 9 million articles)
    * [Openly available public documents from Språkbanken: maalfrid_fylkesmannen](https://data.hplt-project.org/NCC/maalfrid_fylkesmannen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_regjeringen](https://data.hplt-project.org/NCC/maalfrid_regjeringen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_uio](https://data.hplt-project.org/NCC/maalfrid_uio_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_ssb](https://data.hplt-project.org/NCC/maalfrid_ssb_no.txt.gz)
