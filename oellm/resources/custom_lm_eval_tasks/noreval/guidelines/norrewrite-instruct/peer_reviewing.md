## Peer-reviewing guidelines for Creative Rewriting

### Overview

Our annotation is run in iterations, and each iteration includes the following stages:
1. [Training](./data_collection.md): you practice to perform the annotation task for a small number of examples and get a feedback from the annotation curators.
2. [Annotation](./data_collection.md): you create 25 prompt-response pairs from scratch by carefully following the guidelines.
3. [Peer-reviewing](./peer_reviewing.md): you judge the quality of the prompt-response pairs created by other annotators and make necessary edits.

These guidelines describe the training and annotation stages for creative rewriting.

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
* Check out the ```comment```. If the ```context``` is created from scratch, please proceed to the next step. If the ```context``` is taken from some data source, and the source does *not* belong to the [list of recommended resources](#recommended-sources-for-contexts) or might have potential licensing issues based on the information provided in the URLs, please label the example as ```S``` (stands for ```source```) in the ```label``` column.
* Judge the overall quality of the example, paying special attention to:
    * the *relevance* of ```response``` to ```prompt``` & ```context```. Does the response complete the user request and correspond to the intended format, length, style, and other properties specified in the ```prompt```? 
    * the *factual consistency* between ```response```, ```prompt```, and ```context```. Does the ```response``` contain only statements that are entailed by ```context```? Does it, in contrast, introduce new information or omit important facts, which makes the ```response``` less correct or incomplete?
    * the *natural flow* of each component. Do ```prompt```, ```context```, and ```response``` have any formatting, capitalization, grammar, spelling, and style issues?
    * the *originality* of the text is low. Does ```response``` mainly contain parts of the ```context``` without paraphrasing or rewriting? 
    * any other issues that reduce the quality.
* If you find any insignificant issues, please edit the ```prompt```, ```context```, and ```response``` in *the same* columns. Recall that everyone has their own style of writing; please make the changes only if you feel they improve the overall quality of the example.
* However, if the overall quality of the example is unacceptable (e.g., it has too many issues listed above and it requires significant changes, which might take more than 5 minutes), please label the example as ```D``` (stands for ```discard```) in the ```label``` column. Please briefly describe why you choose to discard the example in the ```explanation``` column. 
* Double-check the ```prompt```, ```context```, and ```response```. A tip is to read the example aloud to check for  inconsistencies.
* Ensure you have labelled the example as ```S``` or ```D``` and provided a brief explanation for why the example should be discarded (if needed). If the example should have both labels, assign ```S, D``` in the ```label``` column. Note that if the example is good or you have made some changes, you *do not need* to assign any label.


#### How to submit your task?

##### Task submission

1. Please use any editing tool or application of your convenience to peer-review the examples.
2. We provide the examples for peer-reviewing in the form of a tab-separated dataframe file (i.e., a table with the ```"\t"``` column separator), which can be accessed as ```rewriting_{uio_username}.tsv``` (```uio_username``` is your UiO GitHub username). The dataframe has the following columns: ```prompt```, ```context```, ```response```, ```comment```, ```label```, and ```explanation```. 
3. Please submit your peer-reviewed and edited examples in the form of a *separate* tab-separated dataframe file with the same column names. Please refer to [a submission file example](./peer_review_submission_sample.tsv) and see an example below:

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` | ```explanation```|
|---|---|---|---|---|---|
|Dette er en dummy prompt. | Dette er en dummy kontekst. | Dette er en dummy eksempelsvar. | Dette er en dummy kommentar.| D | Detter en dummy forklaring.|
|Dette er en annen dummy prompt. | Dette er en annen dummy kontekst. | Dette er en annen dummy eksempelsvar. | | |

4. Please use the following filename format and save your reviewed examples as ```rewriting_review.tsv```
5. Recall that at this final stage you will have four dataframes in your repository:
* Training: ```rewriting_training.tsv```.
* Annotation: ```rewriting_submission.tsv```.
* Peer-reviewing: ```rewriting_{uio_username}.tsv``` and ```rewriting_review.tsv```. Please note that the ```rewriting_{uio_username}.tsv``` should remain unchanged, while ```rewriting_review.tsv``` will contain your reviewed examples with the neccessary edits and labels.

### Annotation examples

### Nice quality examples

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` | ```explanation``` |
|---|---|---|---|---|---|
|Kan du omskrive dette slik at det blir lett for en 7-åring å skjønne? Gjerne eventyr-aktig.|Einar Henry Gerhardsen (født Olsen; 1897–1987) var en norsk politiker som representerte Arbeiderpartiet. Han var Norges statsminister fra 1945 til 1951, fra 1955 til 1963 og fra 1963 til 1965. I Norge er Gerhardsen kjent under kallenavnet «landsfaderen», og han er omtalt som en av hovedarkitektene bak den norske velferdsstaten som vokste frem etter andre verdenskrig.|I 1897 ble det født et barn som ble kalt Einar Henry Olsen. Etter hvert ble han kjent under navnet Einar Gerhardsen. Han skulle vokse opp til å bli kjent i Norge for å være dyktig til å styre over landet. Etter andre verdenskrig var det han som var øverste leder etter kongen i Norge, og mange sier han var helt sentral for at Norge er det flotte landet det er i dag, med lite fattigdom og like muligheter for alle. Folk kalte ham "landsfaderen".|Første avsnitt fra https://no.wikipedia.org/wiki/Einar_Gerhardsen|||
|Forklar prinsippet om utjevningsmandater for nordmenn som aldri har stemt før.|Det politiske valgsystemet ved stortingsvalg har flere ganger ført til en skjev fordeling mellom partiene. Dette skyldes at valgoppgjøret skjer fylkesvis, slik at hvis 4–5 mandater fra et valgdistrikt (fylke) skal fordeles på 7–8 partier, kan det ikke bli en mandatfordeling som fullt ut representerer de stemmeberettigedes stemmer. Flere partier kan derfor ikke bli representert. For å rette på dette ble det i 1989 innført en ordning med åtte utjevningsmandater. For at et parti skal kunne få et utjevningsmandat må det ha passert en sperregrense, det vil si oppnådd en oppslutning på minst fire prosent på landsbasis (Grunnloven § 59). Det er derimot ingen sperregrense for valget av distriktsrepresentantene. Fra og med stortingsvalget 2005 ble det innført ett utjevningsmandat fra hvert fylke, og den gamle ordningen med åtte utjevningsmandater fordelt på hele riket falt bort. Det velges dermed 169 stortingsrepresentanter: 150 som distriktsrepresentanter og 19 som utjevningsrepresentanter.|Norge er et stort og langt land der folk bor spredt i mange distrikter. Når vi stemmer til Stortinget, blir de fleste plassene fordelt ut fra hvor mange stemmer hvert parti får i hvert fylke. Men dette kan gjøre det urettferdig for små partier, fordi de kanskje ikke får nok stemmer i ett fylke, selv om de har mange stemmer i hele landet. For å gjøre det mer rettferdig har vi noe som heter utjevningsmandater. Dette er ekstra plasser i Stortinget som små partier kan få hvis de har fått minst 4 % av stemmene i hele Norge. Fra 2005 har hvert fylke fått ett sånt mandat. Dette hjelper små partier med å bli representert og gjør at Stortinget bedre viser hva folk har stemt på.|https://no.wikipedia.org/wiki/Utjevningsmandat|||
|Skriv en tekst i en dagbok til en tenåring med skilte foreldre basert på informasjonen i artikkelen|Barns familier har endret seg kraftig de siste 20-30 årene. I dag har mange barn foreldre som ikke er gifte. Det var ikke vanlig før. Flere barn bor sammen med bare én forelder, og da ofte mor, i kortere eller lengre perioder av barndommen. Barn opplever samlivsbrudd - at mor og far flytter fra hverandre - i større grad enn tidligere. Dette kan føre med seg flere sett av familier å forholde seg til, helsøsken, halvsøsken og stesøsken, stemødre, stefedre og flere sett med besteforeldre. I dag, når antall samlivsbrudd er høyere enn noen gang før, er spørsmålet om hvilke konsekvenser dette har for barna mer aktuelt enn noen gang tidligere.|Foreldrene mine er skilt. Jeg bor sammen med moren min nå. Det har jeg gjort en stund. Når jeg er på besøk hos vennene mine, er det ofte slik at de også kun bor med en forelder. Mamma fant seg en ny mann etter et par år, og han hadde en datter fra før, så hun er nå min halvsøster. Stefaren min og halvsøsteren min er greie, men jeg skulle ønske jeg kunne reise tilbake til tiden mamma og pappa var sammen.||||


### Task formulation

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` | ```explanation``` |
|---|---|---|---|---|---|
|Skriv et fanbrev til Stanley Kubrick basert på følgende informasjon:|Stanley Kubrick (1928–1999) var en amerikansk filmregissør, filmprodusent og fotograf. Han laget 13 spillefilmer og er blant annet kjent for 2001: En romodyssé, A Clockwork Orange, Ondskapens hotell og Full Metal Jacket. Han er ansett for å være blant de ledende filmskapere på 1900-tallet. Kubrick laget filmer i mange sjangre og stiler der hver ny film var helt ulik den forrige og filmene brøt ofte konvensjonene. Han omtales som en perfeksjonist som laget relativt få, men gjennomarbeidete filmer i sin lange karriere. Filmene hans ble til sammen tildelt åtte Oscar, men ingen for beste regi. Kubrick vokste opp i et jødisk miljø i New York og var selvlært som fotograf og filmskaper. Allerede som tenåring laget han fotoreportasjer for fotomagasinet Look og hans kunnskap om fotografering preget hans senere spillefilmer både teknisk og kunstnerisk. Han arbeidet først i USA, men ble misfornøyd med filmindustrien der og flyttet til England for godt. Åtte av tretten spillefilmer ble laget etter at han flyttet til Storbritannia. Kubrick døde seks dager etter at den endelige versjonen av Eyes Wide Shut var klar for kino. Kubricks spillefilmer etter 1956 er basert på romaner og noveller, ikke på manus skrevet spesielt for film, og blant disse er Lolita. Alle hans filmer ble laget med sikte på bred distribusjon, og etter 1955 ble produksjonene finansiert av de store filmselskapene. Med unntak av Spartacus hadde han selv kontroll over alle ledd i produksjonen fra ide til ferdig film. Hans arbeidsform var omstendelig, der han selv gjorde et omfattende forarbeid som strakk seg over år eller i noen tilfeller tiår. Tematisk omhandler filmene blant annet krig og militærvesen, ekteskap og familieforhold, science fiction, vold og kriminalitet. Filmene har et kjølig, distansert preg med innslag av svart humor, og Kubrick hadde en viss interesse for det uvanlige og bisarre. Filmene er kjent for både realisme i fremstilling og tydelige formelle trekk. Kubricks filmer er særlig kjent for nyskapende og gjennomarbeidet foto, og for bruk av musikk og levende bilder som helhet.|Kjære Stanley Kubrick,\n Som stor fan av deg og filmene dine, ønsker jeg å takke deg for ditt mesterlige bidrag til filmens verden. Hver gang jeg ser dem imponerer du med din nyskapning, perfeksjon og bredde. For meg er ikke filmene dine bare filmer - de er kunstverk. Jeg liker spesielt godt måten du forteller historiene dine gjennom foto og musikk. Detaljnivået på arbeidet ditt er også helt utrolig. Jeg har ikke funnet den minste feil - og jeg har lett godt! Hvis jeg en dag får muligheten til å lage film, kommer jeg til å se til dine filmer for inspirasjon. Jeg skulle så inderlig ønske vi fikk se mer av deg. Uansett ønsker jeg å takke deg for de 13 fantastiske filmene du har skapt. De vil bli stående i filmens historie for alltid. Med ærbødig hilsen, din største beundrer|https://no.wikipedia.org/wiki/Stanley_Kubrick|D|The prompt addresses the task of conditional generation rather than rewriting an input text.|

**Brief explanation**: The ```response``` is not likely to represent a rewritten version of the ```context```. The gap between them is a bit large, and a lot of information in the ```context``` is omitted. For example, the ```prompt``` could be "Rewrite this Wikipedia passage about Stanley Kubrick as if his fan wrote it" more suitable. 


### Other 

|```prompt``` | ```context``` | ```response``` | ```comment``` | ```label``` | ```explanation``` |
|---|---|---|---|---|---|
|Dum ned denne teksten|Lysets forplantning er rettlinjet med meget stor hastighet, 299 792 458 m/s (ca. 300 000 km/s) i vakuum og viser en rekke karakteristiske fenomener, såsom absorpsjon, refleksjon, brytning, dobbeltbrytning, interferens og diffraksjon samt polarisasjon. Den elektromagnetiske lysteorien (Maxwell, Lorentz m.fl.), sier at lyset er en elektromagnetisk bølgebevegelse. Fotoelektriske fenomener og strukturen hos ulike spektra m.m. viser på den annen side at lyset også har kvantenatur (Planck, Einstein m.fl.).|Lyset beveger seg i en rettlinje 300 millioner meter per sekund hvis den beveger seg gjennom et vakum. Den kan oppføre seg som en bølge hvor den reflekteres av vegger, deler seg, beveger seg rundt ting, sammtidig som den kan oppføre seg som en fysiskt ting. Når lys er en fysisk ting kaller vi det et foton. Fotoner har egenskaper som gjør at når de treffer andre ting spretter det ting vekk.|https://no.wikipedia.org/wiki/Lys|D|New content (scientific terms) introduced in the response, despite the instruction to simplify the text.|


**Brief explanation**: While the annotator has made a good effort to simplify the ```context```, the ```response``` still primarily contains terms and complex sentence structure. Moreover, it introduces a new term not mentioned in the input text ("foton") and spelling errors. The reviewer should consider rephrasing the ```response``` or discarding the example by assigning the ```D``` label.


### Recommended sources for contexts

* [Norwegian Wikipedia](https://no.wikipedia.org/wiki/Portal:Forside)
* [Norwegian Colossal Corpus](https://huggingface.co/datasets/NbAiLab/NCC) (compressed files, one document per line):
    * [Openly available Norwegian books](https://data.hplt-project.org/NCC/books_no.txt.gz)
    * [Openly available Norwegian newspapers](https://data.hplt-project.org/NCC/newspaper_ocr_no.txt.gz) (caution: 4.4 GB of data, 9 million articles)
    * [Openly available public documents from Språkbanken: maalfrid_fylkesmannen](https://data.hplt-project.org/NCC/maalfrid_fylkesmannen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_regjeringen](https://data.hplt-project.org/NCC/maalfrid_regjeringen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_uio](https://data.hplt-project.org/NCC/maalfrid_uio_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_ssb](https://data.hplt-project.org/NCC/maalfrid_ssb_no.txt.gz)
