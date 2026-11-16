## Data collection guidelines for Creative Rewriting

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

### Definitions

**What is creative rewriting?**

Creative rewriting involves creating a new version of a given text through, e.g.: (i) changing the tone, style, or genre; (ii) simplifying complex content; (iii) enhancing readability; (iv) adapting the content for different age groups. In general, it includes but is not limited to text simplification, style transfer, and paraphrasing. Try to keep it "as Norwegian as possible": that is, you are encouraged to make allusions to Norwegian culture, social life, jokes, memes, etc. The same applies to Norwegian language, e.g. different written norms, dialects, etc.

**What is a prompt-response pair?**

A prompt-response pair contains two key components: (1) a user prompt illustrating the user intent and (2) a response expected from a language model (LM). 

Below is an example of a prompt-response pair for the creative rewriting task in English. Usually, a text to be rewritten (```context```) is included in the prompt; during our annotation project, we are separating it from the prompt for our convenience.

|```prompt``` | ```context``` | ```response``` |
|---|---|---|
Rewrite this paragraph in the 3rd person.|Thinking no one could hear me as I loaded a UPS tractor-trailer, I began to whistle. I was getting into it when a coworker in the next trailer poked his head in. “You know, I always used to wish I could whistle,” he said. “Now I just wish you could.”|Thinking no one could hear him, he loaded a UPS tractor-trailer as he began to whistle. He was getting into it when a coworker in the next trailer poked his head in. “You know, I always used to wish I could whistle,” the co-worker said. “Now I just wish you could.”|

### Annotation task

#### Your task

Please do the following:

* Write a ```prompt``` for the creative rewriting task. Be creative and think about how you would ask an LM to rewrite a text for particular use cases. You can think about the response length (e.g., making a text longer or shorter), the response style (e.g., making the language more engaging and persuasive), the target audience (e.g., rewriting the text so that a child can understand it), the response format and content complexity (e.g., changing complex terminology into simpler words and utilizing bullet points to make the content easier to read), and other aspects that define the prompt-response diversity. 
* Select a ```context``` that will be rewritten by you. Aim to use texts from different domains, such as scientific publications, song lyrics, blog posts, etc. It is important to use sources published under open licenses, so you are asked to employ the [list](#recommended-sources-for-contexts) provided in these guidelines below. You are also allowed to create the contexts yourself, out of your head: if you are OK with them being re-distributed under a permissive license. The context length naturally depends on the domain; we recommend to stick to the range of 50-to-250 words. However, if you feel you can produce high-quality and diverse rewritten text versions for longer contexts counting up to 500 words, you are welcome 😊.
* Write a ```response``` to the ```prompt``` and ```context```. The response should fulfill the user request in the prompt, and it should be high-quality, relevant, and fluent. Generally, the response should be written in Bokmål (occasional Nynorsk is not entirely prohibited, though). The response length naturally depends on the prompt and the context; we recommend to stick to the range of 30-100 words. Please do not overthink the length and keep it natural 😊. Please avoid extracting and combining parts of the context and rephrase, rewrite, and reformulate them instead. Think about a response you would ideally want to get from an LLM. Try to avoid adding new facts or named entities to the response, which were not present in the prompt and context.
* If you think it might be important for your reviewer to know any helpful information at the peer-reviewing stage, you can use the ```comment``` field. In particular, you should always provide an URL to the context, but you can also specify other details for the reviewer to double-check. If the context is created entirely by you, put "self-produced context" in the ```comment``` field.
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
* Training: save your examples as ```rewriting_training.tsv```.
* Annotation: save your examples as ```rewriting_submission.tsv```.

### Annotation examples

Below, we provide annotation examples based on publicly available instruction-tuning datasets for English. Please refer to [a submission file example](./submission_sample.tsv) for more examples.

|```prompt``` | ```context``` | ```response``` | ```comment``` |
|---|---|---|---|
|Turn this poem into a poetic tweet|There will come soft rain and the smell of the ground,  And swallows circling with their shimmering sound;  And frogs in the pools singing at night,  And wild plum trees in tremulous white;  Robins will wear their feathery fire,  Whistling their whims on a low fence-wire;  And not one will know of the war, not one  Will care at last when it is done.  Not one would mind, neither bird nor tree,  If mankind perished utterly;  And Spring herself, when she woke at dawn  Would scarcely know that we were gone.|There will come a spring season that is plush and green Where all the animals will rejoice at the scene The war will be over Left behind a clover||
|Rewrite this Amazon review of Amazon-branded strawberry flavor cereal bars as if the customer loved it.|I got the strawberry. They are not as soft as the name brand ones which is what I like about these types of bars. Just a bit less soft. The strawberry filling is really sweet and this is coming from someone who is addicted to sugar and has to have lots of soda everyday. It's a bit much, but I'll still eat them, just not as much as I would with the brand name because those are so good. Cheap and ok.|I went with the strawberry flavor when I made my order. These are SO MUCH better than the name-brand cereal bars. They aren't too doughy or cakey but have a firmness to them, which is what I like. If you've got an insane sweet tooth like me, the strawberry filling is gonna have the perfect amount of sweetness. I could chow down on these every day and still enjoy them more than the brand-name ones. Cheap, but they've got a very good value for the price.||
|Rewrite the email so that it is more assertive and confident. Rephrase questions into statements as well.|Hi [Name], thanks for letting me know. I checked degree works and it says that ENGR 360 is counting towards my design course (instead of design and tech elective) when I already took a design course. Will you be able to fix that? Also, just to make sure, all of these classes will allow me to graduate next fall, correct? Because I need to have all of my design and tech electives done in order to be able to finish.   Thank you for your help, [Your Name]|Hi, thanks for letting me know. I checked degree works and it says that ENGR 360 is counting towards my design course (instead of the design and tech elective) when I have already taken a design course. I'm sure this is an error, so please fix that for me. I want to make sure that all of these classes will allow me to graduate next fall. For this to happen, I need to have all of my design and tech electives completed.||
|Rewrite the snippet with more positive adjectives.|Joseph Francis Tribbiani Jr. (born 1968/1969) is a fictional character, serving as one of the primary characters of the NBC sitcom Friends and the protagonist of its spin-off Joey. He is portrayed by Matt LeBlanc in both series.|Joseph Francis Tribbiani Jr. is a hilarious fictional character, serving as one of the fantastic primary characters of the outstanding NBC sitcom Friends and the phenomenal protagonist of its spin-off Joey. He is portrayed by the one-of-a-kind Matt LeBlanc in both series.||
|Please rewrite this so it sounds like some "trendy" Instagram influencer wrote it, probably trying to sound too hard to be cool, like they're 14 or something and just discovered TikTok for the first time.|If you’re ready to tackle the takeout favorite while ditching the deep-fryer and the wrapper entirely, then this Healthy Egg Roll In a Bowl is your new go-to weeknight meal.  It is, quite literally, an entire veggie- and protein-packed meal in a bowl with endless ingredient customizations.|lmao we all know egg rolls are great but lets face it...that stuff aint healthy 🤒 u gon gain a bunch of weight if you keep eatin them like i did lmao 🥣 luckily ive found an alternative.....yall just cram all your fave ingredients into a bowl w/out the fried wrapper. and you're DEFINITELY not throwing this in the deep fryer lmao 🍳 naur its healthy and safe and got tons of veggies n proteins, whatever u like. eat it for weekends r sumn like i do. yall can also bring it to work or school 💪🏢 grab all your fave recipes and chow down! until next time babes 💏||
|Rewrite this article so that it's cute and understandable for a young audience|The traditional diets of countries bordering the Mediterranean Sea differ slightly so there are different versions of the Mediterranean diet. However, in 1993 the Harvard School of Public Health, Oldways Preservation and Exchange Trust, and the European Office of the World Health Organization introduced the Mediterranean Diet Pyramid as a guide to help familiarize people with the most common foods of the region. More of an eating pattern than a strictly regimented diet plan, the pyramid emphasized certain foods based on the dietary traditions of Crete, Greece, and southern Italy during the mid-20th century. [1,2] At that time, these countries displayed low rates of chronic disease and higher than average adult life expectancy despite having limited access to healthcare. It was believed that the diet—mainly fruits and vegetables, beans, nuts, whole grains, ﬁsh, olive oil, small amounts of dairy, and red wine—contributed to their health benefits. The pyramid also highlighted daily exercise and the beneficial social aspects of eating meals together.|There is a wonderful place called the Mediterranean. This includes all the countries next to the sparkling Mediterranean Sea. If you eat the way people in those countries eat, it's called a Mediterranean diet. There is also something called the Mediterranean Diet Pyramid that a Harvard school and the World Health Organization made to help people learn what all the yummy Mediterranean foods are that they can eat. When you hear the word diet, you may think it's about losing weight, but this is actually just a delicious way to eat instead of a strict diet. The pyramid includes some typical foods that the people in Greece and Italy used to eat about a hundred years ago. They liked how people from those countries were so healthy and lived longer even though they didn't have many doctors or hospitals, it was like magic. This is probably because they were eating such yummy healthy foods like fruits, vegetables, nuts, grains, ﬁsh, olive oil, a little bit of milk and yogurt, and some red wine. The pyramid also shows that you can make your body very happy if you are active and eat your food together with the people you love.||


### Recommended sources for contexts

* [Norwegian Wikipedia](https://no.wikipedia.org/wiki/Portal:Forside)
* [Norwegian Colossal Corpus](https://huggingface.co/datasets/NbAiLab/NCC) (compressed files, one document per line):
    * [Openly available Norwegian books](https://data.hplt-project.org/NCC/books_no.txt.gz)
    * [Openly available Norwegian newspapers](https://data.hplt-project.org/NCC/newspaper_ocr_no.txt.gz) (caution: 4.4 GB of data, 9 million articles)
    * [Openly available public documents from Språkbanken: maalfrid_fylkesmannen](https://data.hplt-project.org/NCC/maalfrid_fylkesmannen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_regjeringen](https://data.hplt-project.org/NCC/maalfrid_regjeringen_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_uio](https://data.hplt-project.org/NCC/maalfrid_uio_no.txt.gz)
    * [Openly available public documents from Språkbanken: maalfrid_ssb](https://data.hplt-project.org/NCC/maalfrid_ssb_no.txt.gz)
