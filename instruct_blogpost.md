<<<<<<< HEAD
# Scaling Instruction Tuning For Math
The paper [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) introduced the **superficial alignment hypothesis**, whose statement is as follows 
> A model’s knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. If this hypothesis is correct, and alignment [i.e finetuning] is largely about learning style, then a corollary of the Superficial Alignment Hypothesis is that one could sufficiently tune a pretrained language model with a rather small set of examples.
=======
# Scaling Instruction Tuning for Math
*General comments: the biggest problem with the experiments right now is that data quantity, data quality, and data diversity are all being scaled together. This is poor experimental design. It's fine to drop data quality as data quantity increases as this simulates a realistic tradeoff. However, data diversity should be a separate dimension of scaling. For my next draft, I will incorporate this change. Hopefully it will make the experiments more informative and my arguments slightly less sophistic.*
## Introduction
In this post, we seek to quantify the tradeoff between dataset size, quality, and diversity when instruction finetuning language models for math. 
>>>>>>> 9b6958db26942f6cb74c6337c6cf056441f8d4d0

The goal of this post is to see whether this hypothesis holds when instruction tuning `Llemma 7B`, a language model for mathematics. If the SAH holds, we should expect to see a small, highly-curated dataset of mathematical instructions match the performance of larger datasets and outperform any larger but lower-quality dataset. 

We introduce μInstruct, a dataset of 1600 high-quality instructions in the domain of mathematics. Per the SAH, we shouldn't need to scale beyond this dataset in order to achieve strong instruction following. 

Clearly, a larger dataset from the same distribution as μInstruct would yield at least no worse results. However, high quality instruction data is expensive, requiring a significant investment of researcher or crowdworker time. Therefore, given fixed resources for data collection, strategies for scaling up the amount of instruction data would involve either:
1. Using lower quality data, which is cheaper to collect. 
2. Using data less targeted towards the model's use case. Although this more general instruction data does not come from the exact distribution we want to model, it still may enforce helpful biases such as directly addressing the user's request, staying on topic, and reasoning in a chain-of-thought style. 

To reflect these practical constraints, our experiments will testing the SAH by combining μInstruct with either lower quality mathematics data, or data from high-quality instruction datasets that is not related to mathematics. 

<<<<<<< HEAD
## Data
The μInstruct dataset was created from an initial scrape of around 1500 highly-rated stack exchange answers and around 1500 questions from the Khan Academy subset of the AMPS dataset. Because the Khan Academy questions often had formatting issues, they were rewritten into valid markdown $\LaTeX{}$ by `gpt-3.5-turbo`. After this preprocessing, the initial 3,000 instructions were manually filtered to around 1600 instructions. The μInstruct dataset contains a mix of straightforward school-level problems, some quite challenging problems, and open-ended or soft questions. 

To stand in for a lower quality but larger math dataset than μInstruct, we use the Camel-AI math dataset, which contains 50,000 GPT-4 generated solutions to math problems. To stand in for a large, general-domain instruction dataset, we use [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), another dataset of GPT-4 outputs. Note that the OpenHermes-2.5 version of Mistral 7B beats the official Mistral-7B-Instruct by 72 ELO points on chatbot-arena. 
=======
*I need to make this example shorter.*
```
Below are a series of dialogues between various people and an AI mathematical assistant. The AI tries to be helpful, concise, honest, sophisticated, and humble-but-knowledgeable. The assistant always writes their responses in readable Markdown LaTeX. 

-----
Human: simplify (x + 1)/(x^2 + 2x + 1)

Assistant: Note that $x^2 + 2x + 1 = (x + 1)^2$. Therefore your expression simplifies to $(x + 1)/(x + 1)^2 = 1/(x + 1)$.
-----
Human: Why is 1 not a prime number?

Assistant: The definition of a prime number is a natural number greater than 1 that has no factors other than one and itself. One of several reasons that 1 is excluded from the definition is so that every natural number greater than 1 has a unique prime factorization.
----
Human: Explain why x^2 +y^2 = 1 defines a circle.
Assistant: The equation $x^2 + y^2 = 1$ defines a circle of radius 1 centered at the origin. To see this, let $x = r \cos \theta$ and $y = r \sin \theta$, where $r$ is the distance from the origin and $\theta$ is the angle between the positive $x$-axis and the line segment from the origin to the point $(x, y)$. Then $x^2 + y^2 = r^2 \cos^2 \theta + r^2 \sin^2 \theta = r^2 (\cos^2 \theta + \sin^2 \theta) = r^2$. Since $r$ is the distance from the origin, $r = 1$ if and only if the point $(x, y)$ is one unit away from the origin. Therefore, $x^2 + y^2 = 1$ if and only if the point $(x, y)$ is one unit away from the origin, which is the definition of a circle of radius 1 centered at the origin.
```

This is much closer to what we want. However, the in-context learning approach has some notable drawbacks. The short prompt shown here doesn't cover every desideratum for a helpful assistant: we also want to specify that the model should think step-by-step, that it shouldn't try to produce external links or bibliographic references, that it should admit its mistaken when its work is self-contradictory, and so forth. We may also want to convey information about subtle linguistic qualities such as tone and verbosity, which are best described through examples. First, language models, especially small ones such as Llemma 7B, aren't clever enough to follow arbitrary detailed instructions. Past a certain point of complexity, the model won't be clever enough to simultaneously meet all the constraints imposed by the prompt and will ignore parts of it. Moreover, as the prompt expands, we use up valuable space in our finite context length. A closely related issue is that generating text becomes more computationally expensive as the sequence length grows. This can be partially mitigated by [clever optimizations](https://arxiv.org/abs/2402.05099), but it would be best to avoid the problem entirely. 

The solution to these issues is *instruction tuning*, where after pretraining our base model, we finetune it on a curated dataset of instruction and response pairs. Because our instruction tuning dataset can consist of arbitrarily many examples, we can control the model's behavior as finely as we like. And since the directive to follow the user's instruction is written into the weights rather than in the context, we don't have to sacrifice effective context length or inference speed.

## Collecting Instruction Data
Collecting instruction tuning data presents different goals and tradeoffs from collecting data for unsupervised pretraining. Text of sufficiently high quality for pretraining is abundant on the web, and compute, not data availability, is the limiting factor on the size of pretraining datasets. On the other hand, text that represents desirable examples of instruction following by a language model is extremely scrarce. The primary bottleneck on the size of instruction tuning datasets is the cost of data collection, and the compute expenditure of training on these small datasets is trivial compared to pretraining. 

For instruction tuning, unlike in pretraining, more data may not always be better. This counterintuitive hypothesis is expressed in the [LIMA](https://arxiv.org/abs/2305.11206) paper:
> **Superficial Alignment Hypothesis:** A model’s knowledge and capabilities are learnt
almost entirely during pretraining, while alignment teaches it which subdistribution of formats should
be used when interacting with users. If this hypothesis is correct, and alignment is largely about
learning style, then a corollary of the Superficial Alignment Hypothesis is that one could sufficiently tune a pretrained language model with a rather small set of examples

*I really dislike the name superficial alignment hypothesis, because alignment denotes a lot more than instruction following. Need to find a polite way to indicate this and use some other name*

This hypothesis is plausible due to the compute differential between pretraining and finetuning, which is almost always more than two orders of magnitude. The LIMA model is able to achieve competitive instruction following with only 1000 finetuning examples. Prior works such as FLAN have used hundreds of thousands of examples. 

In this post, we wish to investigate the implications of practical tradeoffs faced when collecting instruction data. As you scale the size of your instruction tuning dataset, two things will tend to happen:
1. The average quality of your training examples will go down. If you are scraping instruction data from the web, it will be noisy: filtering the dataset will require a human-in-the-loop.  Given a fixed budget of man-hours for data work, you can only get a larger dataset by filtering less strictly. If you are working with crowdworkers, a fixed budget can get you a few expensive datapoints or many cheaper ones.
2. The specificity of your training examples will go down. In our experiments, we focus on instruction tuning for math. Clearly, the less related to mathematics we demand our data to be, the more we will be able to find.  *Again, this should be a separate dimension of scaling from data quantity*

We train Llemma 7B on three instruction datasets, all reflecting different points on this tradeoff. These datasets are detailed in the next section.

## Instruction Tuning Data

To represent the point on the tradeoff that maximizes quality and specificity, we develop [muInstruct](https://huggingface.co/datasets/EleutherAI/muInstruct), a small instruction dataset for mathematics. Around half the dataset comes from highly-rated answers from quantitative stack exchanges, and the other half comes from Khan Academy problems whose LaTex formatting was repaired by `gpt-3.5-turbo`. A total of around 3,000 examples were scraped from these two sources, and this was manually filtered down to the 1600 examples that form the training set.

To reflect an intermediate point on the tradeoff, we use the [CAMEL](https://huggingface.co/datasets/camel-ai/math) math dataset, which consists of GPT-4 solutions to math problems. This dataset contains around 5 times as many tokens as muInstruct, but is not as aggressively filtered for quality. It still retains a high degree of specificity towards math.

For a dataset that maximizes data quantity and diversity, we use [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), which contains around 1 million training examples and over 30 times as many tokens as muInstruct. OpenHermes-2.5 has been an extremely successful dataset for chat-finetuning open source models. On [chatbot arena](https://chat.lmsys.org/) The OpenHermes-2.5 finetune of Mistral-7b has 72 elo points on the official Mistral-7b-instruct model. While a strong dataset, OpenHermes has no degree of specificity towards math: there are as many questions about Javascript and French as there are about math.
>>>>>>> 9b6958db26942f6cb74c6337c6cf056441f8d4d0

## Experiments

We conduct instruction tuning experiments at three data scales: 1600 instructions, 50k instructions and 1M instructions. For the 1600 instruction case, we train solely on μInstruct: this run reflects the superficial alignment philosophy. At 50k instructions, we train one model on a μInstruct + camel mix and another on a μInstruct + OpenHermes mix. Finally, our only run at the 1M scale is a mixture of μInstruct and OpenHermes. 

We evaluate on the Hungarian National Math Exam, a "real-life" eval of high-school math ability. We compare our instruction tuned models to zero-shot `Llemma 7B`, few-shot `Llemma 7B`, and `MetaMath-Llemma-7B`, a finetune of `Llemma 7B` trained on a data-augmented version of the MATH and GSM8k training sets. 

Our experimental results are below. 

<img src="assets/results2.png" alt="alt text" width="600" height="450"/>

Our results do not appear to be consistent with the SAH. Most notably, we are only able to recover the performance of few-shot prompted by Llemma by training on camel math, which has lower data quality than μInstruct but makes up for it with greater quantity. 

<<<<<<< HEAD
Furthermore, these results illustrate the potential dangers of finetuning. Unless finetuning is done very carefully with a well-optimized data mixture, the finetuned model may perform much worse than the few-shot baseline. It is also worth noting that although `MetaMath-Llemma 7B` beats `Llemma 7B` by a large margin on MATH, the MetaMath model is weaker on Hungarian math. This result suggests the MetaMath model is highly specialized towards the MATH problem distribution and has weaker general problem-solving ability than its MATH score would suggest. 
=======
## Conclusion
*This argument is a little weak right now because the current experiments can't distinguish between the effect of data diversity and the effect of data quality.*
>>>>>>> 9b6958db26942f6cb74c6337c6cf056441f8d4d0

The most important limitation of our findings is that `Llemma 7B` is a small model by 2024 standards, and larger models are both more sample efficient and have greater reasoning capabilities that may be harmed by poor quality data. Both of these factors would suggest small high-quality datasets work better for larger models. 
