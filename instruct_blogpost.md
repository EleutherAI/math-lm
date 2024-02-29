# Scaling Instruction Tuning For Math
The paper [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) introduced the **superficial alignment hypothesis**, whose statement is as follows 
> A model’s knowledge and capabilities are learnt almost entirely during pretraining, while alignment teaches it which subdistribution of formats should be used when interacting with users. If this hypothesis is correct, and alignment [i.e finetuning] is largely about learning style, then a corollary of the Superficial Alignment Hypothesis is that one could sufficiently tune a pretrained language model with a rather small set of examples.

The goal of this post is to see whether this hypothesis holds when instruction tuning `Llemma 7B`, a language model for mathematics. If the SAH holds, we should expect to see a small, highly-curated dataset of mathematical instructions match the performance of larger datasets and outperform any larger but lower-quality dataset. 

We introduce μInstruct, a dataset of 1600 high-quality instructions in the domain of mathematics. Per the SAH, we shouldn't need to scale beyond this dataset in order to achieve strong instruction following. 

Clearly, a larger dataset from the same distribution as μInstruct would yield at least no worse results. However, high quality instruction data is expensive, requiring a significant investment of researcher or crowdworker time. Therefore, given fixed resources for data collection, strategies for scaling up the amount of instruction data would involve either:
1. Using lower quality data, which is cheaper to collect. 
2. Using data less targeted towards the model's use case. Although this more general instruction data does not come from the exact distribution we want to model, it still may enforce helpful biases such as directly addressing the user's request, staying on topic, and reasoning in a chain-of-thought style. 

To reflect these practical constraints, our experiments will testing the SAH by combining μInstruct with either lower quality mathematics data, or data from high-quality instruction datasets that is not related to mathematics. 

## Data
The μInstruct dataset was created from an initial scrape of around 1500 highly-rated stack exchange answers and around 1500 questions from the Khan Academy subset of the AMPS dataset. Because the Khan Academy questions often had formatting issues, they were rewritten into valid markdown $\LaTeX{}$ by `gpt-3.5-turbo`. After this preprocessing, the initial 3,000 instructions were manually filtered to around 1600 instructions. The μInstruct dataset contains a mix of straightforward school-level problems, some quite challenging problems, and open-ended or soft questions. 

To stand in for a lower quality but larger math dataset than μInstruct, we use the Camel-AI math dataset, which contains 50,000 GPT-4 generated solutions to math problems. To stand in for a large, general-domain instruction dataset, we use [OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5), another dataset of GPT-4 outputs. Note that the OpenHermes-2.5 version of Mistral 7B beats the official Mistral-7B-Instruct by 72 ELO points on chatbot-arena. 

## Experiments

We conduct instruction tuning experiments at three data scales: 1600 instructions, 50k instructions and 1M instructions. For the 1600 instruction case, we train solely on μInstruct: this run reflects the superficial alignment philosophy. At 50k instructions, we train one model on a μInstruct + camel mix and another on a μInstruct + OpenHermes mix. Finally, our only run at the 1M scale is a mixture of μInstruct and OpenHermes. 

We evaluate on the Hungarian National Math Exam, a "real-life" eval of high-school math ability. We compare our instruction tuned models to zero-shot `Llemma 7B`, few-shot `Llemma 7B`, and `MetaMath-Llemma-7B`, a finetune of `Llemma 7B` trained on a data-augmented version of the MATH and GSM8k training sets. 

Our experimental results are below. 

<img src="assets/results2.png" alt="alt text" width="600" height="450"/>

Our results do not appear to be consistent with the SAH. Most notably, we are only able to recover the performance of few-shot prompted by Llemma by training on datasets with at least 50k instructions. 

Furthermore, these results illustrate the potential dangers of finetuning. Unless finetuning is done very carefully with a sufficiently large dataset and a well-optimized data mixture, the finetuned model may perform much worse than the few-shot baseline. It is also worth noting that although `MetaMath-Llemma 7B` beats `Llemma 7B` by a large margin on MATH, the MetaMath finetuning doesn't yield an advantage on Hungarian math. This result suggests the MetaMath model is highly specialized towards the MATH problem distribution and has weaker general problem-solving ability than its MATH score would suggest. 

The most important limitation of our findings is that `Llemma 7B` is a small model by 2024 standards, and larger models are both more sample efficient and have greater reasoning capabilities that may be harmed by poor quality data. Both of these factors would suggest small high-quality datasets work better for larger models. 
