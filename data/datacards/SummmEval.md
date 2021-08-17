# Table card: SummmEval (Sum)

This table card grasp all the information for SummmEval. The specific definition of each field is available at this [link](https://github.com/ThomasScialom/BEAMetrics#adding-a-new-dataset).

**Task:** 
`summarization`

**Source Evaluation Set(s):** 
`CNN/DailyMail`

**Language(s):** 
`en`

**Evaluated Dimensions:** 
```
- coherence: ‘the summary should be well-structured and well-organized. The summary should not just be a heap of related information, should build from sentence to sentence to a coherent body of information about a topic.
- consistency: the factual alignment between the summary and the summarized source. A factually consistent summary contains only statements that are entailed by the source document. Annotators were also asked to penalize summaries that contained hallucinated facts.
- fluency: the quality of individual sentences. Drawing again from the DUC quality guidelines, sentences in the summary ‘‘should have no formatting problems, capitalization errors or obviously ungrammatical sentences (e.g., fragments, missing components) that make the text difficult to read.’’
- relevance: selection of important content from the source. The summary should include only important information from the source document. Annotators were instructed to penalize summaries that contained redundancies and excess information.
```

**Evaluation Scale:** 
`likert`

**Number Of Evaluated Texts** 
`1600`

**Number Of  References** 
`11`

**Information About The Annotators** 
`Expert annotators (the original paper also contain non expert annotator ratings)`

**Additional Information** 
```
- The authors release two set of annotations: from experts and turkers; we keep only the expert one.- The present file is derived from the original one released by the authors,it contains in addition a key 'text' that maps the example to the source article.
```

**Data URL:** 
``https://arxiv.org/abs/2007.12626``

**Citation:** 
```
@article{fabbri2020summeval,
        title={SummEval: Re-evaluating Summarization Evaluation},
        author={Fabbri, Alexander R and Kry{'s}ci{'n}ski, Wojciech and McCann, Bryan and Xiong, Caiming and Socher, Richard and Radev, Dragomir},
        journal={arXiv preprint arXiv:2007.12626},
        year={2020}
}
```
