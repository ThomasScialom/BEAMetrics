# Table card: NeurIPS_Efficient_QA_2020 (QA)

This table card grasp all the information for NeurIPS_Efficient_QA_2020. The specific definition of each field is available at this [link](https://github.com/ThomasScialom/BEAMetrics#adding-a-new-dataset).

**Task:** 
`QA`

**Source Evaluation Set(s):** 
`EfficientQA competition at NeurIPS 2020`

**Language(s):** 
`en`

**Evaluated Dimensions:** 
```
- consistency: contains predictions that were determined to be definitely correct by annotators,
- plausibility: contains preditions that were determined to be possibly correct given some interpretation of the question.
```

**Evaluation Scale:** 
`binary`

**Number Of Evaluated Texts** 
`1800`

**Number Of  References** 
`1`

**Information About The Annotators** 
`In 2.3: three separate raters`

**Additional Information** 
```
- Based on our human evaluation, annotations on ambiguity have low agreement rate (61.3%, Cohen’s κ = 22.6), and predictions with the same level of plausibility are often marked as “definitely correct” or “possibly correct” by different human raters. 
        - The annotation and Json keys are described in https://github.com/google-research-datasets/natural-questions/tree/master/nq_open. 
        - Discrepancy between automatic eval and human eval Human raters find 13% and 17% of the predictions that do not match the reference answers to be definitely correct or possibly correct, respectively, overall increasing the accuracy of the systems. Most systems showed 17–25% and 41–54% improvement in accuracy when using definitely correct and possibly correct human evaluation respectively, compared to automatic evaluation metric which only consider exact string match to existing reference answers. An exception is NAVER RDR, which achieves significantly larger improvements (32% and 71%, respectively). We also found that when the gap in automatic measure between systems is marginal (around or smaller than 1%), human evaluation may change the rankings between the models.
```

**Data URL:** 
``https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.efficientqa.test.1.1.jsonl``

**Citation:** 
```
@article{min2021neurips,
        title={NeurIPS 2020 EfficientQA competition: Systems, analyses and lessons learned},
        author={Min, Sewon and Boyd-Graber, Jordan and Alberti, Chris and Chen, Danqi and Choi, Eunsol and Collins, Michael and Guu, Kelvin and Hajishirzi, Hannaneh and Lee, Kenton and Palomaki, Jennimaria and others},
        journal={arXiv preprint arXiv:2101.00133},
        year={2021}
}
```
