# Table card: SummmEval-multi (mSu)

This table card grasp all the information for SummmEval-multi. The specific definition of each field is available at this [link](https://github.com/ThomasScialom/BEAMetrics#adding-a-new-dataset).

**Task:** 
`summarization`

**Source Evaluation Set(s):** 
`MLSUM ('de', 'es', 'fr', 'ru', 'tr'), CNN/DailyMail (en), Liputan6 (id), and LCSTS (zh)`

**Language(s):** 
`de es fr ru tr en zh id`

**Evaluated Dimensions:** 
```
- focus: How much information contained in the evaluated summary text can also be found in the source document?
- coverage: How much information contained in the source document can also be found in the evaluated summary?
```

**Evaluation Scale:** 
`likert`

**Number Of Evaluated Texts** 
`2160`

**Number Of  References** 
`1`

**Information About The Annotators** 
`3 Amazon Mechanical Turk annotators`

**Additional Information** 
```
Direct Assessment (“DA”) method (Graham et al., 2015; Graham et al., 2017), which has become the de facto for MT evaluation in WMT. For each HIT (100 samples), DA incorporates 10 pre-annotated samples for quality control. Crowd-sourced workers are given two texts and asked the question (in the local language): How much information contained in the second text can also be found in the first text? We combine focus and coverage annotation into 1 task, as the only thing that differentiates them is the ordering of the system and reference summaries, which is opaque to the annotators.
```

**Data URL:** 
``https://arxiv.org/pdf/2106.01478.pdf``

**Citation:** 
```
@article{koto2021evaluating,
        title={Evaluating the Efficacy of Summarization Evaluation across Languages},
        author={Koto, Fajri and Lau, Jey Han and Baldwin, Timothy},
        journal={arXiv preprint arXiv:2106.01478},
        year={2021}
}
```
