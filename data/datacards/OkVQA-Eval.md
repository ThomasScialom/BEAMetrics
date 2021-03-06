# Table card: OkVQA-Eval (VQA)

This table card grasp all the information for OkVQA-Eval. The specific definition of each field is available at this [link](https://github.com/ThomasScialom/BEAMetrics#adding-a-new-dataset).

**Task:** 
`VQA`

**Source Evaluation Set(s):** 
`[OkVQA](https://okvqa.allenai.org/)`

**Language(s):** 
`en`

**Evaluated Dimensions:** 
```
- possibility: Question 1: In your opinion, is the answer possible? Does it make sense in some possible situation? 0: possible, in some conceivable situation / 1: impossible, or makes no sense
- obviousness: Question 2: If the image was shown to 100 people and the question was asked, how many people do you think would give the answer? a number between 1 and 100
- correctness: Question 3: Can you say that the answer is definitely factually correct (use Google if necessary)? 0: definitely correct, 1: can't say (matter of opinion, depends on the situation, the question does not have a 'right' answer, 2: definitely incorrect
```

**Evaluation Scale:** 
`binary, likert and multiple choice`

**Number Of Evaluated Texts** 
`300`

**Number Of  References** 
`10`

**Information About The Annotators** 
`Expert annotators working at the Sorbonne University.`

**Additional Information** 
```
The annotation protocol is detailed on the appendix of this paper.
```

**Data URL:** 
``https://github.com/ThomasScialom/BEAMetrics/tree/main/beametrics``

**Citation:** 
```
under review
```
