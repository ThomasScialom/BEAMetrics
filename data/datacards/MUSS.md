# Table card: MUSS (MUS)

This table card grasp all the information for MUSS. The specific definition of each field is available at this [link](https://github.com/ThomasScialom/BEAMetrics#adding-a-new-dataset).

**Task:** 
`simplification`

**Source Evaluation Set(s):** 
`MUSS`

**Language(s):** 
`en fr es`

**Evaluated Dimensions:** 
```
- simplicity: is the output simpler than the original sentence?
- adequacy: Adequacy: to what extent is the meaning expressed in the original sentence preserved in the output?
- fluency: is the output grammatical and well formed?
```

**Evaluation Scale:** 
`likert 1-4`

**Number Of Evaluated Texts** 
`150`

**Number Of  References** 
`1`

**Information About The Annotators** 
`We recruit volunteer native speakers for each language (5 in English, 2 in French, and 2 in Spanish)`

**Additional Information** 
```

        Please sign up [here](https://newsela.com/data/) to access the Newsela data.
        
        For simplicity, please consider only simplification operations between the source and simplification, i.e. modifications that actually make the sentence simpler to read and understand, based on your own judgement.

        Scale:
        4 = Entirely simpler (It seems like all possible simplification operations were conducted, it seems like it is impossible to make the sentence simpler)
        3 = Much simpler (3 or more simplification operations of different types)
        2 = Simpler (2 or 3 simple simplifications or one very relevant simplification)
        1 = Slightly simpler (e.g. only one simple simplification operation such as removing a few words)
        0 = Not simpler or more complicated
        
        Simplification operations  by order of complexity include but are not limited to:
        Removing an adjective or details in parenthesis (unimportant information).
        Shortening a sentence to make it easier to read by removing clauses.
        Replacing a complex word with a simpler one.
        Changing complicated tenses (e.g. past perfect) to simpler ones (e.g. present).
        Splitting long sentences into multiple.
        Reordering the information of a sentence to make it more natural.

```

**Data URL:** 
``https://arxiv.org/pdf/2005.00352.pdf``

**Citation:** 
```
@article{martin2020muss,
        title={MUSS: Multilingual Unsupervised Sentence Simplification by Mining Paraphrases},
        author={Martin, Louis and Fan, Angela and de la Clergerie, {'E}ric and Bordes, Antoine and Sagot, Beno{\^\i}t},
        journal={arXiv preprint arXiv:2005.00352},
        year={2020}
}
```
