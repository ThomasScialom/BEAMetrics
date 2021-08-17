# Table card: Flickr8K (Fli)

This table card grasp all the information for Flickr8K. The specific definition of each field is available at this [link](https://github.com/ThomasScialom/BEAMetrics#adding-a-new-dataset).

**Task:** 
`captioning`

**Source Evaluation Set(s):** 
`Flickr8k`

**Language(s):** 
`en`

**Evaluated Dimensions:** 
```
- score: A score of 4 means that the caption describes the image perfectly (without any mistakes), a score of 3 that the caption almost describes the image (minor mistakes are allowed, e.g. in the number of entities), whereas a score of 2 indicates that the caption only describes some aspects of the image, but could not be used as its description, and a score of 1 indicates that the caption bears no relation to the image.
```

**Evaluation Scale:** 
`likert`

**Number Of Evaluated Texts** 
`5664`

**Number Of  References** 
`5`

**Information About The Annotators** 
`The judges were 21 adult native speakers of American English, mostly recruited from among the local graduate student population.`

**Additional Information** 
```

        - The evaluation procedure is detailed Page 19 in https://www.jair.org/index.php/jair/article/view/10833/25854, in Section 4. Evaluation Procedures and Metrics for Image Description." 
        - Likert scale 1-4. The judges were experts: 21 adult native speakers of American English, mostly recruited from among the local graduate student population. 
        - Inter-annotator agreement, measuredas Krippendorff’s (2004)α, is high (α= 0.81) 
        
```

**Data URL:** 
``jair.org/index.php/jair/article/view/10833/25854``

**Citation:** 
```
@article{hodosh2013framing,
        title={Framing image description as a ranking task: Data, models and evaluation metrics},
        author={Hodosh, Micah and Young, Peter and Hockenmaier, Julia},
        journal={Journal of Artificial Intelligence Research},
        volume={47},
        pages={853--899},
        year={2013}
}
```
