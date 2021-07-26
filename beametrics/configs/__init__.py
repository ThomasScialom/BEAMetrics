from beametrics.configs.config_captionning_composite import CaptioningComposite
from beametrics.configs.config_captionning_Flickr8k import CaptioningFlickr8k
from beametrics.configs.config_captionning_Pascal50s import CaptioningPascal50s
from beametrics.configs.config_data2text_webnlg import Data2textWebNLG
from beametrics.configs.config_simplification_likert_human import SimplificationLikertHuman
from beametrics.configs.config_simplification_likert_system import SimplificationLikertSystem
from beametrics.configs.config_simplification_pairwise_human import SimplificationPairwiseHuman
from beametrics.configs.config_simplification_pairwise_system import SimplificationPairwiseSystem
from beametrics.configs.config_summarization_cnndm import SummarizationCNNDM

D_ALL_DATASETS = {
    #'SimplificationLikertHuman': SimplificationLikertHuman,
    'SimplificationLikertSystem': SimplificationLikertSystem,
    #'SimplificationPairwiseHuman': SimplificationPairwiseHuman,
    #'SimplificationPairwiseSystem': SimplificationPairwiseSystem,
    'SummarizationCNNDM': SummarizationCNNDM,
    'Data2textWebNLG': Data2textWebNLG,
    'CaptioningFlickr8k': CaptioningFlickr8k,
    'CaptioningPascal50s': CaptioningPascal50s,
    #'CaptioningComposite': CaptioningComposite,

}