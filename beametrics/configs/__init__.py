from beametrics.configs.config_captionning_Flickr8k import CaptioningFlickr8k
from beametrics.configs.config_captionning_Pascal50s import CaptioningPascal50s
from beametrics.configs.config_data2text_webnlg import Data2textWebNLG
from beametrics.configs.config_simplification_likert_system import SimplificationLikertSystem
from beametrics.configs.config_summarization_cnndm import SummarizationCNNDM
from beametrics.configs.config_qa_neuripsQA2020 import NeurIPS2020openQA
from beametrics.configs.config_novqa import NoVQA_eval
from beametrics.configs.config_realsum import REALSum_eval
#from beametrics.configs.config_simplification_likert_human import SimplificationLikertHuman
#from beametrics.configs.config_simplification_pairwise_human import SimplificationPairwiseHuman
#from beametrics.configs.config_simplification_pairwise_system import SimplificationPairwiseSystem
#from beametrics.configs.config_captionning_composite import CaptioningComposite
from beametrics.configs.config_novqa_nohuman import NoVQA_eval_no_human


D_ALL_DATASETS = {
    'SimplificationLikertSystem': SimplificationLikertSystem,
    'Data2textWebNLG': Data2textWebNLG,
    'CaptioningFlickr8k': CaptioningFlickr8k,
    'CaptioningPascal50s': CaptioningPascal50s,
    'SummarizationCNNDM': SummarizationCNNDM,
    'REALSum_eval': REALSum_eval,
    'NeurIPS2020openQA': NeurIPS2020openQA,
    'NoVQA_eval': NoVQA_eval,
    'NoVQA_eval_no_human': NoVQA_eval_no_human
}