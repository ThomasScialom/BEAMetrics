from typing import List, Dict
from tqdm import tqdm
import torch
import nubia_score
from beametrics.metrics.metrics import MetricBase


class MetricNubia(MetricBase):

    def __init__(
        self,
        lang: str,
        task: str,
        *args, **kwargs
    ):
        self.metric = nubia_score.Nubia()

    @classmethod
    def metric_name(cls):
        return 'nubia'

    def sub_metric_names(self):
        return ['nubia_score', 'nubia_irrelevancy']

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:

        res_formated = {'nubia_score': [], 'nubia_irrelevancy': []}
        for references, prediction in tqdm(zip(list_references, predictions)):
            score = self.metric.score(
                ref=references[0],
                hyp=prediction,
                verbose=False,
                get_features=True
            )

            res_formated['nubia_score'].append(score['nubia_score'])
            res_formated['nubia_irrelevancy'].append(score['features']['irrelevancy'])

        return res_formated


class MetricPerplexity(MetricBase):

    def __init__(
        self,
        device: str = 'cuda',
        *args,
        **kwargs
    ):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model_id = 'gpt2-large'
        self.model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
        self.device = device

    @classmethod
    def metric_name(cls):
        return 'perplexity'

    def pipeline(
            self,
            predictions: List[str] = None,
            sources: List[str] = None,
            list_references: List[List[str]] = None,
    ) -> Dict:

        res = []
        for _, prediction in tqdm(zip(list_references, predictions)):

            ppl = 999
            if prediction:
                encodings = self.tokenizer(prediction, return_tensors='pt')

                max_length = self.model.config.n_positions
                stride = 512

                lls = []
                for i in range(0, encodings.input_ids.size(1), stride):
                    begin_loc = max(i + stride - max_length, 0)
                    end_loc = min(i + stride, encodings.input_ids.size(1))
                    trg_len = end_loc - i  # may be different from stride on last loop
                    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
                    target_ids = input_ids.clone()
                    target_ids[:, :-trg_len] = -100

                    with torch.no_grad():
                        outputs = self.model(input_ids, labels=target_ids)
                        log_likelihood = outputs[0] * trg_len

                    lls.append(log_likelihood.nan_to_num())

                ppl = torch.exp(torch.stack(lls).sum() / end_loc).item()
            res.append(ppl)

        return {self.metric_name(): res}