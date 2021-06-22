

    # Data2text - System WikiBIO
    config = get_dataset_config(task='data2text',
                                file_name='wikibio.204.json',
                                file_name_processed='processed.wikibio.204.json',
                                load_function=format_files.format_wikibio,
                                mertric_scorer=metrics.Metrics_Scorer_data2text_wikibio,
                                metrics=DEFAULT_METRICS + ['QuestEval', 'QuestEval_refless'],  # PARENT
                                dimensions=['fluency', 'coverage', 'semantics'],
                                correl_function=pearsonr,
                                correl_preproc=None,
                                keep_only=None,
                                nb_refs=1,
                                lang='en',
                                sampled_from='wikibio',
                                citation='None')
    if task is None or task == config['task']:
        dataset_names['Data2text_wikibio204'] = config


    # Image Captioning - capeval1k
    config = get_dataset_config(task='image_captioning',
                                file_name='captioning_human_judgments.pkl',
                                file_name_processed='processed.capeval1k.json',
                                load_function=format_files.format_captioning,
                                mertric_scorer=metrics.Metrics_Scorer,
                                metrics=DEFAULT_METRICS,
                                dimensions=['score'],
                                correl_function=kendalltau,
                                correl_preproc=None,
                                keep_only='capeval1k',
                                nb_refs=5,
                                lang='en',
                                sampled_from='capeval1k',
                                citation='None')
    if task is None or task == config['task']:
        dataset_names['Image_Captionning_capeval1k'] = config









    return dataset_names

