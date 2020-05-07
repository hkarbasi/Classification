from _imports import *
from _utils import *


class BOW(object):

    def __init__(self, _context, experiment, feature_name,
                 df_training=None, df_validation=None,
                 training_ids=None, validation_ids=None):

        self.c = _context
        self.df_training = df_training
        self.df_validation = df_validation
        self.training_ids = training_ids
        self.validation_ids = validation_ids
        self.experiment = experiment
        self.feature_name = feature_name

    # Create Corpus & Term Document Frequency
    def para_corpus(self, text):
        # return self.id2word.doc2bow(text)
        vector = [0] * len(self.id2word)
        indices = self.id2word.doc2idx(text)
        for index in indices:
            if index >= 0:
                vector[index] += 1
        return pd.Series(vector)

    def corpus_creation(self):
        text_training = self.df_training['final_text']
        text_validation = self.df_validation['final_text']
        self.id2word = corpora.Dictionary(text_training)
        # print(type(text_training))
        # print(text_training)

        # self.id2word.filter_extremes(no_below=self.c.features['dynamic'][self.feature_name]['no_below'],
        #                              no_above=self.c.features['dynamic'][self.feature_name]['no_above'])
        # print(type(self.experiment[self.feature_name + '_no_below']))
        # print(type(self.experiment[self.feature_name + '_no_above']))
        # self.id2word.filter_extremes(no_below=eval(self.experiment[self.feature_name + '_no_below']),
        #                              no_above=eval(self.experiment[self.feature_name + '_no_above']))

        print("no_below {}".format(self.experiment))
        self.id2word.filter_extremes(no_below=eval(str(self.experiment[self.feature_name + '_no_below'])),
                                     no_above=eval(str(self.experiment[self.feature_name + '_no_above'])))

        corpus_training = pd.DataFrame(text_training.apply(self.para_corpus))
        corpus_validation = pd.DataFrame(text_validation.apply(self.para_corpus))
        return corpus_training, corpus_validation

    def dynamic_params_name(self, retrieval_mode=False):
        if not retrieval_mode:
            feature_name = self.experiment['name']
            feature_type = self.experiment['type']
        else:
            feature_name = self.feature_name
            feature_type = self.c.features['dynamic'][self.feature_name]['type']

        params_names = '_'.join(str(1.0 * self.experiment[feature_name + '_' + param_name])
                                if not isinstance(self.experiment[feature_name + '_' + param_name], str)
                                else self.experiment[feature_name + '_' + param_name] for param_name in
                                self.c.features['dynamic'][feature_name]['params'].keys())

        return '{}_{}({})_{}_cv({})'.format(self.c.project_name
                                            , feature_type
                                            , feature_name
                                            , params_names
                                            , self.experiment['cv']
                                            )

    def construct_features(self):
        topic_training_df, topic_validation_df = self.corpus_creation()

        topic_training_df.reset_index(drop=True, inplace=True)
        topic_training_df = pd.concat([self.training_ids, topic_training_df], axis=1)

        topic_validation_df.reset_index(drop=True, inplace=True)
        topic_validation_df = pd.concat([self.validation_ids, topic_validation_df], axis=1)

        df_concat = pd.concat([topic_training_df, topic_validation_df], axis=0)
        df_concat.reset_index(drop=True, inplace=True)

        df_concat.to_csv('{}data/features/dynamic_features/{}.csv'.format(self.c.directory['save'],
                                                                          self.dynamic_params_name()),
                         index=False)

        return df_concat

    def retrieve_features(self):
        df = pd.read_csv('{}data/features/dynamic_features/{}.csv'.format(self.c.directory['save'],
                                                                          self.dynamic_params_name(True)))
        return df


def concat_cols(row):
    return ' '.join(map(str, row))


def concat_cols_df(df):
    if isinstance(df, pd.DataFrame):
        if len(df.columns) == 1:
            return df
        return df.apply(concat_cols, axis=1)
    elif isinstance(df, pd.Series):
        return df.apply(concat_cols)
    raise util.Failed('Input was not either dataframe or series!')


def preprocess_text(text, replacements):
    #     replacements=list(replacements)
    #     print(replacements)
    for replacement in replacements:
        try:
            text = re.sub(*replacement, text)
        except TypeError as e:
            print('{} + text {}'.format(replacement, text))
            raise e
    #     return tokenizer_tag(text)
    return text


def preprocess_text_series(series, replacements):
    return series.apply(preprocess_text, args=(replacements,))


stop_words = stopwords.words('english')
ps = PorterStemmer()


def para_words(text):
    if not (isinstance(text, str)):
        return ''
    result = [ps.stem(word) for word in simple_preprocess(text, min_len=1, max_len=60, deacc=True) if
              word not in stop_words]
    return ', '.join(result)


def para_words_series(series):
    return series.apply(para_words)


def para_words_df(df):
    if isinstance(df, pd.DataFrame):
        # if len(df.columns) == 1:
        #     return df
        return df.apply(para_words, axis=1)
    elif isinstance(df, pd.Series):
        return df.apply(para_words)


def para_parsing(text):
    if not isinstance(text, str):
        return ['']
    return [x.strip() for x in text.split(',')]


def para_parsing_df(df):
    return df.apply(para_parsing)


def prep_input(df, context):
    df['final_text'] = parallelize_df(df['final_text'], para_parsing_df, context)
    return df[[context.id_name, 'final_text']]


def prep_features(df, feature, context):
    df_input = df[df.columns[~df.columns.isin([context.id_name])]]
    text_all = parallelize_df(df_input, concat_cols_df, context)
    if isinstance(text_all, pd.DataFrame):
        text_all = text_all.iloc[:, 0]
    cleaned_text_all = preprocess_text_series(text_all, feature['replacements'])
    words = parallelize_df(cleaned_text_all, para_words_df, context)
    # words = para_words_series(cleaned_text_all)

    df = pd.concat([df[context.id_name],
                    text_all,
                    cleaned_text_all,
                    words
                    ],
                   axis=1)
    df.columns = [context.id_name, 'orig_text', 'cleaned_text', 'final_text']
    return df


def construct_features(_context, experiment, feature_name,
                       df_training, df_validation,
                       training_ids, validation_ids):
    instance = BOW(_context, experiment, feature_name,
                   df_training, df_validation,
                   training_ids, validation_ids)
    return instance.construct_features()


def retrieve_features(_context, experiment, feature_name):
    instance = BOW(_context, experiment, feature_name)
    return instance.retrieve_features()
