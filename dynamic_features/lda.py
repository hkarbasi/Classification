from _imports import *
from _utils import *


class LDA(object):

    def __init__(self, _context, experiment, feature_name,
                 df_training=None, df_validation=None,
                 training_ids=None, validation_ids=None):

        self.topics = []
        self.c = _context
        self.df_training = df_training
        self.df_validation = df_validation
        self.training_ids = training_ids
        self.validation_ids = validation_ids
        self.experiment = experiment
        self.feature_name = feature_name

    # Create Corpus & Term Document Frequency
    def para_corpus(self, text):
        return self.id2word.doc2bow(text)

    def corpus_creation(self):
        text_training = self.df_training['final_text']
        text_validation = self.df_validation['final_text']
        self.id2word = corpora.Dictionary(text_training)

        # self.id2word.filter_extremes(no_below=self.c.features['dynamic'][self.feature_name]['no_below'],
        #                              no_above=self.c.features['dynamic'][self.feature_name]['no_above'])
        # print(type(self.experiment[self.feature_name + '_no_below']))
        # print(type(self.experiment[self.feature_name + '_no_above']))
        # self.id2word.filter_extremes(no_below=eval(self.experiment[self.feature_name + '_no_below']),
        #                              no_above=eval(self.experiment[self.feature_name + '_no_above']))

        # print("no_below {}".format(self.experiment))
        # print("feature_name = {}".format(self.feature_name))
        self.id2word.filter_extremes(no_below=eval(str(self.experiment[self.feature_name + '_no_below'])),
                                     no_above=eval(str(self.experiment[self.feature_name + '_no_above'])))

        corpus_training = text_training.apply(self.para_corpus)
        corpus_validation = text_validation.apply(self.para_corpus)
        return corpus_training, corpus_validation

    def topic_modeling(self, corpus_training):
        # temp_dir='{}tmp/{}_{}/'.format(self.c.directory['save'],
        #                                       self.experiment[self.feature_name+'_LDA_num_topics'],
        #                                       self.experiment['cv'])

        if isinstance(self.experiment, pd.Series):
            temp_dir = '{}tmp/{}/'.format(self.c.directory['save'],
                                          '_'.join(list(map(str, list(self.experiment))))
                                          )
        else:
            temp_dir = '{}tmp/{}/'.format(self.c.directory['save'],
                                          '_'.join(list(map(str, list(self.experiment.values()))))
                                          )

        # temp_dir='{}tmp/{}/'.format(self.c.directory['save'],
        #                             self.dynamic_params_name()
        #                             )

        try:
            os.makedirs(temp_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise Failed('Counld\'nt create tmp folder!')

        topic_model = gensim.models.wrappers.LdaMallet(self.c.directory['mallet'],
                                                       corpus=corpus_training,
                                                       num_topics=int(eval(str(
                                                           self.experiment[self.feature_name + '_LDA_num_topics']))),
                                                       id2word=self.id2word,
                                                       workers=self.c.num_cores,
                                                       # workers=1,
                                                       optimize_interval=1,
                                                       iterations=self.c.features['dynamic'][self.feature_name][
                                                           'LDA_iteration_threshold'],
                                                       random_seed=self.c.seed,
                                                       prefix=temp_dir
                                                       )

        return topic_model

    def normalize(self, row):
        # return row
        # print(eval(str(self.experiment[self.feature_name + '_LDA_topic_cutoff_threshold'])))
        try:
            # prop=[x[1] if x[1]>c.topic_cutoff_threshold else 0 for x in row]
            prop = []
            index = 0
            for item in row:
                if item is None:
                    count_sync(1000)
                    prop.append(-1)
                else:
                    while index < item[0]:
                        index += 1
                        prop.append(-1)
                    prop.append(
                        item[1] if item[1] > eval(str(
                            self.experiment[self.feature_name + '_LDA_topic_cutoff_threshold'])) else 0)
                index += 1
            for i in range(index, int(eval(str(self.experiment[self.feature_name + '_LDA_num_topics'])))):
                prop.append(-1)
            # prop = [x[1] if x is not None and x[1] > c.topic_cutoff_threshold else 0 for x in row]
            if np.sum(prop) == 0:
                return pd.Series(prop)
                # return -1
            return pd.Series(prop / np.sum(prop),
                             index=range(int(eval(str(self.experiment[self.feature_name + '_LDA_num_topics'])))))
        except:
            raise ValueError('row = {}'.format(row))

    def topic_normalization(self, corpus_training, corpus_validation, topic_model):
        topic_corpus_training = topic_model[corpus_training]
        topic_corpus_training_df = pd.DataFrame(topic_corpus_training)
        topic_training_df = topic_corpus_training_df.apply(self.normalize, axis=1)

        topic_corpus_validation = topic_model[corpus_validation]
        topic_corpus_validation_df = pd.DataFrame(topic_corpus_validation)
        topic_validation_df = topic_corpus_validation_df.apply(self.normalize, axis=1)

        return topic_training_df, topic_validation_df

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

    def save_topic_keywords(self, topic_model):
        topics = list()
        num_topics = int(eval(str(self.experiment[self.feature_name + '_LDA_num_topics'])))

        for i in range(num_topics):
            wp = topic_model.show_topic(i)
            topics.append(", ".join([word for word, prop in wp]))

        df_topics_saving = pd.DataFrame(topics, columns=['top keywords'])
        df_topics_saving.index = range(1, num_topics + 1)

        folder = "{}data/features/dynamic_features/{}".format(self.c.directory['save'],
                                                              self.dynamic_params_name())
        mkdir(folder)
        df_topics_saving.to_csv('{}/{}_{}_topic_keywords.csv'.format(folder,
                                                                     self.c.project_name,
                                                                     num_topics),
                                encoding='utf-8')
        self.topics = topics

    def save_top_sample_dominant_topics(self, data):
        def find_dominant_topic_for_original_weight(row):
            topic_num = row[0][0]
            prop_topic = row[0][1]

            for item in row[1:]:
                #         c.print_asynch('item in for {} - {} - {}'.format(item, item[0], item[1]))
                if item is None or item[0] is None or item[1] is None:
                    count_sync(1000)
                elif item[1] > prop_topic:
                    topic_num = item[0]
                    prop_topic = item[1]

            # row = sorted(enumerate(row), key=lambda x: (x[1][1] if x[1][1] is not None else 0), reverse=True)
            # topic_num=row[0][1][0]
            # prop_topic=row[0][1][1]

            return pd.Series([int(topic_num),
                              round(prop_topic, 4),
                              self.topics[topic_num]
                              ]
                             )

        df_corpus_topics = data.apply(find_dominant_topic_for_original_weight)
        df_corpus_topics = pd.concat([df_corpus_topics, self.df_training], axis=1)
        col_renamed = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
        col_renamed.extend(list(self.df_training.columns))
        df_corpus_topics.columns = col_renamed
        # display_df(df_corpus_topics.head())

        folder = "{}data/features/dynamic_features/{}".format(self.c.directory['save'],
                                                              self.dynamic_params_name())
        df_corpus_topics.to_csv('{}/{}_dominant_topic.csv'.format(folder,
                                                                  self.c.project_name),
                                encoding='utf-8')

        sent_topics_outdf_grpd = df_corpus_topics.groupby('Dominant_Topic')

        # sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

        sent_topics_sorteddf_mallet = pd.DataFrame()
        for i, grp in sent_topics_outdf_grpd:
            sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                     grp.sort_values(['Perc_Contribution'],
                                                                     ascending=[0]).head(self.c.num_sent)],
                                                    axis=0)

        # Reset Index
        sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)
        # Show
        sent_topics_sorteddf_mallet.to_csv('{}/{}_top_{}_samples_by_dominant_topics.csv'.format(folder,
                                                                                                self.c.project_name,
                                                                                                self.c.num_sent),
                                           index=False)
        # display_df(sent_topics_sorteddf_mallet.head(10))

    def construct_features(self):
        corpus_training, corpus_validation = self.corpus_creation()
        topic_model = self.topic_modeling(corpus_training)
        topic_training_df, topic_validation_df = self.topic_normalization(corpus_training, corpus_validation,
                                                                          topic_model)

        topic_training_df.reset_index(drop=True, inplace=True)
        topic_training_df = pd.concat([self.training_ids, topic_training_df], axis=1)

        topic_validation_df.reset_index(drop=True, inplace=True)
        topic_validation_df = pd.concat([self.validation_ids, topic_validation_df], axis=1)

        df_concat = pd.concat([topic_training_df, topic_validation_df], axis=0)
        df_concat.reset_index(drop=True, inplace=True)

        df_concat.to_csv('{}data/features/dynamic_features/{}.csv'.format(self.c.directory['save'],
                                                                          self.dynamic_params_name()),
                         index=False)

        self.save_topic_keywords(topic_model)

        data = pd.Series(topic_model[corpus_training])
        self.save_top_sample_dominant_topics(data)

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
    raise Failed('Input was not either dataframe or series!')


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
    instance = LDA(_context, experiment, feature_name,
                   df_training, df_validation,
                   training_ids, validation_ids)
    return instance.construct_features()


def retrieve_features(_context, experiment, feature_name):
    instance = LDA(_context, experiment, feature_name)
    return instance.retrieve_features()
