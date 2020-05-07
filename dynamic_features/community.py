from _imports import *
from _utils import *


def btwn_pool_pickable(g_tuple):
    return nx.betweenness_centrality_source(*g_tuple)


class Community(object):

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

    def partitions(self, nodes, n):
        # Partitions the nodes into n subsets"
        nodes_iter = iter(nodes)
        while True:
            partition = tuple(itertools.islice(nodes_iter, n))
            if not partition:
                return
            yield partition

    def btwn_pool(self, g_tuple):
        return nx.betweenness_centrality_source(*g_tuple)

    def between_parallel(self, graph, processes):
        p = Pool(processes=processes)
        # p = MyPool(processes=processes)
        p.daemon = False
        part_generator = 4 * len(p._pool)
        node_partitions = list(self.partitions(graph.nodes(), int(len(graph) / part_generator)
        if len(graph) > part_generator else len(graph)))
        num_partitions = len(node_partitions)

        # bet_map = p.map(btwn_pool_pickable,
        #                 zip(
        #                     [graph] * num_partitions,
        #                     [True] * num_partitions,
        #                     ['Weight' if self.experiment[self.feature_name+'_community_graph_type'] == 'weighted' else None] * num_partitions,
        #                     node_partitions))

        # bet_map = p.map(self.btwn_pool,
        #                 zip([graph] * num_partitions,
        #                     [True] * num_partitions,
        #                     ['Weight' if self.experiment[self.feature_name+'_community_graph_type'] == 'weighted' else None] * num_partitions,
        #                     node_partitions))

        bet_map = list(map(self.btwn_pool,
                           zip([graph] * num_partitions,
                               [True] * num_partitions,
                               ['Weight' if self.experiment[
                                                self.feature_name + '_community_graph_type'] == 'weighted' else None] * num_partitions,
                               node_partitions))
                       )

        bt_c = bet_map[0]
        for bt in bet_map[1:]:
            for n in bt:
                bt_c[n] += bt[n]
        p.close()
        p.join()
        del p
        return bt_c

    # def lookup(hashtag_id):
    def lookup(self, hashtag_id, hashtag_lookup):
        return hashtag_lookup[int(hashtag_id)]

    # def lookup_df(df):
    #     return df.apply(lookup)
    def lookup_df(self, df, hashtag_lookup):
        return df.apply(self.lookup, hashtag_lookup=hashtag_lookup)

    def lookup_col(self, col, hashtag_lookup):
        # def lookup_col(col):
        hashtag_ids = list()
        for hashtag in col:
            hashtag_ids.append(hashtag_lookup[hashtag])
        return hashtag_ids

    def item_first(self, col):
        return col[0]

    def item_second(self, col):
        return col[1]

    def construct_graph_hashtag_lookup(self, df_hashtags):
        edges_list = list()
        hashtags_all = set()

        for hashtags in df_hashtags:
            if not isinstance(hashtags, str):
                splits=['']
            else:
                splits = hashtags.split(',')
            hashtag_unique = set()

            for split in splits:
                hashtag = split.strip().lower()
                hashtag_unique.add(hashtag)
                hashtags_all.add(hashtag)

            for edge in itertools.combinations(hashtag_unique, 2):
                edges_list.append(edge)

        hashtag_lookup = dict()
        counter = 0
        for hashtag in hashtags_all:
            if hashtag not in hashtag_lookup:
                hashtag_lookup[hashtag] = counter
                counter += 1
        hashtag_lookup_df = pd.DataFrame(hashtag_lookup.items(), columns=['Hashtag', 'Id'])
        if self.experiment[self.feature_name + '_community_graph_type'] == 'weighted':
            edge_weight_dict = dict()

            for edge in edges_list:
                if (edge not in edge_weight_dict) and (tuple(reversed(edge)) not in edge_weight_dict):
                    edge_weight_dict[edge] = 1
                    continue
                if edge in edge_weight_dict:
                    edge_weight_dict[edge] += 1
                else:
                    edge_weight_dict[tuple(reversed(edge))] += 1

            edgelist_df = pd.DataFrame(edge_weight_dict.items(), columns=['Tuple', 'W'])
            edgelist_df['Source'] = edgelist_df['Tuple'].apply(self.item_first)
            edgelist_df['Target'] = edgelist_df['Tuple'].apply(self.item_second)
            edgelist_df['Weight'] = edgelist_df['W']
            edgelist_df = edgelist_df.iloc[:, 2:]
            #
            #         display_df(edgelist_df.head(10))
            #         edgelist_df[['Source', 'Target']] = edgelist_df[['Source', 'Target']].apply(lookup_col, axis=0)
            edgelist_df[['Source', 'Target']] = edgelist_df[['Source', 'Target']].apply(self.lookup_col,
                                                                                        hashtag_lookup=hashtag_lookup,
                                                                                        axis=0)
            graph = nx.from_pandas_edgelist(edgelist_df, 'Source', 'Target', edge_attr='Weight',
                                                     create_using=None)
        else:
            edgelist_df = pd.DataFrame(edges_list, columns=['Source', 'Target'])
            edgelist_df = edgelist_df.apply(self.lookup_col, hashtag_lookup=hashtag_lookup, axis=0)
            graph = nx.from_pandas_edgelist(edgelist_df, 'Source', 'Target', edge_attr=None, create_using=None)
        return (graph, hashtag_lookup_df)

    def calculate_betweeness(self, graph):
        betweeness_dict = None
        if self.experiment[self.feature_name + '_community_feature_type'] == 'betweeness':
            betweeness_results = self.between_parallel(graph, self.c.num_cores)
            betweeness = pd.DataFrame(betweeness_results.items(), columns=['Hashtag', 'Between Centrality'])
            hashtag_lookup = dict(zip(self.hashtag_lookup_df['Id'], self.hashtag_lookup_df['Hashtag']))
            betweeness['Hashtag'] = betweeness['Hashtag'].apply(self.lookup, hashtag_lookup=hashtag_lookup)
            betweeness_dict = dict(zip(betweeness['Hashtag'], betweeness['Between Centrality']))
        return betweeness_dict

    def detect_communities(self, graph):
        if self.experiment[self.feature_name + '_community_graph_type'] == 'weighted':
            parts = community.best_partition(graph, weight='Weight', random_state=self.c.seed)
        else:
            parts = community.best_partition(graph, random_state=self.c.seed)

        # values = [parts.get(node) for node in graph.nodes()]

        commuities = dict()
        list_parts = list()
        parts_set = set(parts.values())

        print('Number of communities = {}'.format(len(parts_set)))

        for com in parts_set:
            #     count = count + 1.
            list_nodes = [nodes for nodes in parts.keys()
                          if parts[nodes] == com]
            #     print('{} : {}'.format(com, list_nodes))

            list_parts.append(list_nodes)

        list_parts = sorted(list_parts, key=len, reverse=True)
        list_parts_converted = list()
        hashtag_lookup = dict(zip(self.hashtag_lookup_df['Id'], self.hashtag_lookup_df['Hashtag']))
        for l in list_parts:
            list_parts_converted.append([hashtag_lookup[int(index)] for index in l])

        for index, part in enumerate(list_parts_converted):
            final_str = part[0]
            for i in part[1:]:
                final_str += ' ' + i
            commuities[index] = final_str

        membership_dict = dict()

        for index, community_group in enumerate(list_parts_converted):
            for hashtag in community_group:
                if hashtag not in membership_dict:
                    membership_dict[hashtag] = index
                else:
                    Failed('duplicate found!')

        df_communites = pd.DataFrame(membership_dict.values(), columns=['Hashtag Communities'])
        df_communites.index = membership_dict.keys()

        folder = "{}data/features/dynamic_features/{}".format(self.c.directory['save'],
                                                              self.dynamic_params_name())
        mkdir(folder)
        df_communites.to_csv('{}/{}_communities.csv'.format(folder,
                                                            self.c.project_name),
                             encoding='utf-8')

        return membership_dict

    def community_feature_constructor(self, hashtags):
        membership_dict = self.membership_dict
        betweeness_dict = self.betweeness_dict
        if not isinstance(hashtags, str):
            splits = ['']
        else:
            splits = hashtags.split(',')
        hashtag_unique = set()
        feature = [0] * len(set(membership_dict.values()))

        for split in splits:
            hashtag = split.strip().lower()
            hashtag_unique.add(hashtag)

        if self.experiment[self.feature_name + '_community_feature_type'] == 'binary':
            for hashtag in hashtag_unique:
                if hashtag in membership_dict:
                    feature[membership_dict[hashtag]] = 1
        elif self.experiment[self.feature_name + '_community_feature_type'] == 'weighted':
            for hashtag in hashtag_unique:
                if hashtag in membership_dict:
                    feature[membership_dict[hashtag]] += 1
        elif self.experiment[self.feature_name + '_community_feature_type'] == 'betweeness':
            for hashtag in hashtag_unique:
                if hashtag in membership_dict:
                    feature[membership_dict[hashtag]] += betweeness_dict[hashtag]
        else:
            Failed('community feature type is not correct. ({})'.format(
                self.experiment[self.feature_name + '_community_feature_type']))
        return pd.Series(feature)

    def community_feature_constructor_df(self, df):
        return df.apply(self.community_feature_constructor)

    def dynamic_params_name(self, retrieval_mode=False):
        if not retrieval_mode:
            feature_name = self.experiment['name']
            feature_type = self.experiment['type']
        else:
            feature_name = self.feature_name
            feature_type = self.c.features['dynamic'][self.feature_name]['type']

        params_names = '_'.join(str(1.0*self.experiment[feature_name + '_' + param_name])
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
        graph, hashtag_lookup_df = self.construct_graph_hashtag_lookup(self.df_training['nodes'])
        self.hashtag_lookup_df = hashtag_lookup_df
        self.betweeness_dict = self.calculate_betweeness(graph)
        self.membership_dict = self.detect_communities(graph)

        community_training_df = self.df_training['nodes'].apply(self.community_feature_constructor)
        # init_parallel(False, self.c)
        # community_training_df = self.df_training['nodes'].parallel_apply(self.community_feature_constructor)
        # community_training_df = parallelize_df(self.df_training['nodes'],
        #                                                  self.community_feature_constructor_df,
        #                                                  self.c
        #                                                  )
        community_training_df.reset_index(drop=True, inplace=True)
        community_training_df = pd.concat([self.training_ids, community_training_df], axis=1)

        community_validation_df = self.df_validation['nodes'].apply(self.community_feature_constructor)

        # community_validation_df = parallelize_df(self.df_validation['nodes'],
        #                                                    self.community_feature_constructor_df,
        #                                                    self.c
        #                                                    )
        community_validation_df.reset_index(drop=True, inplace=True)
        community_validation_df = pd.concat([self.validation_ids, community_validation_df], axis=1)

        df_concat = pd.concat([community_training_df, community_validation_df], axis=0)
        df_concat.reset_index(drop=True, inplace=True)

        df_concat.to_csv('{}data/features/dynamic_features/{}.csv'.format(self.c.directory['save'],
                                                                          self.dynamic_params_name()),
                         index=False)

        return df_concat

    def retrieve_features(self):
        df=pd.read_csv('{}data/features/dynamic_features/{}.csv'.format(self.c.directory['save'],
                                                                        self.dynamic_params_name(True)))
        return df


def prep_input(df, context):
    return df


def prep_features(df, feature, context):
    df.columns = [context.id_name, 'nodes']
    return df


def construct_features(_context, experiment, feature_name,
                       df_training, df_validation,
                       training_ids, validation_ids):
    instance = Community(_context, experiment, feature_name,
                         df_training, df_validation,
                         training_ids, validation_ids)
    return instance.construct_features()


def retrieve_features(_context, experiment, feature_name):
    instance = Community(_context, experiment, feature_name)
    return instance.retrieve_features()
