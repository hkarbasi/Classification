##############		variables		##############
##################################################

# experiment_name = features + algorithm + params

project_name = 'ILLAE'

directory = dict(
    # master='/Users/habib/Documents/Argo/Habib/scratch/_RT/WomenInEngineering/',
    # save='/Users/habib/Documents/Argo/Habib/scratch/_RT/WomenInEngineering/',
    # master='/Users/habibkarbasian/Documents/Argo/Habib/scratch/_RT/_ILLAE/',
    # save='/Users/habibkarbasian/Documents/Argo/Habib/scratch/_RT/_ILLAE/',
    # master='/Users/habibkarbasian/Documents/_Courses/_Datasets/_Retweetability/_WomenInEngineering/',
    # save='/Users/habibkarbasian/Documents/_Courses/_Datasets/_Retweetability/_WomenInEngineering/',
    master='/Users/habibkarbasian/Documents/_Courses/_Datasets/_Retweetability/_ILLAE/',
    save='/Users/habibkarbasian/Documents/_Courses/_Datasets/_Retweetability/_ILLAE/',
    # mallet='/Users/habibkarbasian/Documents/_Courses/_Codes/_iNotebook/Mallet/mallet-2.0.8/bin/mallet',
    mallet='/Users/habibkarbasian/Documents/_Courses/_Codes/packages/Mallet-master_8_06/bin/mallet',

    dataset=dict(
        type='txt',  # xls csv txt
        sep='\t',
        filename='#ILookLikeAnEngineer-Tweets-Original',
    )
)
features = dict(
    static=dict(
        fffs=dict(
            type='meta',
            cols=dict(
                friends_count='numerical',
                followers_count='numerical',
                favourites_count='numerical',
                statuses_count='numerical',
            )
        ),
        listed=dict(
            type='meta',
            cols=dict(
                listed_count='numerical',
            )
        ),
        verified=dict(
            type='meta',
            cols=dict(
                verified='numerical',
            )
        ),
        character=dict(
            type='meta',
            cols=dict(
                text='length',
            )
        ),
        exclem=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=['!']
        ),
        question=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=['?']
        ),
        termspos=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=['great', 'like', 'excellent', 'rock on']
        ),
        termsneg=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=['suck', 'fuck', 'fail', 'eww']
        ),
        emopos=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=[':-)', ':)', ';-)']
        ),
        emoneg=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=[':-(', ':(']
        ),
        cusswords=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=['arse','ass','asshole','bastard','bitch','bollocks','child-fucker','christ on a bike',
                   'christ on a cracker','crap','cunt','damn','frigger','fuck','goddamn','godsdamn','hell','holy shit',
                   'horseshit','jesus christ','jesus fuck','jesus h. christ','jesus harold christ','jesus wept',
                   'jesus, mary and joseph','judas priest','motherfucker','nigga','nigger','prick','shit','shit ass',
                   'shitass','slut','son of a bitch','son of a motherless goat','son of a whore','sweet jesus','twat',]
        ),
        emojis=dict(
            type='meta',
            cols=dict(
                text='lookup',
            ),
            words=[':‑)',':)',':‑D',':D',':-))',':‑(',':(',':\'‑(',':\'(',':\'‑)',':\')','D‑\':',':‑O',':O',':-*',':*'
                ,';‑)',';)',':‑P',':P',':‑/',':/',':‑|',':|',':‑X',':X','O:‑)','O:)','>:‑)','>:)','|;‑)',':‑J','#‑)'
                ,'%‑)','%)',':‑###..',':###..','<:‑|','\',:-|',':-]',':]','8‑D','8D',':‑c',':c','D:<',':‑o',':o','*-)'
                ,'*)','X‑P','XP','://)','://3',':‑#',':#','0:‑3','}:‑)','}:)','|‑O','\',:-l',':-3',':3','x‑D','xD',':‑<'
                ,':<','D:',';‑]',';]','x‑p','xp',':‑&',':&','0:‑)','0:)','3:‑)','3:)',':->',':>','X‑D','XD',':‑[',':['
                ,'D8',':‑p',':p','8-)','8)','D;',':‑Þ',':Þ','>:3','>;3',':-}',':}','D=',':‑þ',':þ','DX',':‑b',':b']
        ),
        anew=dict(
            type='meta',
            cols=dict(
                text='anew',
            )
        ),
        url=dict(
            type='meta',
            cols=dict(
                external_url='boolean',
            )
        ),
        media=dict(
            type='meta',
            cols=dict(
                external_media_type='boolean',
            )
        ),
        mentions=dict(
            type='meta',
            cols=dict(
                user_mentions='boolean'
            )
        ),
        hashtag=dict(
            type='meta',
            cols=dict(
                hashtags='hashtag'
            )
        ),
        reply=dict(
            type='meta',
            cols=dict(
                in_reply_to_user_id='boolean'
            )
        ),
        age=dict(
            type='meta',
            cols=dict(
                profile_created_at_date='age_of_account'
            )
        ),
        liwc=dict(
            type='liwc',
            filename='#ILookLikeAnEngineer-Original-LIWC.csv'
        )
    ),
    dynamic=dict(
        bow_text=dict(
            type='bow',
            cols=['text'],
            replacements=[
                # Remove Emails
                ('\S*@\S*\s?', ' '),
                # Remove @text
                ('@\S+', ' '),
                # Remove new line characters
                ('\s+', ' '),
                # Remove URL
                ('http\S+', ' '),
                # Remove hashtag
                ('#', ' '),
            ],
            params=dict(
                no_below=[10],
                no_above=[0.5],
            )
        ),
        terms_text=dict(
            type='terms',
            cols=['text', 'retweet_count'],
            replacements=[
                # Remove Emails
                ('\S*@\S*\s?', ' '),
                # Remove @text
                ('@\S+', ' '),
                # Remove new line characters
                ('\s+', ' '),
                # Remove URL
                ('http\S+', ' '),
                # Remove hashtag
                ('#', ' '),
            ],
            params=dict(
                no_below=[10],
                no_above=[0.5],
            )
        ),
         hashtags=dict(
            type='community',
            cols=['hashtags'],
            params=dict(
                community_graph_type=[
                    # 'unweighted',
                    'weighted'
                ],
                community_feature_type=[
                    # 'binary',
                    'weighted',
                    # 'betweeness'
                ],
            )
        ),
        lda_hashtags=dict(
            type='lda',
            # cols=['text', 'hashtags'],
            cols=['hashtags'],
            replacements=[
            ],
            min_keyword_threshold=1,
            lang_detect=True,

            # LDA_iteration_threshold=10,
            # params=dict(
            #     no_below=[1],
            #     no_above=[0.5],
            #     LDA_topic_cutoff_threshold=[0.1],
            #     LDA_num_topics=list(range(5, 11, 5)),
            # )

            LDA_iteration_threshold=1000,
            params=dict(
                no_below=[1],
                no_above=[1],
                LDA_topic_cutoff_threshold=[0],
                LDA_num_topics=list(range(20, 21, 5)),
            )
        ),
        full_text=dict(
            type='lda',
            # cols=['text', 'hashtags'],
            cols=['text'],
            replacements=[
                # Remove Emails
                ('\S*@\S*\s?', ' '),
                # Remove @text
                ('@\S+', ' '),
                # Remove new line characters
                ('\s+', ' '),
                # Remove URL
                ('http\S+', ' '),
                # Remove hashtag
                ('#', ' '),
            ],
            min_keyword_threshold=1,
            lang_detect=True,

            LDA_iteration_threshold=1000,
            params=dict(
                no_below=[1],
                no_above=[1],
                LDA_topic_cutoff_threshold=[0],
                LDA_num_topics=list(range(20, 21, 5)),
            )

            # LDA_iteration_threshold=500,
            # params=dict(
            #     no_below=[1, 10],
            #     no_above=[0.5, 0.99],
            #     LDA_topic_cutoff_threshold=[0, 0.1],
            #     LDA_num_topics=list(range(5, 51, 5)),
            # )
        ),
    )
)


select_features = dict(
    static=[
        'fffs',
        'age',
        'listed',
        'verified',
        'character',
        'url',
        'media',
        'mentions',
        'reply',
        # 'exclem',
        # 'question',
        # 'termspos',
        # 'termsneg',
        # 'emopos',
        # 'emoneg',
        # 'anew',
        # 'cusswords',
        # 'emojis',
        'hashtag',
        'liwc'
    ],
    dynamic=[
        # 'bow_text',
        # 'terms_text',
        # 'hashtags',
        'lda_hashtags',
        # 'full_text'
    ]
)


algorithm = dict(
    name='svm',
    params_name='test',
    kernels=[
        'linear',
        'poly',
        'rbf',
        'sigmoid'
    ],
    params=dict(
        cost=[1],
        gamma=[0.1],
        coef0=[10],
        degree=[2, 3]
    )
)

# algorithm = dict(
#     name='svm',
#     params_name='final',
#     kernels=[
#         'linear',
#         'poly',
#         'rbf',
#         'sigmoid'
#     ],
#     params=dict(
#                 cost=[1, 10],
#                 gamma = [0.01, 1],
#                 coef0 = [0, 10],
#                 degree = [2, 3]

        # cost=[0.001, 0.01, 0.1, 1, 10, 100, 1000],
        # gamma=[0.001, 0.01, 0.1, 1],
        # coef0=[0.1, 1, 10, 100],
        # degree=[2, 3, 4]
    # )
# )

index_start = 0
index_step = 1

sampling = 'balanced'  # complete
label = dict(
    col='retweet_count',
    type='boolean'
)
id_name = 'tweet_id'
inner = 5
outer = 5
num_cores = 15
mem = 4000
# max_iter_optimization = 1000000
max_iter_optimization = -1
cache_size = 1000
seed = 1982
num_sent=20

check_date = True
min_date_val = '2014-05-01'
max_date_val = '2019-11-01'

# day:  %d 12
# month:  %b Aug    %B August   %m: 08
# year:  %y 99    %Y 1999
date_format = '%b %d %Y'  # timestamp
date_col = 'tweet_created_at_date'

gc = False
