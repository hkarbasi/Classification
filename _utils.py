##############		functions		##############
###################################################

from _imports import *
from _params import *

lock_print = Lock()
lock_count = Lock()
counter_sync = 1


def str_timestamp_to_date(date):
    try:
        return datetime.fromtimestamp(int(date))
    except ValueError as e:
        #         count_sync(10)
        #         return default_date
        raise Failed('{} has issue!\n{}'.format(date, e))


def str_timestamp_to_date_df(df):
    return df.apply(str_timestamp_to_date)


def str_date_to_date_df(df, date_format=None):
    try:
        if date_format is None:
            return pd.to_datetime(df, infer_datetime_format=True)
        return pd.to_datetime(df, format=date_format)
    except ValueError as e:
        #         print_asynch(df)
        raise Failed('There was an issue with dataframe!\n{}'.format(e))


def date_to_timestamp(date):
    try:
        return int(date.replace(tzinfo=timezone.utc).timestamp())
    except Exception as e:
        raise Failed('{} has type with {} value!\n{}'.format(type(date), date, e))


def date_to_timestamp_df(df):
    return df.apply(date_to_timestamp)


def count_sync(interval):
    global counter_sync
    lock_count.acquire()
    if (counter_sync % interval) == 0:
        print_asynch('{}'.format(counter_sync))
    counter_sync += 1
    lock_count.release()


def print_asynch(text):
    global lock_print
    lock_print.acquire()
    print(text)
    lock_print.release()


def display_df(df):
    if isinstance(df, pd.DataFrame):
        display(HTML(df.to_html()))
    else:
        print('input is not a dataframe to be displayed!')


def init_parallel(*args, **kwds):  # init_parallel(True/False, context):
    pandarallel.initialize(
        progress_bar=args[0] if args[1].progress_bar_allowed else False,
        shm_size_mb=args[1].mem // args[1].num_cores if len(kwds) == 0 else kwds['size'],
        nb_workers=args[1].num_cores if len(kwds) == 0 else kwds['cores']
    )


def decode_tag_sub_v0(tag, context):
    if re.match('zzzl\w+lzzz', tag) is None:
        return tag
    tag = tag.replace('zzz', '')
    tag = tag.strip()
    # print(tag)
    nums = tag[1:-1].split('l')
    final_num = 0
    for num in nums:
        # if num in num_dict.values():
        try:
            final_num = final_num * 10 + list(context.num_dict.values()).index(num)
        except ValueError as e:
            raise Failed('{} was not in list!\n{}\n'.format(num, tag, e))
        # else:
        #     print('{} original tag, vs {} index'.format(original_tag, num))
        #     return original_tag
    return context.tags_dict[final_num]


def decode_tag_sub(tag, context):
    # if re.match('zzz\w+zzz', tag) is None:
    #     return tag
    tag = tag.replace('zzzl', ' ')
    tag = tag.replace('lzzz', ' ')
    tag = tag.strip()
    # print(tag)
    #     nums = tag[1:-1].split('l')
    words = tag.split()
    tag = words[0]
    if (len(words) > 1):
        tag = words[1]

    nums = tag.split('l')
    final_num = 0
    for num in nums:
        # if num in num_dict.values():
        try:
            final_num = final_num * 10 + list(context.num_dict.values()).index(num)
        except ValueError as e:
            raise Failed('{} was not in list!\n{}\n'.format(num, tag, e))
        # else:
        #     print('{} original tag, vs {} index'.format(original_tag, num))
        #     return original_tag
    if (len(words) == 1):
        return context.tags_dict[final_num]
    return '{} {}'.format(words[0], context.tags_dict[final_num])


def decode_tag(tag, context):
    if context.tags is None:
        return tag
    t = tag
    m = re.search('zzz(.+?)zzz', t)
    try:
        while m:
            t = t.replace(m.group(0), ' ' + decode_tag_sub(m.group(0), context) + ' ')
            m = re.search('zzz(.+?)zzz', t)
    except ValueError as e:
        print(tag)
        raise Failed('{} was wrongly coded!\n{}'.format(tag, e))
    # t = re.sub('\s+', ' ', t)
    return t.strip()


def param_setup(args, context):
    global params
    try:
        print('args = {}'.format(args))
        opts, args_l = getopt.getopt(args,
                                     "",
                                     [p + "=" for p in params])
    except getopt.GetoptError as e:
        print('{}\nformat: file.py {}'.format(e, ' '.join(['[--' + p + '= ]' for p in params])))
        sys.exit(2)
    for opt, arg in opts:
        for param in params:
            if opt == '--' + param:
                setattr(context, param, int(arg))
                print('{} is set to {}'.format(param, arg))


def param_setup_ipython(args, context):
    global params
    for param in params:
        if param in args:
            setattr(context, param, args[param])
            print('{} is set to {}'.format(param, args[param]))


def save_df_to_sql(df, save_loc, db_name, index=False):
    engine = create_engine('sqlite:///' + save_loc, echo=False)
    df.to_sql(db_name, con=engine, if_exists='replace', index=index)


def experiment_name(context):
    return '{}_{}_{}_{}_{}'.format(context.project_name,
                                   context.algorithm['params_name'],
                                   context.algorithm['name'],
                                   '_'.join(context.select_features['static']),
                                   '_'.join('{}({})'.format(context.features['dynamic'][feature]['type'], feature) \
                                            for feature in context.select_features['dynamic']),
                                   )


def mkdir(path_to_dir):
    try:
        os.makedirs(path_to_dir)
        print('folder {} has been created.'.format(path_to_dir))
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
        print('folder {} already exists!'.format(path_to_dir))


def rmdir(path_to_dir):
    try:
        shutil.rmtree(path_to_dir)
    except FileNotFoundError as e:
        pass


def data_prepropressing(df_ids_ref, df_features_ref, df_class_ref, context, imputer=None, scaler=None):

    df_features = pd.merge(df_ids_ref, df_features_ref, on=context.id_name, how='left')
    df_features.drop([context.id_name], axis=1, inplace=True)
    df_class = pd.merge(df_ids_ref, df_class_ref, on=context.id_name, how='left')
    df_class.drop([context.id_name], axis=1, inplace=True)

    if imputer is None:
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean').fit(df_features)
    df_features = imputer.transform(df_features)

    if scaler is None:
        scaler = preprocessing.StandardScaler().fit(df_features)
    df_features = scaler.transform(df_features)

    return dict(
        features=df_features,
        labels=df_class,
        imputer=imputer,
        scaler=scaler
    )


def construct_static_features(context):
    if len(context.select_features['static']) == 0:
        return pd.DataFrame()
    df_features = pd.read_csv('{}data/features/{}_feature_{}.csv'.format(context.directory['save'],
                                                                         context.project_name,
                                                                         context.select_features['static'][0]
                                                                         ))

    for feature_name in context.select_features['static'][1:]:
        df = pd.read_csv('{}data/features/{}_feature_{}.csv'.format(context.directory['save'],
                                                                    context.project_name,
                                                                    feature_name
                                                                    ))
        df_features = pd.merge(df_features, df, on=context.id_name, how='left')

    return df_features


class Failed(Exception):
    pass


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def for_parallel(func, index_list, num_cores, params=None, gc_cleaner=False):
    print("Creating %i (daemon) workers and jobs in child." % num_cores)
    # pool = MyPool(min(num_cores, len(index_list)))
    pool = MyPool(num_cores)
    if params is None:
        result = pool.map(func, index_list)
    else:
        result = pool.map(functools.partial(func, **params), index_list)

    pool.close()
    pool.join()
    del pool
    if gc_cleaner:
        gc.collect()
    return result


# with params, make sure the series/df should be the last in the func not first!!
def parallelize_df(df, func, context, params=None):
    # num_partitions = 10 * context.num_cores
    num_partitions = min(1 * context.num_cores, len(df))
    df_split = np.array_split(df, num_partitions)
    print('parallel: {} partitions with {} cores for {}'.format(num_partitions, context.num_cores, func.__name__))
    pool = Pool(min(context.num_cores, len(df)), maxtasksperchild=1)
    if params is None:
        df = pd.concat(pool.map(func, df_split), ignore_index=True)
    else:
        df = pd.concat(pool.map(functools.partial(func, *params), df_split), ignore_index=True)
        # df = pd.concat(pool.map(lambda p: func(p, **params), df_split), ignore_index=True)

    pool.close()
    pool.join()
    del pool
    gc.collect()
    return df


def parallelize_df_dask(df, func, context):
    num_partitions = min(context.num_cores, len(df))
    # df=copy.deepcopy(df_input)
    df = dd.from_pandas(df, npartitions=num_partitions)
    print('parallel: {} partitions with {} cores for {}'.format(num_partitions, context.num_cores, func.__name__))
    return df.map_partitions(func).compute(scheduler='processes')


class Failed(Exception):
    pass
