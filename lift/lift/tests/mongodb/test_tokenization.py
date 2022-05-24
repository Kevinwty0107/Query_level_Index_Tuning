from lift.util.parsing_util import token_generator


def test_tokenization():
    tokens = ['registration_date', '$gt', 'follower_count', '$lt']
    query = {"registration_date": {"$gt": 5}, "follower_count": {"$lt": 4}}

    print(list(token_generator(op_dict=query, tokens=tokens)))


def test_sorted_tokenization():
    tokens = ['f1', 'f10', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f1',
              'f10', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', '$and', '$or',
              '$nor', '$not', '$eq', '$gt', '$lt', '$gte', '$lte', '$nin', 'sort', 'limit', 'count']

    query = {'$nor': [{'f10': {'$lte': {'$date': 1442068323727}}}, {'f4': {'$eq': 960545}}]}
    print(list(token_generator(op_dict=query, tokens=tokens)))


def test_action_tokenization():
    index_name = 'f2_-1_f3_-1'
    token_list = index_name.split('_')
    output_sequence = []

    # Converts field_1_field_-1 into (field, 1), (field, -1) ist of tuples
    index_tuple_list = list(zip(token_list[0::2], token_list[1::2]))

    for index_tuple in index_tuple_list:
        string_index_tuple = '{}_{}'.format(index_tuple[0], index_tuple[1])
        output_sequence.append(string_index_tuple)

    print(output_sequence)