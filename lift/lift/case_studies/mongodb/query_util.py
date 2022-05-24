# Utils for generating field data
import random
import string
import numpy as np
import datetime
import pycountry
import faker
from random_words import RandomWords

countries = list(pycountry.countries)
fake = faker.Faker()
random_word_gen = RandomWords()


def random_string(self, min=1, max=1):
    """
    Generates random string within given range of length.

    :param min:
    :param max:
    :return:
    """
    elements = min + np.random.randint(low=0, high=max)

    return self.string_generation(elements)


def string_generation(size=1, chars=string.ascii_letters):
    return ''.join(random.choice(chars) for _ in range(size))


def random_int_array(length=1, max=1000):
    elements = 1 + np.random.randint(low=0, high=length)

    result = []
    for i in range(elements):
        result.append(np.random.randint(low=0, high=max))

    return result


def random_location():
    index = + np.random.randint(low=0, high=len(pycountry.countries))

    return countries[index].name


def random_word():
    """
    Uses random words module, install via 'pip install RandomWords'

    :return: A random word
    """
    return random_word_gen.random_word()


def first_name():
    """
    Uses random words module, install via 'pip install RandomWords'

    :return: A random word
    """
    return fake.first_name()


def last_name():
    """
    Uses random words module, install via 'pip install RandomWords'

    :return: A random word
    """
    return fake.last_name()


def random_date(years=1):
    """
    Creates a random date.

    :return:
    """
    day = np.random.randint(1,28)
    month = np.random.randint(1, 12)
    year = 2016 - np.random.randint(0, years)

    return datetime.datetime.today().replace(year=year, day=day, month=month)


def random_text(length=1):
    elements = 1 + np.random.randint(low=0, high=length)

    result = []
    for i in range(elements):
        result.append(random_word_gen.random_word())

    return result


def check_stage(index_used, query, stage):
    print('Index used = ', index_used)
    print('query = ', query)
    print('stage = ', stage)
    index_name = stage["indexName"]
    assert index_name in index_used, \
        "Index found {} is not in index used, allowed are: {}," \
        "check name formats?".format(
            index_name, index_used
        )
    index_used[index_name][0] += 1
    index_used[index_name][1].append(str(len(query.query_columns)))