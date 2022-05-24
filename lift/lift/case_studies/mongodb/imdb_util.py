import numpy as np

mongo_query_quote_char = "'"
mongo_query_delimiter = "#"

# Mapping of imdb attributes.
imdb_collection_info = {
  # "titleId": ["string", 2, 30],
  "ordering": ["int", 0, 1000000],
  "title": ["string", 2, 30],
  "region": ["string", 2, 30],
  "language" : ["string", 2, 30],
  "types": ["string_array", 25],
  # "attributes" : ["string_array", 25],
  "isOriginalTitle": ["bool"],
  # "tconst": ["string", 2, 30],
  "titleType": ["string", 2, 30],
  "primaryTitle": ["string", 2, 30],
  # "originalTitle": ["string", 2, 30],
  "isAdult": ["bool"],
  "startYear": ["date", 10],
  "endYear": ["date", 10],
  "runtimeMinutes": ["int", 0, 1000],
  "genres": ["string_array", 25],
  "averageRating":["int", 0, 1000000],
  "numVotes": ["int", 0, 1000000]
}

# db.imdb_all.aggregate(
#           [{$group: {
#               _id: "$title",
#               count: { $sum : 1 }
#             }}, {
#             $group: {
#               _id: "$_id",
#               count: { $sum : "$count" }
#             }},{
#               $out: "distinctCount"
#             }],
#          {allowDiskUse:true}
# )
# db.imdb_all.aggregate([{ $group: { _id: "$title"}  },{ $group: { _id: 1, count: { $sum: 1 } } } ])
IMDB_SELECTIVITY = {
    "ordering": 1.0 / 104.0,
    "title":  1.0 / 83925.0,
    "region": 1.0 / 247.0,
    "language": 1.0 / 93.0,
    "types": 1.0 / 17.0,
    "isOriginalTitle": 1.0 / 4.0,
    "titleType": 1.0 / 11.0,
    "primaryTitle": 1.0 / 4416772.0,
    "isAdult": 1.0 / 3.0,
    "startYear": 1.0 / 148.0,
    "endYear": 1.0 / 95.0,
    "runtimeMinutes": 1.0 / 788.0,
    "genres": 1.0 / 2199.0,
    "averageRating": 1.0 / 92.0,
    "numVotes": 1.0 / 16007.0,

}

field_type_operators = {
  "string": ["$eq"],
  "int": ['$eq', '$gt', '$lt', '$gte', '$lte'],
  "date": ['$eq', '$gt', '$lt'],
  "bool": ["$eq"],
  "string_array": ["$eq"]
}

# Test genre query.
genres = ["Animation", "Short", "Comedy", "Romance", "Fantasy", "Sci-Fi", "Horror", "Drama", "Mystery"]
language = ["en", "\\N"]
region = ["\\N", "US", "RU", "FR", "HU", "AD", "XEU", "XWW", "BR", "GB", "DE"]

primary_title = [
    "Unusual Cowboys", "Clodette", "Un bon bock"
    "Testimonis del Temps", "Nick", "The Girl from Mars",
    "The Inferno",  "Chinese Opium Den", "The Vagabond Sky"
    "Solos RPG", "Kingdoom", "Edison Kinetoscopic Record of a Sneeze",
    "The Waterer Watered", "Jim Corbett vs. Peter Courtney", "The Dreams I Have"
    "Persefone: Spiritual Migration", "Future Tense", "Printempo", "Fantastic Beasts and Where to Find Them 5",
    "The Masked man",  "100 Years", "The Falling Planet", "Avatar 4", "The Diamond from the Sky",
    "The Long Arm of the Law", "The Boundary Rider", "An Odyssey of the North", "The Brute",
    "Santo contra cerebro del mal", "Devil's Partner", "The Challenge of Ideas", "Murder on the Campus",
    "The Vision of William Blake", "Mara", "The Female Bunch", "Exklusiv!",  "Isle of the Snake People",
    "Story of a Girl Alone",  "Zorro, Rider of Vengeance",  "The Lost Angel",  "The Hour and Turn of Augusto Matraga",
    "Fornicon", "The Female Bunch", "Bruce's Deadly Fingers","The Cage", "Under the Gun", "Another Time, Another Place",
    "Dama de noche",  "Taro the Dragon Boy", "The Meanest Men in the West", "Swan Lake", "The Six Directions of Boxing",
    "Six Tickets to Hell", "Swan Lake", "Game of Death II", "Indeks",  "Mord in der Oper",  "Firecracker"

]
title = [
    "Baby's Meal", "The Cabinet of Mephistopheles", "Niagara Falls", "The Fisherman at the Stream",
    "Arrival of a Train at Vincennes Station", "Boat Leaving the Port",  "The Niagara Waterfall",
    "The Photographical Congress Arrives in Lyon",  "Tables Turned on the Gardener", "The Clown and His Dogs",
    "The Corbett-Courtney Fight",  "Blacksmith Scene", "Blacksmithing", "Employees Leaving the Lumière Factory",
    "The Sea", "Around a Bathing Hut",  "Opening of the Kiel Canal", "Fishing for Goldfish",  "The Dreyfus Affair",
    "The Laboratory of Mephistopheles", "The Miller and the Sweep",  "Above the Limit",  "King John",
    "Distributing a War Extra", "A Tour in Spain and Portugal", "The Miller and the Sweep", "La columna de fuego",
    "Game of Cards", "Italienischer Bauerntanz",  "Les forgerons",  "La mer", "The Clown and His Dogs",  "Miss Jerry",
    "The Sea", "Le clown et ses chiens"
]

title_types = ["movie", "short", "tvseries", "tvepisode", "video"]
types = ["alternative", "dvd", "festival", "tv", "video", "working", "original", "imdbDisplay"]

imdb_sampling_fns = {
    "isOriginalTitle": lambda: np.random.randint(0, 2),
    "isAdult": lambda: np.random.randint(0, 2),
    "runtimeMinutes": lambda: np.random.randint(5, 200),
    "ordering": lambda: np.random.randint(1, 13),
    "numVotes": lambda: np.random.randint(0, 100000),
    # E.g. 5.7
    "averageRating": lambda: round(np.random.rand() * 10, 1),
    "startYear": lambda: np.random.randint(1890, 2017),
    "endYear": lambda: np.random.randint(1890, 2017),
    "types": lambda : np.random.choice(types),
    "region": lambda: np.random.choice(region),
    "titleType": lambda: np.random.choice(title_types),
    "genres": lambda : np.random.choice(genres),
    "primaryTitle": lambda: np.random.choice(primary_title),
    "title": lambda: np.random.choice(title),
    "language": lambda: np.random.choice(language)
}