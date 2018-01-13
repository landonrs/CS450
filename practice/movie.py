import random
from random import Random
import decimal
import numpy as np

class Movie:

    def __init__(self, title = "", year = 0, runtime = 0):
        self.title = title
        self.year = year
        if runtime < 0:
            self.runtime = 0
        else:
            self.runtime = runtime

    def __repr__(self):
        return self.title + " (" + str(self.year) + ") - " + str(self.runtime) + " mins"

    def get_runtime_hours_min(self):
        return str(int(self.runtime / 60)) + " hrs " + str(self.runtime % 60) + " mins"

def create_dictionary(movie_list):
    movie_dictionary = {}
    for movie in movie_list:
        movie_dictionary[movie.title] = float(decimal.Decimal(random.randrange(0, 500) / 100))
    return movie_dictionary

def create_movie_list():
    movie_list = []

    movie1 = Movie("Home Alone", 1992, 145)
    movie2 = Movie("Home Alone 2", 1994, 175)
    movie3 = Movie("Home Alone 3", 1996, 145)
    movie4 = Movie("Home Alone 4", 1998, 160)
    movie5 = Movie("Home Alone 5", 2003, 145)

    movie_list.append(movie1)
    movie_list.append(movie2)
    movie_list.append(movie3)
    movie_list.append(movie4)
    movie_list.append(movie5)

    return movie_list


def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    # random = Random()

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100

        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    np.set_printoptions(precision=2)
    np.set_printoptions(suppress=True)
    return array

def main():
    movie_list = create_movie_list()
    new_movie_list = [movie for movie in movie_list if movie.runtime > 150]
    for movie in movie_list:
        print(movie)
    for movie in new_movie_list:
        print(movie)
    movie_dictionary = create_dictionary(movie_list)
    for key, value in movie_dictionary.items():
        print("{}: {} stars".format(key, value))
    numArray = get_movie_data()
    print(numArray.shape[0])
    print(numArray.shape[1])
    print(numArray)
    sliced_rows = numArray[0:2]
    print(sliced_rows)
    sliced_col = numArray[numArray.shape[0] - 1][1:]
    array = numArray[:, 1]
    print(sliced_col)
    print(array)


main()





# movie = Movie("Jurassic World", 2015, 124)
# print(movie)
# print(movie.get_runtime_hours_min())
# #
# movie2 = Movie()
# print(movie2)