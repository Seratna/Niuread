__author__ = 'Antares'

import logging
import datetime
import mysql.connector
import numpy as np
import pandas as pd
from collaborative_filtering import learn


class NiureadRecommender(object):
    """

    """
    CONFIG_FILE_PATH = "connection.conf"
    DB_NAME = "onebook"
    TABLE_BOOK_INFO = "t_book_info"
    ATTR_BOOK_INFO_ID = "bookInfoId"
    TABLE_USER_INFO = "t_userInfo"
    ATTR_USER_INFO_ID = "userInfoId"
    TABLE_RATING_HISTORY = "t_userRatedBookScore"
    ATTR_RATING_SCORE = "score"
    TABLE_RECOMMENDATION_HISTORY = "t_userRecommendedBook"
    ATTR_RECOMMENDED_SCORE = "calculatedScore"

    NUM_FEATURES = 10
    REGULATION_LAMBDA = 0.5
    NUM_ITERATION = 100

    # DOUBAN_WEIGHT = 5

    def __init__(self):
        """

        """
        pass
        # TODO setup logger

    def recommend(self):
        """

        """
        # get data from database
        with MyConnection(option_files=self.CONFIG_FILE_PATH) as cnx:
            books = self.get_books(cnx)
            users = self.get_users(cnx)
            rating_history = self.get_rating_history(cnx)
            recommendation_history = self.get_recommendation_history(cnx)

        attr = ["book_index", "user_index", self.ATTR_RATING_SCORE]
        merged_rating = rating_history.merge(books).merge(users)[attr]
        indexed_rating = np.array(merged_rating)

        attr = ["book_index", "user_index"]
        merged_recommendation = recommendation_history.merge(books).merge(users)[attr]
        indexed_recommendation = np.array(merged_recommendation)

        print("books: \n", books, "\n")
        print("users: \n", users, "\n")
        print("rating history: \n", rating_history, "\n")
        print("recommendation history: \n", recommendation_history, "\n")
        print("merged rating: \n", merged_rating, "\n")
        print("indexed rating: \n", indexed_rating, "\n")
        print("merged recommendation: \n", merged_recommendation, "\n")
        print("indexed recommendation: \n", indexed_recommendation, "\n")

        num_books = books.shape[0]
        num_users = users.shape[0]

        # prepare arrays
        y = np.zeros(shape=(num_books, num_users), dtype="int32")
        r = np.zeros(shape=(num_books, num_users), dtype="int32")
        y[indexed_rating[:, 0], indexed_rating[:, 1]] = indexed_rating[:, 2]
        r[indexed_rating[:, 0], indexed_rating[:, 1]] = 1

        # start training
        theta, x, y_mean, cost, reg_cost = learn(shape_theta=(num_users, self.NUM_FEATURES),
                                                 shape_x=(num_books, self.NUM_FEATURES),
                                                 y=y,
                                                 r=r,
                                                 reg_lambda=self.REGULATION_LAMBDA,
                                                 n_iter=self.NUM_ITERATION)
        logging.debug("regulation cost is {:0.3}% of total cost".format((reg_cost/cost[-1])*100))

        # TODO calculate y_mean according to user rating and douban rating
        all_douban = np.array(books['doubanScore'])
        valid_douban = (all_douban-1)/9*4 + 1  # scale
        valid_douban[valid_douban<1] = 0
        valid_douban[np.isnan(valid_douban)] = 0
        valid_douban[valid_douban<=0] = 2.5
        valid_douban = valid_douban.reshape((-1, 1))

        mean = y_mean + valid_douban*(y_mean<=0)

        # generate prediction
        p = x.dot(theta.T) + mean.dot(np.ones((1, num_users)))

        print(y_mean[633, 0])
        print(valid_douban[633, 0])
        # generate recommendation
        p[indexed_recommendation[:, 0], indexed_recommendation[:, 1]] = 0
        p[indexed_rating[:, 0], indexed_rating[:, 1]] = 0
        prediction = np.argmax(p, axis=0)
        score = np.max(p, axis=0)

        recommendations = []
        date = datetime.datetime.today().strftime("%Y-%m-%d")
        for i in range(num_users):
            entry = (0,
                     users.iloc[i][self.ATTR_USER_INFO_ID],
                     books.iloc[prediction[i]][self.ATTR_BOOK_INFO_ID],
                     date,
                     score[i])
            recommendations.append(entry)
        print(recommendations)

        # push recommendations to database
        self.push_recommendations(recommendations)

    def get_books(self, cnx):
        """

        @param cnx:
        @return:
        """
        query = "SELECT {}, doubanScore FROM {}.{} WHERE offline=0".format(self.ATTR_BOOK_INFO_ID, self.DB_NAME, self.TABLE_BOOK_INFO)
        # # TODO for test ONLY
        # query = "SELECT {} FROM {}.{} WHERE bookInfoId >= 450".format(self.ATTR_BOOK_INFO_ID,
        #                                                               self.DB_NAME,
        #                                                               self.TABLE_BOOK_INFO)
        books = pd.read_sql(query, con=cnx)
        books["book_index"] = books.index

        return books

    def get_users(self, cnx):
        """

        @param cnx:
        @return:
        """
        query = "SELECT {} FROM {}.{}".format(self.ATTR_USER_INFO_ID, self.DB_NAME, self.TABLE_USER_INFO)
        users = pd.read_sql(query, con=cnx)
        users["user_index"] = users.index

        return users

    def get_rating_history(self, cnx):
        query = "SELECT {}, {}, {} FROM {}.{}".format(self.ATTR_USER_INFO_ID,
                                                      self.ATTR_BOOK_INFO_ID,
                                                      self.ATTR_RATING_SCORE,
                                                      self.DB_NAME, self.TABLE_RATING_HISTORY)
        rating_history = pd.read_sql(query, con=cnx)

        return rating_history

    def get_recommendation_history(self, cnx):
        query = "SELECT {}, {}, {} FROM {}.{}".format(self.ATTR_USER_INFO_ID,
                                                      self.ATTR_BOOK_INFO_ID,
                                                      self.ATTR_RECOMMENDED_SCORE,
                                                      self.DB_NAME, self.TABLE_RECOMMENDATION_HISTORY)
        recommendation_history = pd.read_sql(query, con=cnx)

        return recommendation_history

    def push_recommendations(self, recommendations):
        """
        push the recommendations generated from collaborative filtering to database

        @param recommendations: a list contains tuples of recommendation. a tuple should be like:
                                (0, userInfoId, bookInfoId, RecommendedDate, calculatedScore)
        """
        with MyConnection(option_files=self.CONFIG_FILE_PATH) as cnx:
            with MyCursor(cnx, buffered=True) as cursor:
                print("start pushing recommendations to {} users".format(len(recommendations)))
                for r in recommendations:
                    query = "INSERT INTO {}.{} VALUES {}".format(self.DB_NAME, self.TABLE_RECOMMENDATION_HISTORY, r)
                    print(query)
                    cursor.execute(query)
                cnx.commit()
                print("finished")

    # def fake_ratings(self):
        # with MyConnection(option_files=self.CONFIG_FILE_PATH) as cnx:
        #     with MyCursor(cnx, buffered=True) as cursor:
        #         for i in range(57):
        #             u = np.random.randint(5, 29)
        #             while True:
        #                 b = np.random.randint(1, 465)
        #                 if b != 172:
        #                     break
        #             r = np.random.randint(1, 10)
        #             t = (u, b, r)
        #             query = "INSERT INTO {}.{} VALUES {}".format(self.DB_NAME, self.TABLE_RATING_HISTORY, t)
        #             print(query)
        #             cursor.execute(query)
        #         cnx.commit()

    def test_query(self, query):
        with MyConnection(option_files=self.CONFIG_FILE_PATH) as cnx:
            with MyCursor(cnx, buffered=True) as cursor:
                print(query)
                cursor.execute(query)
                cnx.commit()
                print("finished")


class MyConnection(object):
    """
    this class is a wrapper of mysql.connector.connection.MySQLConnection,
    to enable the use of "with statement"
    """
    def __init__(self, *args, **kwargs):
        self.cnx = mysql.connector.connect(*args, **kwargs)

    def __enter__(self):
        logging.debug("MySQL connection entered")
        return self.cnx

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cnx.close()
        logging.debug("MySQL connection closed")


class MyCursor(object):
    """
    this class is a wrapper of mysql.connector.cursor.MySQLCursor,
    to enable the use of "with statement"
    """
    def __init__(self, cnx: mysql.connector.connection.MySQLConnection, *args, **kwargs):
        self.cursor = cnx.cursor(*args, **kwargs)

    def __enter__(self):
        logging.debug("MySQL cursor entered")
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cursor.close()
        logging.debug("MySQL cursor closed")


def main():
    logging.basicConfig(level=logging.DEBUG)
    nr = NiureadRecommender()
    nr.recommend()

if __name__ == '__main__':
    main()