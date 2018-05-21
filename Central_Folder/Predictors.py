import pandas as pd
import numpy as np
import os

import sys
sys.path.append("../Data/Map")

cwd = os.getcwd()
os.chdir(r'C:/Users/ekoulier/Desktop/Infectious Diseases/Data/Map')
post_to_mun = pd.read_csv('Post_to_Mun.txt')
mun_to_GGD = pd.read_csv('Mun_to_GGD.csv', sep=';')
mun_to_GGD.replace({'Nuenen Gerwen en Nederwetten':
                    'Nuenen, Gerwen en Nederwetten'}, regex=True)
os.chdir(cwd)


class monthly_transfrom(object):
    """ A class to create the aggregated dataframe. Two methods provided:
    The aggregation by GGD and the aggregation by muicipality.
    """

    def __init__(self, df):
        """ Initialize the dataframe object. The dataframe has the following
        format:
        Date  PostCode  Sex  Year of Birth etc
        """
        self.df = df
        data = self.df.copy()

        # First work with the Date Column
        try:
            data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')
        except ValueError:
            data['Date'] = pd.to_datetime(data['Date'], format='%d-%m-%Y')

        data['Date'] = data['Date'].dt.to_period('M')

        # Create the Count column for the pivot table
        data['Count'] = 1

        self.df = data

    def find_mun(self):
        """ Creates the Municipality column to the DataFrame. The Dataframe
        should not have a Municipality column and should have a column named
        PostCode.
        """

        try:
            assert 'Municipality' not in self.df.columns
            assert 'PostCode' in self.df.columns

            self.df = pd.merge(self.df, post_to_mun, on='PostCode')

        except AssertionError:
            print('There is already a column named Municipality or there is' +
                  + 'no column named PostCode')

    def find_GGD(self):
        """ Given the fact that there is a Municipality column, this method
        finds the corresponding GGD department.
        """

        try:
            assert 'GGD' not in self.df.columns
            assert 'Municipality' in self.df.columns

            self.df = pd.merge(self.df, mun_to_GGD, on='Municipality')

        except AssertionError:
            print('There is already a column named GGD or there is' +
                  + 'no column named Municipality')

    def monthly_municipality(self, mun_index=True):
        """
        A function that creates the pivot dataframe:
        Month    Oss  Tilburg  Breda
        2004-2    1      3       0
        2004-3    2      0       1
        Parameters
        -------------------------
        df: Pandas DataFrame
            It must have at least two columns named Municipality and Date.
            It is the general appended dateframe that has the following format:
            index Municipality Date       Sex
            0      Oss         2005-8  Man
            1      Eindhoven   2016-2  Vraouw
        mun_index: Bool
            If False then the dataframe to be returned will be the following:
            index Date     Breda   Eindhoven
            0     2004-1     2         8
            1     2004-2     5         0
        """
        try:
            assert 'Municipality' in self.df.columns
            assert 'Date' in self.df.columns

            data = self.df.copy()

            # Select the relevant columns
            data = data[['Date', 'Municipality', 'Count']]

            # Create the pivot table
            if mun_index:
                Mun_df = data.pivot_table(index='Municipality', columns='Date',
                                          values='Count', aggfunc=np.sum, fill_value=0)
            else:
                Mun_df = data.pivot_table(index='Date', columns='Municipality',
                                          values='Count', aggfunc=np.sum, fill_value=0)

            # Delete the extra heading of the pivot table
            Mun_df = pd.DataFrame(Mun_df.to_records())

            return Mun_df

        except AssertionError:
            print('There sould be a Date and a Municipality column')

    def monthly_GGD(self, GGD_index=False):
        """
        A function that creates the pivot dataframe:
        Month    HVB  WB  BZO
        2004-2    1    3   0
        2004-3    2    0   1
        Parameters
        -------------------------
        df: Pandas DataFrame
            It must have at least two columns named GGD and Date.
            It is the general appended dateframe that has the following format:
            index Municipality Date       Sex     GGD
            0      Oss         2005-8  Man     HVB
            1      Eindhoven   2016-2  Vraouw  BZO
        GGD_index: Bool
            If false the index of the dataframe is the Date it loos like:
            index Date     HVB  BZO  WB
            0     2008-04   1    2    1
            1     2008-05   0    5    6
        """

        try:
            assert 'Municipality' in self.df.columns
            assert 'Date' in self.df.columns

            data = self.df.copy()

            if GGD_index:
                GGD_df = data.pivot_table(index='GGD', columns='Date',
                                          values='Count', aggfunc=np.sum,
                                          fill_value=0)
            else:
                GGD_df = data.pivot_table(index='Date', columns='GGD',
                                          values='Count', aggfunc=np.sum,
                                          fill_value=0)

            GGD_df = pd.DataFrame(GGD_df.to_records())

            return GGD_df

        except AssertionError:
            print('There sould be a Date and a GGD column')
