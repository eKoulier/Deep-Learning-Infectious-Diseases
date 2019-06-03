import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import copy as cp

import sys
sys.path.append("../Data/Map")

# First we read the Files to be use
cwd = os.getcwd()
os.chdir(r'../Data/Map')
post_to_mun = pd.read_csv('Post_to_Mun.txt', sep=';')
mun_to_GGD = pd.read_csv('Mun_to_GGD.csv', sep=';')
mun_to_GGD['Municipality'] = mun_to_GGD['Municipality'].replace('Nuenen',
                                                                'Nuenen, Gerwen en Nederwetten')
shp_netherlands = gpd.read_file('GEO.Gemeente_2015.shp')
os.chdir(cwd)


class MonthlyTransform(object):
    """ A class to create the aggregated dataframe. Two methods provided:
    The aggregation by GGD and the aggregation by municipality.
    """

    def __init__(self, df):
        """ Initialize the dataframe object. The dataframe has the following
        format:
        Date  PostCode  Sex  Year of Birth etc
        Parameters
        -------------------------
        df: Pandas DataFrame
            It must have at least two columns named Municipality and Date.
            It is the general appended dateframe that has the following format:
            index Municipality  Date           Sex
            0      Oss          2005-8-15      Man
            1      Eindhoven    2016-2-22      Vraouw
        """
        self.df = df
        df_data = self.df.copy()

        # First work with the Date Column
        try:
            df_data['Date'] = pd.to_datetime(df_data['Date'], format='%Y-%m-%d')
        except ValueError:
            df_data['Date'] = pd.to_datetime(df_data['Date'], format='%d-%m-%Y')

        df_data['Date'] = df_data['Date'].dt.to_period('M')
        df_data = df_data[df_data["Date"].dt.year >= 2004]

        # Create the Count column for the pivot table
        df_data['Count'] = 1

        time_df = pd.DataFrame()
        time_df['Date'] = pd.period_range(start=df_data['Date'].min(),
                                          end=df_data['Date'].max(), freq='M')

        # Probably there are missing months in the data and this is why we do the following
        df_time_extended = time_df.merge(df_data, on=['Date'], how='outer')
        df_time_extended['Count'] = df_time_extended['Count'].fillna(0)
        df_time_extended['PostCode'] = df_time_extended['PostCode'].fillna(5731)

        self.df = df_time_extended

    def find_mun(self):
        """ Creates the Municipality column to the DataFrame. The Data frame
        should have a column named PostCode.
        """

        try:
            assert 'PostCode' in self.df.columns

            if 'Municipality' in self.df.columns:
                del self.df['Municipality']

            self.df = pd.merge(self.df, post_to_mun, on='PostCode')
            self.df['Municipality'] = self.df['Municipality'].replace('Nuenen',
                                                                      'Nuenen, Gerwen en Nederwetten')

        except AssertionError:
            print('There is no column named PostCode.')

    def find_GGD(self):
        """ Given the fact that there is a Municipality column, this method
        finds the corresponding GGD department.
        """

        try:
            assert 'Municipality' in self.df.columns

            if 'GGD' in self.df.columns:
                del self.df['GGD']

            self.df = pd.merge(self.df, mun_to_GGD, on='Municipality')

        except AssertionError:
            print('There in no column named PostCode.')

    def monthly_municipality(self, mun_index=True, Brabant=True):
        """
        A function that creates the pivot dataframe:
        Month    Oss  Tilburg  Breda
        2004-2    1      3       0
        2004-3    2      0       1
        Parameters
        -------------------------
        self.df: Pandas DataFrame
            It must have at least two columns named Municipality and Date.
            It is the general appended dateframe that has the following format:
            index Municipality Date       Sex
            0      Oss         2005-8     Man
            1      Eindhoven   2016-2     Vraouw
        mun_index: Bool
            If False then the dataframe to be returned will be the following:
            index Date     Breda   Eindhoven
            0     2004-1     2         8
            1     2004-2     5         0
        Brabant: Bool
            If True, we limit ourselves to the Noord-Brabant Municipalities.
            Good to have it to build the interactive map.
        """

        assert 'Municipality' in self.df.columns,\
               'There sould be a Municipality column'
        assert 'Date' in self.df.columns,\
               'There should be a Date column'

        data = self.df.copy()

        # Select the relevant columns
        data = data[['Date', 'Municipality', 'Count']]

        # Create the pivot table

        mun_df = data.pivot_table(index='Municipality',
                                  columns='Date', values='Count',
                                  aggfunc=np.sum, fill_value=0)

        # Delete the extra heading of the pivot table
        mun_df = pd.DataFrame(mun_df.to_records())


        if Brabant:
            br_mun_list = mun_to_GGD['Municipality'][mun_to_GGD['GGD'].isin(['BZO', 'HVB', 'WB'])]
            br_mun_list = br_mun_list.tolist()
            mun_df = mun_df[mun_df['Municipality'].isin(br_mun_list)]
            data_cols = mun_df.columns.tolist()
            data_cols.remove('Municipality')

            for mun in br_mun_list:
                if str(mun) not in mun_df['Municipality'].unique().tolist():
                    temp_dict = {'Municipality': mun}
                    for col in data_cols:
                        temp_dict[col] = 0
                    temp_series = pd.Series(temp_dict)

                    mun_df = mun_df.append(temp_series, ignore_index=True)

        if not mun_index:
            mun_df = mun_df.transpose()
            mun_df.columns = mun_df.iloc[0, :].tolist()
            mun_df = mun_df.iloc[2:, :]
            mun_df['Date'] = mun_df.index

            # Need to modify the date to plot the municipal time series
            mun_df['Date'] = mun_df['Date'].astype(str)
            mun_df['Date'] = pd.to_datetime(mun_df['Date'], format='%Y-%m')

            mun_df = mun_df.reset_index(drop=True)

        return mun_df

    def monthly_GGD(self, GGD_index=True, gtrends=True, Brabant=True):
        """
        A function that creates the pivot pd.DataFrame:
        Month    HVB  WB  BZO  Trends
        2004-2    1    3   0     8
        2004-3    2    0   1     2
        Parameters
        -------------------------
        df: Pandas DataFrame
            It must have at least two columns named GGD and Date.
            It is the general appended dateframe that has the following format:
            index Municipality Date       Sex     GGD
            0      Oss         2005-8     Man     HVB
            1      Eindhoven   2016-2    Vraouw   BZO
        GGD_index: Bool
            If false the index of the dataframe is the Date it looks like:
            index Date     HVB  BZO  WB
            0     2008-04   1    2    1
            1     2008-05   0    5    6
        gtrends: Bool
            If True the user is asked to provide a the directory of the csv file that
            has the goole trends data. Then the returned DataFrame is looks like:
            Month    HVB  WB  BZO  Trends
            2004-2    1    3   0     38
            2004-3    2    0   1     42
        """

        try:
            assert 'Municipality' in self.df.columns
            assert 'Date' in self.df.columns

            data = self.df.copy()

            if Brabant:
                data = data[data['GGD'].isin(['BZO', 'HVB', 'WB'])].reset_index(drop=True)

            if GGD_index:
                ggd_df = data.pivot_table(index='Date', columns='GGD',
                                          values='Count', aggfunc=np.sum,
                                          fill_value=0)

            else:
                ggd_df = data.pivot_table(index='GGD', columns='Date',
                                          values='Count', aggfunc=np.sum,
                                          fill_value=0)

            ggd_df = pd.DataFrame(ggd_df.to_records())

            if gtrends:
                gtrends_dir = str(input('Please Give me the Directory of the Google Trends data: '))
                os.chdir(gtrends_dir)
                gtrends_data = pd.read_csv('trends.csv')

                # In case the user did not delete the empty gtrends rows.
                if len(gtrends_data.columns) == 1:
                    gtrends_data = pd.read_csv('trends.csv', skiprows=[0, 1])

                gtrends_data.columns = ['Date', 'Trends']
                gtrends_data['Date'] = pd.to_datetime(gtrends_data['Date'], format='%Y-%m').dt.to_period('M')

                ggd_df = pd.merge(ggd_df, gtrends_data, on='Date')

                # We convert the Date column to pandas periods so that we can plot the number of incidences
                ggd_df['Date'] = ggd_df['Date'].astype(str)
                ggd_df['Date'] = pd.to_datetime(ggd_df['Date'], format='%Y-%m')

            return ggd_df

        except AssertionError:
            print('There sould be a Date and a GGD column')

    def center_of_mass(self):
        """
        A function that returns the longitude and latitude series that have the format:
            2004-01  2004-02  2004-03
            5.37758  5.13163  5.37601
            2004-01  2004-02  2004-03
            51.5588  51.5767  51.5961
        """

        thisdir = os.getcwd()
        os.chdir(r'../Data/Map')
        coord = pd.read_csv('nl_postal_codes.csv', encoding="ISO-8859-1")
        os.chdir(thisdir)

        coord = coord[['County', 'Latitude', 'Longitude']]
        coord.columns = ['Municipality', 'Latitude', 'Longitude']

        # erase municipality word in front of every municipality
        coord['Municipality'] = coord['Municipality'].str[9:]
        coord['Municipality'] = coord['Municipality']\
            .replace('Nuenen', 'Nuenen, Gerwen en Nederwetten')

        municipalities = mun_to_GGD['Municipality'][mun_to_GGD['GGD'].isin(['BZO', 'HVB', 'WB'])]
        municipalities = municipalities.tolist()

        coord = coord[coord['Municipality'].isin(municipalities)].reset_index(drop=True)

        # Avoid coordinate redundancy
        coord = coord.groupby('Municipality').head(1).reset_index(drop=True)

        # import the data
        aggregated_data = cp.copy(self)
        aggregated_data.find_mun()
        aggregated_data = aggregated_data.monthly_municipality()

        # find the number of patients per month
        totalpatients = aggregated_data.copy().sum()

        # Make ymonth and xmonth data frames that correspond to the
        # product of coordinates with the patients
        xmonth = aggregated_data.copy()
        ymonth = aggregated_data.copy()

        # Merge the relevant coordinates for every dataframe
        xmonth = xmonth.merge(coord, on='Municipality')
        ymonth = ymonth.merge(coord, on='Municipality')

        # Make ymonth and xmonth dataframes that correspond to the
        # product of coordinates with the patients
        xmonth = aggregated_data.copy()
        ymonth = aggregated_data.copy()

        # Merge the relevant coordinates for every dataframe
        xmonth = xmonth.merge(coord, on='Municipality')
        ymonth = ymonth.merge(coord, on='Municipality')

        # Drop the irrelevant columns for every dataframe
        xmonth = xmonth.drop(['Latitude'], axis=1)
        ymonth = ymonth.drop(['Longitude'], axis=1)

        # Find the time columns to calculate the center of mass
        columns = [col for col in xmonth.columns.tolist() if col not in
                   ['Municipality', 'Longitude', '2003-12']]

        # Multiply the coordinates of each municipality with the number of patients
        # and divide with the total number of patients of that month
        for col in columns:
            xmonth[col] = (xmonth[col]*xmonth['Longitude'])/totalpatients[col]
            ymonth[col] = (ymonth[col]*ymonth['Latitude'])/totalpatients[col]

        # Sum the product of the center of mass for each municipality
        latitude = ymonth.sum()
        longitude = xmonth.sum()

        # Delete the reduntant first and last column
        latitude = latitude.iloc[1:-1]
        longitude = longitude.iloc[1:-1]

        # Some monthns might have no incidences and no center of mass, thus we put it in the middle
        latitude = latitude.fillna((coord['Latitude'].max() + coord['Latitude'].min())/2)
        longitude = longitude.fillna((coord['Longitude'].max() + coord['Longitude'].min())/2)

        return longitude, latitude


def make_df_shapefile(df):
        """
        A function that creates the shape dataframe:
        Municipality  2004-01 2004-2  2004-03 ... geometry
        Eindhoven        1       3       0        POLYGON()
        Breda            2       0       1        POLYGON()
        Parameters
        -------------------------
        df: Pandas DataFrame
            It must have the following format:
            Municipality  2004-01 2004-2  2004-03
            Eindhoven        1       3       0
            Breda            2       0       1
        """

        try:
            assert 'Municipality' in df.columns

            # Preprocess the shapefile to be used
            columns = shp_netherlands.columns
            columns = [x for x in columns if x not in ['GMNAAM', 'geometry']]

            for column in columns:
                del shp_netherlands[column]

            shp_netherlands.columns = ['Municipality', 'geometry']

            # This Municipality does not exist anymore
            if 'Maasdonk' in df['Municipality']:
                df['Municipality'] = df['Municipality'].replace('Maasdonk', 'Oss')
            if 'Nuenen' in df['Municipality']:
                df['Municipality'] = df['Municipality'].replace('Nuenen', 'Nuenen, Gerwen en Nederwetten')

            # Now we limit muns to the municipalitites that only exist in df
            exist_Mun = df['Municipality'].unique().tolist()
            shp_Mun = shp_netherlands[shp_netherlands['Municipality'].isin(exist_Mun)]

            # We reset the index because of deleted rows
            shp_Mun = shp_Mun.reset_index(drop=True)

            # We make the coordinates according to the International system
            shp_Mun = shp_Mun.to_crs({'init': 'epsg:4326'})

            # Merge the shp_netherlands with the mun_df
            mun_df = pd.merge(df, shp_Mun, on='Municipality')

            # The Pandas data frame should be converted to geo pandas Data frame
            mun_df = gpd.GeoDataFrame(mun_df)

            return mun_df

        except AssertionError:
            print('There is no Column Named Municipality')


def disease_studied(keep=False):
    """ Directs to the relevant data directory of the disease. It then concatenates
    all the data frames of that disease.
    keep: Bool
        If True we keep all the columns of the dataframe to be returned.
    """
    fndir = os.getcwd()

    directory = str(input('Please Copy and Paste the Data Directory: '))
    print('I will concatenate all the csv files for you.')

    def find_csv_filenames(path_to_dir, suffix=".csv"):

        filenames = os.listdir(path_to_dir)
        return [filename for filename in filenames if filename.endswith(suffix)]

    os.chdir(directory)
    csv_list = find_csv_filenames(directory)

    # Initialize empty Pandas DataFrame
    all_df = pd.DataFrame({'PostCode': [], 'Date': []})

    for csv in csv_list:
        df = pd.read_csv(csv)
        for column in ['PostCode', 'Date']:
            if column not in df.columns.tolist():
                df = pd.read_csv(csv, sep=';')
                for column in ['PostCode', 'Date']:
                    if column not in df.columns.tolist():
                        print('There is no column named ' + column + 'in ' + csv)

        all_df = all_df.append(df).reset_index(drop=True)

    if not keep:
        for column in all_df.columns.tolist():
            if column not in ['PostCode', 'Date']:
                del all_df[column]

    os.chdir(fndir)

    print('The data is merged correctly!')
    return all_df


class PrepareTimeDf(object):
    """ The class that prepares the data to be used by the predictors.
    """
    def __init__(self, df):
        """ Initialize the object.
        """
        assert type(df) == pd.core.frame.DataFrame,\
            'What you have provided is not a Pandas DataFrame'

        self.df = df

    def create_lag(self, brabant=True, nlags=3):
        """ Preprocesses the dataset in order to create time lags for 'HVB', 'WB', 'BZO' and 'Trends'.
        df: Pandas DataFrame
        brabat: Bool
            If True, then only the columns named 'BZO', 'WB', 'HVB' will be kept
        lag: int
            This is the time lag to be created for each column
        """
        assert type(nlags) == int

        if 'Date' in self.df.columns:
            del self.df['Date']

        if brabant:
            df = self.df[['HVB', 'BZO', 'WB', 'Trends']]
        else:
            df = self.df

        for column in df.columns.tolist():
            for lag in range(1, nlags+1):
                df[column+'-'+str(lag)] = df[column].shift(lag)
            df[column+'+1'] = df[column].shift(-1)

        # We don't need Trends +1, no reason to forecast it. Remove also the last lag of trends
        del df['Trends+1']
        del df['Trends-'+str(nlags)]
        del df['Trends-'+str(nlags-1)]

        # Due to the shift that creates nans, we delete the first nlags rows and the last row.
        df = df[nlags:].reset_index(drop=True)
        df = df[:-1]

        self.df = df

    def split_data(self, train=0.33, val=0.1):
        """Splits the data to create a train, a validation, and a test set.
        train: float
            Percentage of the train set.
        val: float
            Percentage of the validation set.
        """
        assert train < 1 and train > 0,\
            "Train set must be between 0 and 1"
        assert val < 1 and val > 0,\
            "Validation set must be between 0 and 1"

        # Create a list of the target columns
        target_cols = [col for col in self.df.columns if '+1' in col]

        # Isolate the target columns
        y = self.df[target_cols]

        # Isolate the feature columns
        features = [col for col in self.df.columns if col not in target_cols]
        X = self.df[features]

        # Split the data
        X_T, X_test, y_T, y_test = train_test_split(X, y, test_size=train, shuffle=False)
        # Make the validation dataset
        X_train, X_val, y_train, y_val = train_test_split(X_T, y_T, test_size=val, shuffle=True)

        return X_train, X_test, X_val, y_train, y_test, y_val
