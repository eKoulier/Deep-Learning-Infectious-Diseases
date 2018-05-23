import pandas as pd
import geopandas as gpd
import numpy as np
import os

import sys
sys.path.append("../Data/Map")

# First we read the Files to be use
cwd = os.getcwd()
os.chdir(r'../Data/Map')
post_to_mun = pd.read_csv('Post_to_Mun.txt', sep=';')
mun_to_GGD = pd.read_csv('Mun_to_GGD.csv', sep=';')
mun_to_GGD['Municipality'] = mun_to_GGD['Municipality'].replace('Nuenen', 'Nuenen, ' +
                                                                'Gerwen en Nederwetten')
shp_Nether = gpd.read_file('GEO.Gemeente_2015.shp')
os.chdir(cwd)


class MonthlyTransfrom(object):
    """ A class to create the aggregated dataframe. Two methods provided:
    The aggregation by GGD and the aggregation by muicipality.
    """

    def __init__(self, df):
        """ Initialize the dataframe object. The dataframe has the following
        format:
        Date  PostCode  Sex  Year of Birth etc
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
        time_df['Date'] = pd.period_range(start=df_data['Date'].min() - 2,
                                          end=df_data['Date'].max(), freq='M')

        # Probably there are missing months in the data and this is why we do the following
        df_time_extended = time_df.merge(df_data, on=['Date'], how='outer')
        df_time_extended['Count'] = df_time_extended['Count'].fillna(0)
        df_time_extended['PostCode'] = df_time_extended['PostCode'].fillna(5731)

        self.df = df_time_extended

    def find_mun(self):
        """ Creates the Municipality column to the DataFrame. The Dataframe
        should have a column named PostCode.
        """

        try:
            assert 'PostCode' in self.df.columns

            if 'Municipality' in self.df.columns:
                del self.df['Municipality']

            self.df = pd.merge(self.df, post_to_mun, on='PostCode')
            self.df['Municipality'] = self.df['Municipality'].replace('Nuenen', 'Nuenen, Gerwen en Nederwetten')

        except AssertionError:
            print('LEFTERIS\'S MESSEGE!: ' +
                  + 'There in no column named PostCode')

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
            print('LEFTERIS\'S MESSEGE!: ' +
                  + 'There in no column named PostCode')

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
        msg = 'LEFTERIS\'S MESSEGE!: There sould be a Date and a Municipality column'
        assert('Municipality' in self.df.columns, msg)
        assert('Date' in self.df.columns, msg)

        data = self.df.copy()

        # Select the relevant columns
        data = data[['Date', 'Municipality', 'Count']]

        # Create the pivot table
        if mun_index:
            Mun_df = data.pivot_table(index='Municipality',
                                      columns='Date', values='Count',
                                      aggfunc=np.sum, fill_value=0)

            # Delete the extra heading of the pivot table
            Mun_df = pd.DataFrame(Mun_df.to_records())

        else:
            Mun_df = data.pivot_table(index='Date',
                                      columns='Municipality',
                                      values='Count',
                                      aggfunc=np.sum, fill_value=0)

            # Delete the extra heading of the pivot table
            Mun_df = pd.DataFrame(Mun_df.to_records())

        if Brabant:
            br_mun_list = mun_to_GGD['Municipality'][mun_to_GGD['GGD'].isin(['BZO', 'HVB', 'WB'])]
            br_mun_list = br_mun_list.tolist()
            Mun_df = Mun_df[Mun_df['Municipality'].isin(br_mun_list)]
            data_cols = Mun_df.columns.tolist()
            data_cols.remove('Municipality')

            for mun in br_mun_list:
                if str(mun) not in Mun_df['Municipality'].unique().tolist():
                    temp_dict = {'Municipality': mun}
                    for col in data_cols:
                        temp_dict[col] = 0
                    temp_series = pd.Series(temp_dict)

                    Mun_df = Mun_df.append(temp_series, ignore_index=True)

        return Mun_df

    def monthly_GGD(self, GGD_index=True, gTrends=True, Brabant=True):
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
        gTrends: Bool
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
                GGD_df = data.pivot_table(index='Date', columns='GGD',
                                          values='Count', aggfunc=np.sum,
                                          fill_value=0)

            else:
                GGD_df = data.pivot_table(index='GGD', columns='Date',
                                          values='Count', aggfunc=np.sum,
                                          fill_value=0)

            GGD_df = pd.DataFrame(GGD_df.to_records())

            if gTrends:
                gTrendsDir = str(input('Please Give me the Directry of the Google Trends data: '))
                os.chdir(gTrendsDir)
                gTrendsData = pd.read_csv('trends.csv')

                # In case the user did not delete the empty gTrends rows.
                if len(gTrendsData.columns) == 1:
                    gTrendsData = pd.read_csv('trends.csv', skiprows=[0, 1])

                gTrendsData.columns = ['Date', 'Trends']
                gTrendsData['Date'] = pd.to_datetime(gTrendsData['Date'], format='%Y-%m').dt.to_period('M')

                GGD_df = pd.merge(GGD_df, gTrendsData, on='Date')

            return GGD_df

        except AssertionError:
            print('There sould be a Date and a GGD column')


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

            # Preprocessing the shapefile to be used
            columns = shp_Nether.columns
            columns = [x for x in columns if x not in ['GMNAAM', 'geometry']]

            for column in columns:
                del shp_Nether[column]

            shp_Nether.columns = ['Municipality', 'geometry']

            # This Municipality does not exist anymore
            if 'Maasdonk' in df['Municipality']:
                df['Municipality'] = df['Municipality'].replace('Maasdonk',
                                                                'Oss')
            if 'Nuenen' in df['Municipality']:
                df['Municipality'] = df['Municipality'].replace('Nuenen', 'Nuenen, Gerwen en ' +
                                                                + 'Nederwetten')

            # Now we limit muns to the municipalitites that only exist in df
            exist_Mun = df['Municipality'].unique().tolist()
            shp_Mun = shp_Nether[shp_Nether['Municipality'].isin(exist_Mun)]

            # We reset the index because of deleted rows
            shp_Mun = shp_Mun.reset_index(drop=True)

            # We make the coordinates according to the International system
            shp_Mun = shp_Mun.to_crs({'init': 'epsg:4326'})

            # Merge the shp_Nether with the Mun_df
            Mun_df = pd.merge(df, shp_Mun, on='Municipality')

            # The Pandas dataframe should be converted to geopandas Dataframe
            Mun_df = gpd.GeoDataFrame(Mun_df)

            return Mun_df

        except AssertionError:
            print('There is no Column Named Municipality')


def disease_studied():
    """ Directs to the relevant data directory of the disease. It then concatenates
    all the dataframes of that disease.
    """
    fndir = os.getcwd()

    directory = str(input('Please Copy and Paste the Data Directory'))
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
                    if column not in ['PostCode', 'Date']:
                        print('LEFTERIS MESSAGE: CAREFUL! there is no column named ' + column + 'in ' + csv)

        all_df = all_df.append(df).reset_index(drop=True)

    for column in all_df.columns.tolist():
        if column not in ['PostCode', 'Date']:
            del all_df[column]

    os.chdir(fndir)

    return all_df
