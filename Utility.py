import numpy as np
import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

class Utility():
   
    def standardize_column_names(self, df):
        """
        Standardize column names by replacing space by "_" and removing capital letters
        Returns the dataframe
        """
        df.columns = [c.replace(" ","_").lower() for c in df.columns]
        return df
        
# =============================================================================
#     Operation on numerical columns
# =============================================================================
   
    def to_percentile(self, df):
        """
        df: dataframe
        
        Transform numerical df into percentile in O(NlogN)
        Return the percentiles
        """
        return np.argsort(np.argsort(df)) * 100. / (len(df) - 1)
    
    def numerical_to_categorical(self, df):
        """
        Convert continuous numerical column to categorical column ("Very low", "low","average",...)
        based on percentile. Returns converted column
        
        0-10%: very low
        10-25%: low
        25-75%: average
        75-90%: high
        90-100%: very high
        """
        percentiles = self.to_percentile(df)/100
        def convert(x):
            if x<0.1:
                return "Very low"
            elif x<0.25:
                return "Low"
            elif x<0.75:
                return "Average"
            elif x<0.9:
                return "High"
            else:
                return "Very High"
        convert = np.vectorize(convert)
        return convert(percentiles)
    
    def remove_outlier_percentile(self, df, cols = None, threshold = 95):
        """
        Removes outliers from both ends, based on percentile. 
        Returns the modified dataframe
        
        cols: LIST of columns names
        threshold: integer between 0 and 100, percentile to keep
        """
        diff = (100 - threshold) / 2
        if cols == None:
            cols = df.columns
        for col in cols:
            if np.issubdtype(df[col].dtype, np.number):
                minval, maxval = np.percentile(df[col], [diff, 100 - diff])
                df = df[(df[col] > minval) & (df[col] < maxval)]
        return df
    
    def remove_outlier_zscore(self, df, cols = None, threshold = 3):
        """
        Removes outliers from both ends, based on z-score. 
        Returns the modified dataframe
        
        cols: LIST of columns names
        threshold: integer between 1 and 3, those with z-score higher or equal than this will be removed 
        """
        from scipy import stats
        if cols == None:
            cols = df.columns
        for col in cols:
            if np.issubdtype(df[col].dtype, np.number):
                df['z_score'] = np.abs(stats.zscore(df[col]))
                df = df[(df['z_score'] < threshold)]
        return df

    def normalize(self, df, cols = None):
        """
        normalize given columns. Return the dataframe
        """
        if cols == None:
            cols = df.columns
        from sklearn.preprocessing import StandardScaler
        for col in cols:
            if np.issubdtype(df[col].dtype, np.number):
                scaler = StandardScaler()
                df[col] = scaler.fit_transform(df[col][:, np.newaxis])
        return df
    
# =============================================================================
# Operation on string columns
# =============================================================================
    
    def unpack_multitext(self, df):
        """
        unpack a pd.Series of tuples/list (e.g ("'Wifi','Microwave'") into different pd.Series and one-hot encode
        Return the columns
        """
        l = list(df.amenities)
        l = [[word.strip('[" ]') for word in row[1:-1].split(',')] for row in l] # Strip symbols
        cols=set(word for row in l  for word in row) #forming a set of distinct text
        cols.remove('')
        # One-hot encode
        new_df=pd.dfFrame(columns=cols)
        for col in cols:
            new_df[col] = df.amenities.apply(lambda x: int(col in x))
        return new_df


# =============================================================================
# Operation on categorical columns
# =============================================================================
    def keep_top_x_categories(self, df, cols = [], keep = 5, drop = False):
        """
        For given columns, keep only top X categories and group other categories under "Other" label
        Returns the dataframe with modified categories
        
        keep: Up to X-th biggest category is kept
        drop: if True, remove Others from dataframe
        """
        for col in cols:
            need = df[col].value_counts().index[:keep]
            df[col] = np.where(df[col].isin(need), df[col], 'Other')
            if drop == True:
                df = df[df[col] != 'Other']
        return df
            
    def label_encode(self, series):
        """
        Takes a categorical pd.Series ("cat","dog","dog","rat","cat","rat")
        and returns the label encoded (0,1,1,2,0,2,...) column
        
        df must not have NaN
        """
        le = LabelEncoder()
        
        if type(series) == pd.core.series.Series:
            if series.isna().any():
                series = series.fillna("NaN")
                image = le.fit_transform(series)
                image_nan = le.transform(["NaN"])[0]
                image = np.where(image == image_nan, np.nan, image)
                return image
        else:
            return le.fit_transform(series) # array

    def one_hot_encode(self,df):
        """
        Takes a categorical pd.Series and returns several one-hot encoded columns
        """
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = self.label_encode(df)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        return onehot_encoder.fit_transform(integer_encoded)
    
    def encode_categories(self, df, cols = None, max_nunique = 5):
        """
        For all categorical columns in df[cols], one-hot encode the column and add it to dataframe
        Returns the transformed dataframe
        
        cols: list of column names. if None, takes all columns of df
        max_unique: max number of unique elements in a column to encode it. If too high, gets messy
        """
        if cols == None:
            cols = df.columns
        for col in cols:
            if (df[col].nunique() <= max_nunique) and (df[col].nunique() > 2): #  categorical variable !
                dummies = pd.get_dummies(df[col], prefix = col+'_is')
                df = pd.concat([df, dummies], axis = 1)
        df.to_csv("data_one_hot_encoded.csv")
        return df
    
     
# =============================================================================
# NAN 
# =============================================================================
    
    def fillna(self, data, cols = None, value = None, method = None):
        """
        Fill NaN values. Returns dataframe
        
        value: if specified, fill columns with this value
        method: if value not specified, fill according to method ("bfill","ffill","pad")
        if none of value, method is specified, by default fill numerical with mean and categorical with None
        """
        df = copy.deepcopy(data)
        if cols == None:
            cols = df.columns
        if value != None:
            df[cols] = df[cols].fillna(value)
        elif method != None:
            df[cols] = df[cols].fillna(method = method)
        else: # Behaviour by default
            for col in cols:
                if np.issubdtype(df[col].dtype, np.number): # Fill by mean
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna("None")
        return df
    

# =============================================================================
# Groupby mecanics
# =============================================================================
    def groupby(self, df, groupby_col, value_col, func_name, custom_func = None):
        """
        Group by the groupby_col
        Apply the aggregate function on the value_col
        Assign the grouped value to each row
        Return the dataframe
        
        func : string or custom function
        
        Example of func: mean, min, max, unique, count, distinct_count
            - Aggregate on numerical value_col: np.mean, np.min, np.max
            - count: lambda x: x.count()
            - distinct count: lambda x: x.nunique()
        """
        func = {"mean": np.mean,
                "min": np.min,
                "max": np.max,
                "sum": np.sum,
                "prod": np.prod,
                "unique": lambda x: x.unique(),
                "count": lambda x: x.count(),
                "distinct_count": lambda x: x.nunique()}.get(func_name, custom_func)
        grouped = df.groupby(groupby_col)[value_col].agg(func).reset_index()
        grouped = grouped.rename(columns ={value_col: func_name + "_of_" + value_col +"_by_"+groupby_col})
        df = df.merge(grouped, how = 'left', on =  groupby_col)
        return df
    
# =============================================================================
# Miscellaneous
# =============================================================================
    def state_name_to_abbrev(self, df):
        """
        State name to 2 letter abbreviation
        """
        return [us_state_abbrev.get(x) for x in df]
    
    def abbrev_to_state_name(self, df):
        """
        2 letter abbreviation to state name
        """
        inv_map = {v: k for k, v in us_state_abbrev.items()}
        return [inv_map.get(x) for x in df]


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Palau': 'PW',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY',
}