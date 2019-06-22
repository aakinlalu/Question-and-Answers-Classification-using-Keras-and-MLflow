import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


class FeatureEngineering(DataProcessing):
    
    def __init__(self, df: pd.DataFrame, feature1:str, feature2:str, feature3:str):
        super().__init__(df, feature1, feature2, feature3)
        
    #feature = '_'.join([self.feature1, self.feature3])
        
    def join_features(self) -> pd.DataFrame:
        '''
        Function will return dataframe
        Return
        ------
        pd.DataFrame
        '''
        result = super().consolate_features()
        feature = '_'.join([self.feature1, self.feature3])
        return result[[feature, self.feature2]]
    
    def determine_maxlen(self) -> List[int]:
        '''
        The function will return list
        Return 
        ------
        List
        '''
        feature = '_'.join([self.feature1, self.feature3])
        s = self.join_features()[feature]
        #print(s)
        result = super().text_with_longest_length(s)
        return result[1]
               
    
    def dataset_split(self) -> pd.DataFrame:
        '''
        The function will return four dataframes
        Return
        ------
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, 
        '''
        feature = '_'.join([self.feature1, self.feature3])
        new_df = self.join_features()
        x_train,x_test, y_train,y_test = train_test_split(new_df[feature], new_df[self.feature2], test_size=0.25, random_state=0)
        return x_train,x_test, y_train,y_test
    
    def text_tokenize(self, text:pd.Series, num_words:int=None):
        '''
        The function will return function
        Return
        ------
        func
        '''
        tokenizer = Tokenizer(num_words=num_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',lower=True)
        tokenizer.fit_on_texts(text)
        return tokenizer
    
    def tokenize_sequence(self, text:pd.Series, func, maxlen:int, padding:str="post") -> List[list]:
        '''
        The function will return matrix
        Return
        ------
        Matrix
        '''
        new_text=func.texts_to_sequences(text)
        new_text= pad_sequences(new_text, padding=padding, maxlen=maxlen)
        return new_text