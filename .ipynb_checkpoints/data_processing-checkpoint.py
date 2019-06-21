from typing import *
from dataclasses import dataclass


@dataclass
class DataProcessing:
    '''
    This class will have different methods for data processing
    '''
    df: pd.DataFrame
    feature1: str
    feature2: str
    feature3: str
        
    def question_with_no_answer(self) -> List[str]:
        '''
        The function will take argument feature and use dataframe provided its class. 
        Parameters:
        ----------
       
        
        return
        ------
        List of strings
        
        '''
        qna_df = self.df.groupby([self.feature1, self.feature2])[self.feature3].count().reset_index()
        qna_df = qna_df.pivot(index=self.feature1, columns=self.feature2, values=self.feature3).reset_index()
        qna_null_df = qna_df[qna_df[1].isnull()]
        return  list(qna_null_df[self.feature1])
    
    def questions_with_answer(self) -> List[str]:
        '''
        The function will take no argument feature and use dataframe and feature  provided its class. 
        Parameters:
        ----------
       
        
        return
        ------
        List of strings
        '''
        question_list = self.question_with_no_answer()
        list_of_questions = set([item for item in self.df[self.feature1] if item not in question_list])
        return list_of_questions
    
    def question_df(self, question_list:List[str]) -> pd.DataFrame:
        '''
        The function will take argument list of strings and use dataframe and features provided its class. 
        Parameters:
        ----------
        
        return
        ------
        pd.Dataframe
        '''
        list_of_dfs = [self.df[self.df[self.feature1]==item] for item in question_list]
        question_df = pd.concat( list_of_dfs, axis=0)
        return  question_df
    
    def consolate_features(self) -> pd.DataFrame:
        '''
        function join two Series into one and return a new dataframe
        return
        -------
        pd.Dataframe
        '''
        feature = '_'.join([self.feature1, self.feature3])
        df[feature] = self.df[self.feature1] +' '+ self.df[self.feature3]
        return df
    

    def text_with_longest_length(self, s: pd.Series) -> dict:
        '''
        '''
        result = {}
        max_length = 0
        min_length= 5
        max_text = None
        min_text = None
        for item in s:
            length =len(str(item).split(' '))
            if length > max_length:
                max_length = length 
                max_text = item
        #result['max'] = [max_text, max_length]
        return [max_text, max_length]