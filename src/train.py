import sys
import warnings

import mlflow
import mlflow.tensorflow

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

from data_processing import DataProcessing 
from feature_engineering import FeatureEngineering
from classifier import QuestionAnswerClassifer


if __name__ == '__main__':
    
    Feature_Engineering = FeatureEngineering(new_df, 'Question', 'Label','Sentence')
    
    maxlen =  Feature_Engineering.determine_maxlen() + 10
    print(f'Maxlen:{maxlen}')
    
    #Split the dataset into train and test set
    new_df['Question_Sentence'] = new_df['Question'] +' '+ new_df['Sentence']
    features = new_df['Question_Sentence']
    target = new_df['Label']
    x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=0)
    
    tokenizer = Feature_Engineering.text_tokenize(x_train.values, num_words=5000)
    
    vocab_size = len(tokenizer.word_index) + 1
    
    xtrain= Feature_Engineering.tokenize_sequence(x_train, tokenizer, maxlen=maxlen)  
    xtest = Feature_Engineering.tokenize_sequence(x_test, tokenizer, maxlen=maxlen)  
    
    embedding_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    epochs = int(sys.argv[2]) if len(sys.argv) > 1 else 20
    batch_size = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    
    with mlflow.start_run():
        cls = QuestionAnswerClassifer(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen)
        
        model = cls.compile_model()
        
        model.summary()
        print(' ')
        
        history = model.fit(xtrain, y_train, 
                    epochs=epochs, 
                    verbose=False,
                    validation_data=(xtest, y_test),
                    batch_size=batch_size)
        
        print('Evaluation of model accuracy for trainingset')
        train_loss, train_accurracy = model.evaluate(xtrain, y_train)
        print(' ')
        print('Evaluation of model accuracy for testset')
        test_loss, test_accurracy = model.evaluate(xtest, y_test)
        print(' ')
        cls.plot_history(history)
        
        #log the metrics and parameters
        mlflow.log_param('embedding', embedding_dim)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('batch_size', batch_size)
        mlflow.metric("train_loss", train_loss)
        mlflow.metric("train_accuracy", train_accuracy)
        mlflow.metric("test_loss", test_loss)
        mlflow.metric("test_accuracy", test_accuracy)
        mlflow.metric("summary", model.summary())
    
        
        mlflow.tensorflow.log_model(model, "model")