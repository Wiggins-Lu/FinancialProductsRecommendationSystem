'''
Diff with DIN:
    1、GRU with attentional update gate (AUGRU) 
    2、auxiliary loss function with click or not click product(negetive sampleming)
'''
import pandas as pd
import tensorflow as tf
from tensorflow.python.ops import math_ops
import numpy as np
import random

training_samples_file_path = tf.keras.utils.get_file("trainingSamples.csv",
                                                     "file:///Users/WigginsLu/Workspace/FinancialProductsRecSys/data/trainingSamples.csv")

test_samples_file_path = tf.keras.utils.get_file("testSamples.csv",
                                                 "file:///Users/WigginsLu/Workspace/FinancialProductsRecSys/data/testSamples.csv")



def get_dataset_with_negtive_product(path,batch_size,seed_num):
    tmp_df = pd.read_csv(path)
    tmp_df.fillna(0,inplace=True)
    random.seed(seed_num)
    negtive_product_df=tmp_df.loc[:,'userRatedproduct2':'userRatedproduct5'].applymap( lambda x: random.sample( set(range(0, 1001))-set([int(x)]), 1)[0]  )
    negtive_product_df.columns = ['negtive_userRatedproduct2','negtive_userRatedproduct3','negtive_userRatedproduct4','negtive_userRatedproduct5']
    tmp_df=pd.concat([tmp_df,negtive_product_df],axis=1)

    for i in tmp_df.select_dtypes('O').columns:
        tmp_df[i] = tmp_df[i].astype('str')
    
    if tf.__version__<'2.3.0':
        tmp_df = tmp_df.sample(  n= batch_size*( len(tmp_df)//batch_size   )   ,random_state=seed_num ) 
    
    
    dataset = tf.data.Dataset.from_tensor_slices( (  dict(tmp_df)) )
    dataset = dataset.batch(batch_size)
    return dataset

train_dataset = get_dataset_with_negtive_product(training_samples_file_path,12,seed_num=2020)
test_dataset = get_dataset_with_negtive_product(test_samples_file_path,12,seed_num=2021)

# Config
RECENT_productS = 5  
EMBEDDING_SIZE = 10

# define input for keras model
inputs = {
    'productAvgRating': tf.keras.layers.Input(name='productAvgRating', shape=(), dtype='float32'),
    'productRatingStddev': tf.keras.layers.Input(name='productRatingStddev', shape=(), dtype='float32'),
    'productRatingCount': tf.keras.layers.Input(name='productRatingCount', shape=(), dtype='int32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'),
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),

    'productId': tf.keras.layers.Input(name='productId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userRatedproduct1': tf.keras.layers.Input(name='userRatedproduct1', shape=(), dtype='int32'),
    'userRatedproduct2': tf.keras.layers.Input(name='userRatedproduct2', shape=(), dtype='int32'),
    'userRatedproduct3': tf.keras.layers.Input(name='userRatedproduct3', shape=(), dtype='int32'),
    'userRatedproduct4': tf.keras.layers.Input(name='userRatedproduct4', shape=(), dtype='int32'),
    'userRatedproduct5': tf.keras.layers.Input(name='userRatedproduct5', shape=(), dtype='int32'),

    'userType1': tf.keras.layers.Input(name='userType1', shape=(), dtype='string'),
    'userType2': tf.keras.layers.Input(name='userType2', shape=(), dtype='string'),
    'userType3': tf.keras.layers.Input(name='userType3', shape=(), dtype='string'),
    'userType4': tf.keras.layers.Input(name='userType4', shape=(), dtype='string'),
    'userType5': tf.keras.layers.Input(name='userType5', shape=(), dtype='string'),
    'productType1': tf.keras.layers.Input(name='productType1', shape=(), dtype='string'),
    'productType2': tf.keras.layers.Input(name='productType2', shape=(), dtype='string'),
    'productType3': tf.keras.layers.Input(name='productType3', shape=(), dtype='string'),
    
    'negtive_userRatedproduct2': tf.keras.layers.Input(name='negtive_userRatedproduct2', shape=(), dtype='int32'),
    'negtive_userRatedproduct3': tf.keras.layers.Input(name='negtive_userRatedproduct3', shape=(), dtype='int32'),
    'negtive_userRatedproduct4': tf.keras.layers.Input(name='negtive_userRatedproduct4', shape=(), dtype='int32'),
    'negtive_userRatedproduct5': tf.keras.layers.Input(name='negtive_userRatedproduct5', shape=(), dtype='int32'), 
    
    'label':tf.keras.layers.Input(name='label', shape=(), dtype='int32')
}


# user id embedding feature
user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=30001)
user_emb_col = tf.feature_column.embedding_column(user_col, EMBEDDING_SIZE)

# type features vocabulary
type_vocab = ['PropertyInsurance', 'LifeInsurance', 'MedicalInsurance', 'AccidentInsurance',
              'ShortTermBonds', 'LediumTermBonds', 'LongTermBonds',
              'StockFunds', 'BondFunds', 'MoneyMarketFunds', 'MixedFunds',
              'Gold', 'Silver', 'Platinum', 'Palladium',
              'LDW', 'LDZW', 'ZDLW']
# user type embedding feature
user_type_col = tf.feature_column.categorical_column_with_vocabulary_list(key="userType1",
                                                                           vocabulary_list=type_vocab)
user_type_emb_col = tf.feature_column.embedding_column(user_type_col, EMBEDDING_SIZE)
# item type embedding feature
item_type_col = tf.feature_column.categorical_column_with_vocabulary_list(key="productType1",
                                                                           vocabulary_list=type_vocab)
item_type_emb_col = tf.feature_column.embedding_column(item_type_col, EMBEDDING_SIZE)



candidate_product_col = [ tf.feature_column.numeric_column(key='productId', default_value=0),   ]

# user behaviors
recent_rate_col = [
    tf.feature_column.numeric_column(key='userRatedproduct1', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct2', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct3', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct4', default_value=0),
    tf.feature_column.numeric_column(key='userRatedproduct5', default_value=0),
]


negtive_product_col = [
    tf.feature_column.numeric_column(key='negtive_userRatedproduct2', default_value=0),
    tf.feature_column.numeric_column(key='negtive_userRatedproduct3', default_value=0),
    tf.feature_column.numeric_column(key='negtive_userRatedproduct4', default_value=0),
    tf.feature_column.numeric_column(key='negtive_userRatedproduct5', default_value=0),
]



# user profile
user_profile = [
    user_emb_col,
    user_type_emb_col,
    tf.feature_column.numeric_column('userRatingCount'),
    tf.feature_column.numeric_column('userAvgRating'),
    tf.feature_column.numeric_column('userRatingStddev'),
]

# context features
context_features = [
    item_type_emb_col,
    tf.feature_column.numeric_column('releaseYear'),
    tf.feature_column.numeric_column('productRatingCount'),
    tf.feature_column.numeric_column('productAvgRating'),
    tf.feature_column.numeric_column('productRatingStddev'),
]

label =[ tf.feature_column.numeric_column(key='label', default_value=0),   ]


candidate_layer = tf.keras.layers.DenseFeatures(candidate_product_col)(inputs)
user_behaviors_layer = tf.keras.layers.DenseFeatures(recent_rate_col)(inputs)
negtive_product_layer = tf.keras.layers.DenseFeatures(negtive_product_col)(inputs)
user_profile_layer = tf.keras.layers.DenseFeatures(user_profile)(inputs)
context_features_layer = tf.keras.layers.DenseFeatures(context_features)(inputs)
y_true = tf.keras.layers.DenseFeatures(label)(inputs)

# Activation Unit
product_emb_layer = tf.keras.layers.Embedding(input_dim=1001,output_dim=EMBEDDING_SIZE,mask_zero=True)# mask zero

user_behaviors_emb_layer = product_emb_layer(user_behaviors_layer) 
candidate_emb_layer = product_emb_layer(candidate_layer) 
negtive_product_emb_layer = product_emb_layer(negtive_product_layer) 

candidate_emb_layer = tf.squeeze(candidate_emb_layer,axis=1)

user_behaviors_hidden_state=tf.keras.layers.GRU(EMBEDDING_SIZE, return_sequences=True)(user_behaviors_emb_layer)

class attention(tf.keras.layers.Layer):
    def __init__(self, embedding_size=EMBEDDING_SIZE, time_length=5, ):
        super().__init__()
        self.time_length = time_length  
        self.embedding_size = embedding_size
        self.RepeatVector_time = tf.keras.layers.RepeatVector(self.time_length)
        self.RepeatVector_emb = tf.keras.layers.RepeatVector(self.embedding_size)        
        self.Multiply  =   tf.keras.layers.Multiply()
        self.Dense32   =   tf.keras.layers.Dense(32,activation='sigmoid')
        self.Dense1    =   tf.keras.layers.Dense(1,activation='sigmoid')        
        self.Flatten   =   tf.keras.layers.Flatten()    
        self.Permute   =   tf.keras.layers.Permute((2, 1))
        
    def build(self, input_shape):
        pass
    
    def call(self, inputs):
        candidate_inputs,gru_hidden_state=inputs
        repeated_candidate_layer = self.RepeatVector_time(candidate_inputs)
        activation_product_layer = self.Multiply([gru_hidden_state,repeated_candidate_layer]) 
        activation_unit = self.Dense32(activation_product_layer)
        activation_unit = self.Dense1(activation_unit)  
        Repeat_attention_s=tf.squeeze(activation_unit,axis=2)
        Repeat_attention_s=self.RepeatVector_emb(Repeat_attention_s)
        Repeat_attention_s=self.Permute(Repeat_attention_s)

        return Repeat_attention_s

attention_score=attention()( [candidate_emb_layer, user_behaviors_hidden_state])



class GRU_gate_parameter(tf.keras.layers.Layer):
    def __init__(self,embedding_size=EMBEDDING_SIZE):
        super().__init__()
        self.embedding_size = embedding_size        
        self.Multiply =   tf.keras.layers.Multiply()
        self.Dense_sigmoid = tf.keras.layers.Dense( self.embedding_size,activation='sigmoid'   )
        self.Dense_tanh =tf.keras.layers.Dense( self.embedding_size,activation='tanh'    )
        
    def build(self, input_shape):
        self.input_w =   tf.keras.layers.Dense(self.embedding_size,activation=None,use_bias=True)   
        self.hidden_w =   tf.keras.layers.Dense(self.embedding_size,activation=None,use_bias=False)   

    def call(self, inputs,Z_t_inputs=None ):
        gru_inputs,hidden_inputs = inputs
        if Z_t_inputs==None:
            return  self.Dense_sigmoid(  self.input_w(gru_inputs) + self.hidden_w(hidden_inputs) )
        else:           
            return self.Dense_tanh(  self.input_w(gru_inputs) + self.hidden_w(self.Multiply([hidden_inputs,Z_t_inputs]) ))

                                                                                                                                                                
class AUGRU(tf.keras.layers.Layer):
    def __init__(self,embedding_size=EMBEDDING_SIZE,  time_length=5):
        super().__init__()
        self.time_length = time_length
        self.embedding_size = embedding_size      
        self.Multiply =   tf.keras.layers.Multiply()
        self.Add=tf.keras.layers.Add()                                                                                
    
    def build(self, input_shape):
        self.R_t = GRU_gate_parameter()
        self.Z_t = GRU_gate_parameter()                                                                                     
        self.H_t_next = GRU_gate_parameter()     

    def call(self, inputs ):
        gru_hidden_state_inputs,attention_s=inputs
        initializer = tf.keras.initializers.GlorotUniform()
        AUGRU_hidden_state = tf.reshape(initializer(shape=(1,self.embedding_size )),shape=(-1,self.embedding_size ))
        for t in range(self.time_length):            
            r_t=   self.R_t(   [gru_hidden_state_inputs[:,t,:],  AUGRU_hidden_state]    )
            z_t=   self.Z_t(   [gru_hidden_state_inputs[:,t,:],  AUGRU_hidden_state]    )
            h_t_next=   self.H_t_next(   [gru_hidden_state_inputs[:,t,:],  AUGRU_hidden_state] , z_t  )
            Rt_attention =self.Multiply([attention_s[:,t,:] , r_t])
            
            AUGRU_hidden_state = self.Add( [self.Multiply([(1-Rt_attention),AUGRU_hidden_state  ] ), self.Multiply([Rt_attention ,h_t_next ] )])

        return AUGRU_hidden_state

augru_emb=AUGRU()(  [ user_behaviors_hidden_state   ,attention_score  ]  )

concat_layer = tf.keras.layers.concatenate([ augru_emb,  candidate_emb_layer,user_profile_layer,context_features_layer])

output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
y_pred = tf.keras.layers.Dense(1, activation='sigmoid')(output_layer)


class auxiliary_loss_layer(tf.keras.layers.Layer):
    def __init__(self,time_length=5 ):
        super().__init__()
        self.time_len = time_length-1        
        self.Dense_sigmoid_positive32 =   tf.keras.layers.Dense(32,activation='sigmoid')
        self.Dense_sigmoid_positive1 =   tf.keras.layers.Dense(1,activation='sigmoid')        
        self.Dense_sigmoid_negitive32 =   tf.keras.layers.Dense(32,activation='sigmoid')             
        self.Dense_sigmoid_negitive1 =   tf.keras.layers.Dense(1,activation='sigmoid')           
        self.Dot =   tf.keras.layers.Dot(axes=(1, 1))
        self.auc =tf.keras.metrics.AUC()
        
    def build(self, input_shape):
        pass
    
    def call(self, inputs,alpha=0.5):
        negtive_product_t1,postive_product_t0,product_hidden_state,y_true,y_pred=inputs
        #auxiliary_loss_values = [] 
        positive_concat_layer=tf.keras.layers.concatenate([  product_hidden_state[:,0:4,:],  postive_product_t0[:,1:5,:]  ])
        positive_concat_layer=self.Dense_sigmoid_positive32(   positive_concat_layer     )
        positive_loss = self.Dense_sigmoid_positive1(positive_concat_layer)
        
        negtive_concat_layer=tf.keras.layers.concatenate([  product_hidden_state[:,0:4,:],  negtive_product_t1[:,:,:]  ])
        negtive_concat_layer=self.Dense_sigmoid_negitive32(   negtive_concat_layer     )
        negtive_loss = self.Dense_sigmoid_negitive1(negtive_concat_layer)        
        auxiliary_loss_values = positive_loss + negtive_loss
        
        final_loss = tf.keras.losses.binary_crossentropy( y_true, y_pred )-alpha* tf.reduce_mean(  tf.reduce_sum(    auxiliary_loss_values,axis=1 ))
        self.add_loss(final_loss, inputs=True)
        self.auc.update_state(y_true, y_pred )
        self.add_metric(self.auc.result(), aggregation="mean", name="auc_value")        
        
        return  final_loss

auxiliary_loss_value=auxiliary_loss_layer()(  [ negtive_product_emb_layer,user_behaviors_emb_layer,user_behaviors_hidden_state,y_true,y_pred]  )

model = tf.keras.Model(inputs=inputs, outputs=[y_pred,auxiliary_loss_value])

model.compile(optimizer="adam")

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss,  test_roc_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {},  Test ROC AUC {},'.format(test_loss, test_roc_auc))



model.summary()

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[0][:12], list(test_dataset)[0]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))
