import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report
import pickle

 
def get_clean_data():
    
    data = pd.read_csv('data.csv')  
    data  = data.drop(['Unnamed: 32' , 'id'] ,axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M':1 ,'B':0})
    
    return data

def create_model(data): 
    
    X = data.drop(['diagnosis'],axis=1) # predictos 
    y = data['diagnosis'] # target variable
    
    ## scaling the data ,as some of the variabless are much larger/smaller than the others
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    ## spilt the data
    X_train , X_test , y_train , y_test = train_test_split(
        X,y,test_size=0.2,random_state=42
    )
    
    ## training the data
    
    model = LogisticRegression()
    model.fit(X_train,y_train)
    
    ## testing the model
    y_pred = model.predict(X_test)
    print("Accuracy: " , accuracy_score(y_test,y_pred))
    print("Classifcation report : \n" , classification_report(y_test,y_pred))
    
    return model , scaler 


    
# we have to export the model and the scalar cause if we just put  it in our app that means:
# the model will have to be trained again and agian , takes resources and time
def main():   
    
    data = get_clean_data()
     
    ##creating the model 
    model , scaler  = create_model(data)
    
    with open('cancer-prject/model.pkl' , 'wb') as f :
        pickle.dump(model,f)
    with open('cancer-prject/scaler.pkl' , 'wb') as f :    
        pickle.dump(scaler,f)
        
    
    
    
    
     
if __name__ == '__main__' :
    main()   
    
