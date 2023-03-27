import pandas as pd
import pickle


class HospitalPricingPredictionModel:
    '''
        Class that predicts hospital pricing using the deployed model.
    '''
    def __init__(self):
        '''
            Constructor loads the deployed model along with scaler and encoder objects.
        '''
        with open('.bins/model_covered_charges','rb') as model_file, open(
            '.bins/scaler_covered_charges', 'rb') as scaler_file, open(
            '.bins/encoder_covered_charges', 'rb') as encoder_file:
            self.reg_1 = pickle.load(model_file)
            self.scaler_1 = pickle.load(scaler_file)
            self.encoder_1 = pickle.load(encoder_file)
        
        with open('.bins/model_total_payments','rb') as model_file, open(
            '.bins/scaler_total_payments', 'rb') as scaler_file, open(
            '.bins/encoder_total_payments', 'rb') as encoder_file:
            self.reg_2 = pickle.load(model_file)
            self.scaler_2 = pickle.load(scaler_file)
            self.encoder_2 = pickle.load(encoder_file)
        
        with open('.bins/model_medicare_payments','rb') as model_file, open(
            '.bins/scaler_medicare_payments', 'rb') as scaler_file, open(
            '.bins/encoder_medicare_payments', 'rb') as encoder_file:
            self.reg_3 = pickle.load(model_file)
            self.scaler_3 = pickle.load(scaler_file)
            self.encoder_3 = pickle.load(encoder_file)

    def load_and_clean_data(self, data_file):
        '''
            Method loads the data and perform all neccessary preprocessing.
        
            Args:
                data_file (str): File path of the inputs that needs to be predicted
        
            Returns:
                None
        '''
        df = pd.read_csv(data_file)
        
        # we have included this line of code if we need to call the 'preprocessed data'
        self.preprocessed_data = df.copy()
        
        # Droping the irrelevant columns
        df = df.drop(['Provider Id', 'Provider Street Address',
                      'Provider City', 'Provider State'], axis=1)
        
        # Ensuring that the zipcode is stored as unique string
        df['Provider Zip Code'] = df['Provider Zip Code'].astype(str).str.zfill(5)
        
        # Encoding and scaling the data for respective predictions
        # Such as covered charges, total payments and medicare payments
        x_test_1_loo = self.encoder_1.transform(df)
        x_test_2_loo = self.encoder_2.transform(df)
        x_test_3_loo = self.encoder_3.transform(df)
        self.data_1 = self.scaler_1.transform(x_test_1_loo)
        self.data_2 = self.scaler_2.transform(x_test_2_loo)
        self.data_3 = self.scaler_3.transform(x_test_3_loo)

    def predict_outputs(self):
        '''
            Method predicts the ouputs and writes the result to a csv file.
            
            Returns:
                None
        '''
        # Predict the covered charges and append it to a new column called 'Predicted Covered Charges'
        self.preprocessed_data['Predicted Covered Charges'] = self.reg_1.predict(self.data_1)
        
        # Predict the total payments required to be paid
        self.preprocessed_data['Predicted Total Payments'] = self.reg_2.predict(self.data_2)
        
        # Predict the Medicare costs
        self.preprocessed_data['Predicted Medicare Payments'] = self.reg_3.predict(self.data_3)
        
        # Write the result to a CSV file
        self.preprocessed_data.to_csv('Outputs/result.csv', index = False)
