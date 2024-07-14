import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import optuna
import numpy as np

class AttributeOptimizer:
    def __init__(self, dataset, target_attributes, objective, generate_marketing_plan=False, 
                 preprocessing_method='standard_scaler', feature_engineering_method='pca', 
                 ml_model='random_forest', correlation_analysis_threshold=0.5):
        self.dataset = dataset
        self.target_attributes = target_attributes
        self.objective = objective
        self.generate_marketing_plan = generate_marketing_plan
        self.preprocessing_method = preprocessing_method
        self.feature_engineering_method = feature_engineering_method
        self.ml_model = ml_model
        self.correlation_analysis_threshold = correlation_analysis_threshold
        self.preprocessed_data = None
        self.feature_importances = None
        self.correlation_matrix = None
        self.solutions = None
        self.marketing_plan = None

    def data_preprocessing_module(self):
        if self.preprocessing_method == 'tandard_scaler':
            scaler = StandardScaler()
            self.dataset[self.dataset.columns] = scaler.fit_transform(self.dataset[self.dataset.columns])
        elif self.preprocessing_method == 'in_max_scaler':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            self.dataset[self.dataset.columns] = scaler.fit_transform(self.dataset[self.dataset.columns])
        else:
            raise ValueError('Invalid preprocessing method')

        self.preprocessed_data = self.dataset

    def feature_engineering_module(self):
        if self.feature_engineering_method == 'pca':
            pca = PCA(n_components=0.95)
            self.dataset = pca.fit_transform(self.dataset)
        elif self.feature_engineering_method == 't_sne':
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, random_state=42)
            self.dataset = tsne.fit_transform(self.dataset)
        else:
            raise ValueError('Invalid feature engineering method')

    def machine_learning_module(self):
        if self.ml_model == 'random_forest':
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            X = self.preprocessed_data.drop(self.target_attributes, axis=1)
            y = self.preprocessed_data[self.target_attributes]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            print(f'MSE: {mse:.2f}, R2: {r2:.2f}')
            self.feature_importances = rf.feature_importances_
        else:
            raise ValueError('Invalid machine learning model')

    def correlation_analysis_module(self):
        self.correlation_matrix = self.dataset.corr()
        strong_correlations = self.correlation_matrix[abs(self.correlation_matrix) > self.correlation_analysis_threshold]

    def solution_generation_module(self):
        opportunities = []
        for i, feature in enumerate(self.feature_importances):
            if feature > 0.1:
                opportunities.append((feature, self.dataset.columns[i]))

        solutions = []
        for opportunity in opportunities:
            feature, column = opportunity
            if self.objective == 'increase':
                solutions.append(f'Increase {column} by 10%')
            else:
                solutions.append(f'Decrease {column} by 10%')

        self.solutions = solutions

    def solution_ranking_module(self):
        solution_scores = []
        for solution in self.solutions:
            impact = 0.5
            feasibility = 0.8
            cost = 0.3
            score = impact * feasibility / cost
            solution_scores.append(score)

        ranked_solutions = sorted(zip(self.solutions, solution_scores), key=lambda x: x[1], reverse=True)

        return ranked_solutions

    def marketing_plan_generation_module(self):
        if not self.generate_marketing_plan:
            return

        marketing_plan = []
        for solution in self.solutions:
            if 'discount' in solution:
                marketing_plan.append('Create a targeted email campaign offering a 10% discount')
            elif 'increase' in solution:
                marketing_plan.append(f'Increase {solution.split()[1]} by 10%')
            elif 'decrease' in solution:
                marketing_plan.append(f'Decrease {solution.split()[1]} by 10%')

        self.marketing_plan = marketing_plan

def main():
    st.title('Attribute Optimizer')
    st.write('Welcome to the Attribute Optimizer!')

    dataset= st.file_uploader('Upload your dataset', type=['csv', 'xlsx'])
    if dataset is not None:
        dataset = pd.read_csv(dataset) if dataset.name.endswith('.csv') else pd.read_excel(dataset)

        target_attributes = st.multiselect('Select target attributes', dataset.columns)
        objective = st.selectbox('Select objective', ['increase', 'decrease'])
        generate_marketing_plan = st.checkbox('Generate marketing plan')

        preprocessing_method = st.selectbox('Select preprocessing method', ['standard_scaler', 'in_max_scaler'])
        feature_engineering_method = st.selectbox('Select feature engineering method', ['pca', 't_sne'])
        ml_model = st.selectbox('Select machine learning model', ['random_forest'])
        correlation_analysis_threshold = st.slider('Correlation analysis threshold', 0.0, 1.0, 0.5)

        optimizer = AttributeOptimizer(dataset, target_attributes, objective, generate_marketing_plan, 
                                        preprocessing_method, feature_engineering_method, ml_model, 
                                        correlation_analysis_threshold)

        optimizer.data_preprocessing_module()
        optimizer.feature_engineering_module()
        optimizer.machine_learning_module()
        optimizer.correlation_analysis_module()
        optimizer.solution_generation_module()
        optimizer.solution_ranking_module()
        optimizer.marketing_plan_generation_module()

        st.write('Solutions:')
        for solution, score in optimizer.solution_ranking_module():
            st.write(f'{solution}: {score:.2f}')

        if optimizer.generate_marketing_plan:
            st.write('Marketing Plan:')
            for plan in optimizer.marketing_plan:
                st.write(plan)

if __name__ == '__main__':
    main()
