import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA

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
        for col in self.dataset.columns:
            self.dataset[col] = pd.to_numeric(self.dataset[col], errors='coerce')  # convert to numeric, replacing non-numeric values with NaN
            self.dataset.dropna(inplace=True)  # remove rows with NaN values
            min_val = self.dataset[col].min()
            max_val = self.dataset[col].max()
            denominator = max_val - min_val
            if denominator == 0:
                denominator = 1e-10  # add a small value to avoid division by zero
            self.dataset[col] = (self.dataset[col] - min_val) / denominator

    def feature_engineering_module(self):
        self.dataset.replace([np.inf, -np.inf], np.nan, inplace=True)  # replace infinite values with NaN
        self.dataset.dropna(inplace=True)  # remove rows with NaN values
    
        if self.feature_engineering_method == 'none':
            pass
        elif self.feature_engineering_method == 'pca':
            X = self.dataset.drop(['bike_id'] + self.target_attributes, axis=1)  # select features
            pca = PCA(n_components=0.95)  # retain 95% of the variance
            X_pca = pca.fit_transform(X)
            self.dataset = pd.concat([self.dataset[['bike_id']], pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])]), self.dataset[self.target_attributes]], axis=1)
        elif self.feature_engineering_method == 'tandardization':
            scaler = StandardScaler()
            X = self.dataset.drop(['bike_id'] + self.target_attributes, axis=1)  # select features
            X_scaled = scaler.fit_transform(X)
            self.dataset = pd.concat([self.dataset[['bike_id']], pd.DataFrame(X_scaled, columns=X.columns), self.dataset[self.target_attributes]], axis=1)
        elif self.feature_engineering_method == 'normalization':
            scaler = MinMaxScaler()
            X = self.dataset.drop(['bike_id'] + self.target_attributes, axis=1)  # select features
            X_scaled = scaler.fit_transform(X)
            self.dataset = pd.concat([self.dataset[['bike_id']], pd.DataFrame(X_scaled, columns=X.columns), self.dataset[self.target_attributes]], axis=1)
        else:
            raise ValueError("Invalid feature engineering method. Please choose from 'none', 'pca', 'tandardization', or 'normalization'.")
        def machine_learning_module(self):
            if self.ml_model == 'random_forest':
                X = self.dataset.drop(self.target_attributes, axis=1)
                y = self.dataset[self.target_attributes]
                clf = RandomForestClassifier(n_estimators=100, random_state=42)
                clf.fit(X, y)
                self.feature_importances = clf.feature_importances_
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

        optimizer = AttributeOptimizer(dataset, target_attributes, objective, generate_marketing_plan)
        optimizer.data_preprocessing_module()
        optimizer.feature_engineering_module()
        optimizer.machine_learning_module()
        optimizer.correlation_analysis_module()
        optimizer.solution_generation_module()
        optimizer.solution_ranking_module()
        optimizer.marketing_plan_generation_module()

        st.write('## Solutions:')
        for i, solution in enumerate(optimizer.solutions):
            st.write(f'{i+1}. {solution}')

        if generate_marketing_plan:
            st.write('## Marketing Plan:')
            for i, plan in enumerate(optimizer.marketing_plan):
                st.write(f'{i+1}. {plan}')

        st.write('## Correlation Matrix:')
        fig = px.imshow(optimizer.correlation_matrix, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
