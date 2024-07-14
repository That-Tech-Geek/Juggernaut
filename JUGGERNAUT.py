import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class AttributeOptimizer:
    def __init__(self, dataset, attribute1, attribute2, objective, generate_marketing_plan=False, 
                 preprocessing_method='standard_scaler', feature_engineering_method='pca', 
                 ml_model='gradient_boosting', correlation_analysis_threshold=0.5):
        self.dataset = dataset
        self.attribute1 = attribute1
        self.attribute2 = attribute2
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
        
        id_column = [col for col in self.dataset.columns if 'id' in col.lower()][0]
        
        X = self.dataset.drop([id_column, self.attribute1, self.attribute2], axis=1)  # select features
        X = X.select_dtypes(include=[np.number])  # select only numeric columns
        
        y = self.dataset[[self.attribute1, self.attribute2]]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.values)  # use X.values to ensure a numpy array
        
        self.dataset = pd.concat([self.dataset[[id_column]], pd.DataFrame(X_scaled, columns=X.columns), self.dataset[[self.attribute1, self.attribute2]]], axis=1)
        
        gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gb_model.fit(X_scaled, y)
        self.feature_importances = gb_model.feature_importances_
        
        return self.dataset, self.feature_importances

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

    dataset = st.file_uploader('Upload your dataset', type=['csv', 'xlsx'])
    if dataset is not None:
        dataset = pd.read_csv(dataset) if dataset.name.endswith('.csv') else pd.read_excel(dataset)

        attribute1 = st.selectbox('Select first attribute', dataset.columns)
        attribute2 = st.selectbox('Select second attribute', dataset.columns)
        objective = st.selectbox('Select objective', ['increase', 'decrease'])
        generate_marketing_plan = st.checkbox('Generate marketing plan')

        optimizer = AttributeOptimizer(dataset, attribute1, attribute2, objective, generate_marketing_plan)
        optimizer.data_preprocessing_module()
        optimizer.feature_engineering_module()
        optimizer.correlation_analysis_module()
        optimizer.solution_generation_module()
        ranked_solutions = optimizer.solution_ranking_module()
        optimizer.marketing_plan_generation_module()

        st.write('## Solutions:')
        for i, solution in enumerate(ranked_solutions):
            st.write(f'{i+1}. {solution[0]} (Score: {solution[1]:.2f})')

        if generate_marketing_plan:
            st.write('## Marketing Plan:')
            for i, plan in enumerate(optimizer.marketing_plan):
                st.write(f'{i+1}. {plan}')

        st.write('## Correlation Matrix:')
        fig = px.imshow(optimizer.correlation_matrix, color_continuous_scale='RdBu_r')
        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
