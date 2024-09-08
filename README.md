# **Student Performance Prediction - MLOps Project**

## **Project Overview**

This project aims to predict student performance based on various inputs using machine learning. The project is designed with an MLOps approach, incorporating data analysis, feature engineering, model development, and deployment through a CI/CD pipeline. The project is deployed using Docker for containerization and AWS ECR for CI/CD.

## **Key Features**
- **Exploratory Data Analysis (EDA)**: Detailed analysis of the dataset to understand patterns and relationships in the data.
- **Feature Engineering**: Creating meaningful features to enhance model performance.
- **Machine Learning Models**: Building models to predict student performance with high accuracy.
- **MLOps Implementation**: Using a structured MLOps pipeline for smooth workflow integration and model management.
- **Deployment**: The model is containerized using Docker and deployed using AWS ECR, with a CI/CD pipeline for automated updates.


## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - Pandas, NumPy (Data manipulation)
  - Scikit-learn (Modeling)
  - Matplotlib, Seaborn (Visualization)
  - Docker (Containerization)
  - AWS ECR (Deployment and CI/CD)
- **Tools**:
  - Jupyter Notebooks (EDA, model development)
  - Docker (Containerization)
  - AWS ECR (Deployment pipeline)
  - GitHub Actions (CI/CD pipeline setup)

## **Modeling Process**
1. **Data Preprocessing**: 
   - Handling missing values, categorical encoding, scaling.
2. **Exploratory Data Analysis (EDA)**: 
   - Visualizations and insights into relationships among variables.
3. **Feature Engineering**: 
   - Creating new features to improve model performance.
4. **Model Selection and Training**: 
   - Using multiple algorithms like Random Forest, Gradient Boosting, etc.
5. **Evaluation**:
   - Evaluating model performance using metrics like accuracy, F1-score.

## **Deployment Pipeline**
- **Containerization**: The project is containerized using Docker to ensure consistency across environments.
- **AWS ECR for CI/CD**: A CI/CD pipeline is built using GitHub Actions and AWS ECR to automate model deployment. Every code change triggers the pipeline to update the model in the deployed environment.

## **How to Run the Project Locally**
1. Clone the repository:
   ```bash
   git clone https://github.com/roybishal362/student-performance-mlops.git





