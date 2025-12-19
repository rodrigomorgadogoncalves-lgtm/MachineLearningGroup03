# MachineLearningGroup03

Here are the 3 notebooks of our final delivery.

Steps: Clone the repository:

Install dependencies: This project requires specific libraries like rapidfuzz and scikit-learn.

Execution Order:

Run Group_03_notebook.ipynb first to see the analysis.

Run Group03_models.ipynb to run the models in which we make predictions.

Order analysis:

Group_03_notebook.ipynb (Part I):
Executes steps 1–5
Purpose: Exploratory Data Analysis (EDA), data cleaning and pre-processing.

Group03_models.ipynb (Part I):
Executes up to the Blending section.
Purpose: Training, hyperparameter tuning, and validation of individual models (RF, HGB, ET...).

Group_03_notebook.ipynb (Part II):
Executes Section 6.
Purpose: Open ended section.

Group03_models.ipynb (Part II):
Executes the Blending & Submission sections.
Purpose: Running the Global Optimization (Differential Evolution) to find optimal weights and generating the final submission.csv.


Data Source: The dataset contains information regarding car attributes (brand, engine, year, etc.) and their selling prices.

Results & Methodology: We utilized a blending strategy of five-based models. The final model weights used for the submission were. This approach allowed us to balance the variance and bias of the individual models to minimize the Mean Absolute Error (MAE).

Here is the schema of our project: 

![WhatsApp Image 2025-12-19 at 16 37 53](https://github.com/user-attachments/assets/1cd4cbbb-8f8b-45cb-afc1-e23b11d54d6d)





