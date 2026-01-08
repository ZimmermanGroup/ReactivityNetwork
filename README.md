# Navigating parallel library synthesis with a Reactivity Map

Source code accompanying the paper to reproduce results. 
Raw data of Suzuki reactions will be made available once the paper is published in a journal.

### Contents
1. dataset.py

    Code for processing the raw data into a dataset object. Various arrays, including feature and yield matrices, can be easily accessed.

2. dataset_analysis.ipynb

    Notebook that conducts exploratory data analysis as described in Figure 2 and Section 4 of the SI.

3. feature_comparison.ipynb

    Notebook that compares predictivity of various features, including adversarial controls, as described in pages S18–S19 of the SI.

4. rmap.py

    Implements the Reactivity Map, building on the dataset object above.
    Utility functions, such as preparing excel files ready for visualization with Gephi, are available.
    Selection of first batch of 'representatives' is implemented here as well.


5. first_selection.py

    Code to compare the quality of the first batch of selected BBs using various strategies.
    Plots in Figure 5 was generated using this code. 

6. second_selection.py

    Code to conduct the second batch of selections under different conditions.
    Joblib files that save the result of each run is generated.

7. gcn.py

    Code that defines a simple graph convolutional network with one or two hidden layers.

8. second_selection.ipynb

    Specific analyses of the quality of the prediction of remaining BBs, such as Figure 6D Figure S25, were generated in this notebook. 

9. analyzer.py

    Plotting code for analyzing the quality of the second selections, generating Figure 6B and Figures S20–S24.
