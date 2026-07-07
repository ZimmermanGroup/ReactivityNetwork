# Navigating parallel library synthesis with a Reactivity Network

Source code accompanying the paper to reproduce results. 
Raw data of Suzuki reactions will be made available once the paper is published in a journal.

Create the conda environment used for this work by running `conda env create -f rmap_env.yml`.  
Additionally, `pytorch` and `torch_geometric` were installed using pip. Versions used were 2.3.1 and 2.5.3, respectively.    
If vscode does not recognize the newly-made environment, run `python -m ipykernel install --user --name rmap --display-name "RMAP"`  
and then restart vscode.

### Contents
1. **dataset.py**

    Code for processing the raw data into a dataset object.  
    Rather than generating a separate processed file, we use this central **SuzukiDataset** object.  
    The object allows easy access to various arrays, including feature and yield matrices, necessary for ML through attributes.  
    Analysis of the dataset used is provided in the `dataset_analysis.ipynb` below.  

    Example usage:  
    ```python
    from dataset import *
    suzuki_data = SuzukiDataset(for_conventional_models=False, keepPhBr=False)
    y_reactivity = suzuki_data.reactivity_array
    print(np.unique(y_reactivity))
    >>> [0 1 2]
    X_fp = suzuki_data.X_fp
    print(X_fp.shape)
    >>> (13, 69, 2048)
    ```

2. **dataset_analysis.ipynb**

    Notebook that conducts exploratory data analysis as described in Figure 2 and Section 4 of the SI.  
    *Generates Table S1 and Figures S5–S6 (and Figure 2D)*

3. **feature_comparison.ipynb**

    Notebook that compares predictivity of various features, including adversarial controls, as described in pages S18–S19 of the SI.  
    A toy problem was used to compare the features:  
    * use one of the highly reactive cores as a target
    * randomly select 10 BBs and obtain their results with the target core
    * combine with all other BB X core combination to form the input data
    * The corresponding yield classes were binarized with those >40% being the positive class.
    * Train random forest classifiers (RFCs) using grid-search CV with a random 5-fold split.
    * Hyperparameters searched over are : `n_estimators=[10, 25, 50, 100]`; `max_depth=[2,3,5,None]`  
    
    *Generates Figure S17*

4. **rmap.py**

    Implements the Reactivity Map, building on the dataset object above.  
    Utility functions, such as preparing excel files ready for visualization with Gephi, are available.  
    Selection of first batch of 'representatives' is implemented here as well, as in the example below.  

    Example usage:  
    ```python
    from dataset import *
    from rmap import *
    suzuki_data = SuzukiDataset(for_conventional_models=False, keepPhBr=False)
    suzuki_rmap = ReactivityMap(source_core_inds=None, target_core_ind=8, test_bb_inds=[], dataset=suzuki_data)
    rmap.get_similarity_matrix()
    sim_mat = rmap.similarity_matrix
    print(sim_mat.shape)
    >>> (69, 69)
    print(np.mean(sim_mat), np.std(sim_mat))
    >>> 0.6887908702653504 0.1337334501804974
    selected_initial_BBs = rmap.select_first_batch(n_select=6)
    print(selected_initial_BBs)
    >>> [52, 11, 2, 62, 67, 58]
    ```

5. **first_selection.py**

    Code to compare the quality of the first batch of selected BBs using various strategies.
    To reproduce results in the paper:  
    Run `python src/first_selection.py`  
    As a result, a prompt `Where in the manuscript will the figures go?` will appear.  
    *To generate Figure 5A–5C*: Type `main` and press enter.  
    *To generate Figures S19–S20 and 5E*: Type `si` and press enter.  

6. **second_selection.py**

    Code to conduct the second batch of selections under different strategies.
    Joblib files that save the result of each run is generated.  
    Configuration of models are shown below:  
    | Model | Hyperparameters | Cross-validation |
    |-------|-----------------|------------------|
    |RFC|`n_estimators`$\in${10,20,50} <br> `max_depth`$\in${2,3,5,None} | 5-fold CV, Accuracy|
    |GCN|`hidden_size1`$\in${10,20,30,40} <br> `hidden_size2`$\in${0,5,10} <br> `learning_rate`$\in${0.0001,0.001,0.01} <br> `weight_decay`=0 | optuna, Binary Cross Entropy|
    |LP|`reactivity similarity threshold`=0.82 | n/a |
    |LP (Tanimoto)|`Tanimoto similarity threshold`=0.33 <br> `nbits`=2048, `rad`=3 <br> count Morgan Fingerprint | n/a| 
    |KNN|`n_neighbors`$\in${1,2,3}|leave-one-out CV, accuracy|
    |SVM|`reactivity similarity threshold`=0.82 | n/a |  

    Of note is that when RFCs are trained on the combined source and target data, we emphasize the target data with a weight 3 times that of the source.  
    Although all commands necessary to reproduce this paper is included in `run_all.py`, below is an example command that runs  
    label propagation on the Reactivity Map after initial selection through Reactivity Map clustering when the target core is 1:  
    ```python
    python src/second_selection.py --target_core 0 --model rmap
    ```
    
7. **gcn.py**

    Code that defines a simple graph convolutional network with one or two hidden layers.

8. **run_all.py**

    Runs all models and configurations used in this work.  
    Results are saved as joblib files, which are then used in `analyzer.py` below to be plotted.  
    Takes ~30 min to run without the GCN portion.

8. **analyzer.py**

    Must be run AFTER running `run_all.py`. 
    Plotting code for analyzing the quality of the second selections, *generating Figures 6B–6C and Figures S21–S28 and S30.*  
    Code block that leads to each figure is clearly marked towards the bottom of the script.  
    Simply un-comment the portion you are interested in and run:  
    ```python
    python src/analyzer.py
    ```

9. **second_selection.ipynb**

    Specific analyses of the quality of the prediction of remaining BBs.  
    *Generates Figure S31 and extracts the selected examples in Figure 6D*

10. **explore_initialBB_thresholds.ipynb**

    Evaluates the sensitivity of the second-step predictions to the number of pilot BBs and reactivity similarity threshold.  
    *Generates Figures S18 and S29* 
