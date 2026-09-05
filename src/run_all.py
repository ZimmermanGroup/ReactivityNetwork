from second_selection import *
import os

for i in [0, 1, 3, 7, 8]: 
    print("Running for target core:", i)
    # For Figure 6
    # os.system(f"python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n1 6 --n2_strategy cv --lp_epsilon 0.001")
    # os.system(f"python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n1 6 --n2_strategy cv --lp_epsilon 0")
    # os.system(f"python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n1 6 --n2_strategy cv --lp_epsilon 1")
    # os.system(f"python src/second_selection.py --target_core {i} --model baseline --n1_strategy rmap_cv --n1 6")
    # os.system(f"python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy rmap_cv --n1 6")
    
    # # For Figure 6 but without random removal of 14 BBs
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n1 6 --n_test 0 --n_bootstrap 1 --n2_strategy cv")
    # os.system(f" python src/second_selection.py --target_core {i} --model baseline --n1_strategy rmap_cv --n1 6 --n_test 0 --n_bootstrap 1")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy rmap_cv --n1 6 --n_test 0 --n_bootstrap 1")
    
    # For Figure S21
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy uncertainty")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy desc_cluster")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_target --n1_strategy uncertainty")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_target --n1_strategy desc_cluster")
    
    # # For Figure S22
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy desc_cluster")
    
    # # For Figure S23
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n2_strategy predefined")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy random --n2_strategy predefined")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy modularity --n2_strategy predefined")
    
    # # For Figure S24 -- need to configure so that GCN models can run.
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n2_strategy predefined")
    # # os.system(f" python src/second_selection.py --target_core {i} --model gcn --feature desc --n1_strategy rmap_cv")
    # # os.system(f" python src/second_selection.py --target_core {i} --model gcn --feature ohe --n1_strategy rmap_cv")
    
    # # For Figure S25: comparing sensitivity of CV vs. rule-based
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n2_strategy predefined --n1_strategy rmap --n1 6") # Totally fixed
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n2_strategy cv --n1_strategy rmap --n1 6") # Pilot selection fixed
    
    # # For Figure S 26
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n2_strategy cv --n1_strategy rmap_cv")
    # os.system(f" python src/second_selection.py --target_core {i} --model svm --n2_strategy cv --n1_strategy rmap_cv")
    # os.system(f" python src/second_selection.py --target_core {i} --model knn --n2_strategy cv --n1_strategy rmap_cv")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap  --n2_strategy cv --n1_strategy rmap_cv --tanimoto")
    
    # # For Figures S31–S32 (need to run downstream python notebook) comparing selection of 6 pilot BBs and determined by CV
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n1 -1 --n2_strategy cv")
        
    # For Figure S42 epsilon sensitivity analysis (need to run a notebook separately)
    # os.system(f"python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap --n1 6 --n2_strategy predefined")
    # os.system(f"python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap --n1 6 --n2_strategy predefined --lp_epsilon 0")

    # For Figure S40 yield aggregation sensitivity analysis 
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n2_strategy cv --aggregation median --include_zero_from_median")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n2_strategy cv --aggregation average --include_zero_from_median")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy rmap_cv --n2_strategy cv --aggregation maximum --include_zero_from_median")
    # os.system(f" python src/second_selection.py --target_core {i} --model baseline --n1_strategy rmap_cv --aggregation median --include_zero_from_median")
    # os.system(f" python src/second_selection.py --target_core {i} --model baseline --n1_strategy rmap_cv --aggregation average --include_zero_from_median")
    # os.system(f" python src/second_selection.py --target_core {i} --model baseline --n1_strategy rmap_cv --aggregation maximum --include_zero_from_median")