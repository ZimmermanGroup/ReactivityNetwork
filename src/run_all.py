from second_selection import *
import os

for i in [0, 1, 3, 7, 8]:
    print("Running for target core:", i)
    # For Figure S20
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy uncertainty")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy desc_cluster")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_target --n1_strategy uncertainty")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_target --n1_strategy desc_cluster")
    # # For Figure S21
    os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined")
    # # For Figure S22
    os.system(f" python src/second_selection.py --target_core {i} --model rmap")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy random")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy modularity")
    # # For Figure S23 -- need to configure so that GCN models can run.
    # # os.system(f" python src/second_selection.py --target_core {i} --model gcn --feature desc")
    # # os.system(f" python src/second_selection.py --target_core {i} --model gcn --feature ohe")
    # # For Figure S24
    os.system(f" python src/second_selection.py --target_core {i} --model baseline")
    # # For Figure S XX
    # os.system(f" python src/second_selection.py --target_core {i} --model svm")
    # os.system(f" python src/second_selection.py --target_core {i} --model knn")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --tanimoto")
    # For Figure 6 but without random removal of 14 BBs
    os.system(f" python src/second_selection.py --target_core {i} --model rmap --n_test 0 --n_bootstrap 1")
    os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n_test 0 --n_bootstrap 1")
    os.system(f" python src/second_selection.py --target_core {i} --model baseline --n_test 0 --n_bootstrap 1")