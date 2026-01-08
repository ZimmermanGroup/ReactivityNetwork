from second_selection import *
import os

# for i in [10, 11]:
for i in [0, 1, 3, 7, 8]:
    print("Running for target core:", i)
    # print("    Running RMAP...")
    os.system(f" python src/second_selection.py --target_core {i} --model rmap")
    os.system(
        f" python src/second_selection.py --target_core {i} --model rmap --subgraph"
    )
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy random")
    # os.system(f" python src/second_selection.py --target_core {i} --model rmap --n1_strategy modularity")
    # print("    Running RFC...")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --augment")
    # print("    Running Baseline...")
    # os.system(f" python src/second_selection.py --target_core {i} --model baseline")
    # print()
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy uncertainty")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_combined --n1_strategy desc_cluster")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_target --n1_strategy uncertainty")
    # os.system(f" python src/second_selection.py --target_core {i} --model rfc_target --n1_strategy desc_cluster")
