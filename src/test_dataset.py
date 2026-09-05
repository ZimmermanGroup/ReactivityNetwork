from dataset import *
import numpy
import pandas as pd


# Test 1: few hand calculations among different categories
hand_calculated_dicts = [
    {
        "id": "3-replicate-median",
        "r1": "CC1=C(N2CCCC2=O)C=CC(Br)=C1",
        "r2": "OCCc1ccc(B(O)O)cc1",
        "plates": ["010", "010", "010"],
        "raw_values": [49.3922, 80.7347, 74.0474],
        "hand calculated": 74.0474,
    },
    {
        "id": "3-replicate-nonzero-median",
        "r1": "CN1C=NC=C(C1=O)Br",
        "r2": "B1(OC(C(O1)(C)C)(C)C)C2=CCOCC2",
        "plates": ["010", "010", "010"],
        "raw_values": [0, 60.3475, 0],
        "hand calculated": 60.3475,
    },
    {
        "id": "3-replicate-all-zero",
        "r1": "CC1=C(N2CCCC2=O)C=CC(Br)=C1",
        "r2": "Cc1ccc(B2OC(C)(C)C(C)(C)O2)c(C)c1",
        "plates": ["010", "010", "010"],
        "raw_values": [0, 0, 0],
        "hand calculated": 0.0,
    },
    {
        "id": "18-replicate-median",
        "r1": "CC1=C(N2CCCC2=O)C=CC(Br)=C1", 
        "r2": "COc1cc(B(O)O)cnc1OC",
        "plates": ["008b"]*1 + ["010"] * 3 + ["011"] * 12 + ["012"] * 2,
        "raw_values": [43.0395, 
                       126.165565, 129.5397, 130.7101, 
                       53.0088, 59.91516079, 84.82986644, 62.80057071, 61.53205476, 86.68794991, 
                       66.29125308, 82.58480226, 54.40921126, 131.9089286, 58.35724561, 130.8766205, 
                       151.805665, 149.8591],
        "hand calculated": 83.7073,
    },
]
aggregated_dict = pd.read_excel("data/aggregated_suzuki_dataset.xlsx")
for hand_dict in hand_calculated_dicts :
    r1 = hand_dict["r1"]
    r2 = hand_dict["r2"]
    # Checking yield aggregation went correctly
    print(aggregated_dict[
        (aggregated_dict["reactant_1&smiles"] == r1) & (aggregated_dict["reactant_2&smiles"]== r2)
    ]["suzuki_product_CAD_yield"].values[0], len(aggregated_dict[
        (aggregated_dict["reactant_1&smiles"] == r1) & (aggregated_dict["reactant_2&smiles"]== r2)
    ]))
    print(hand_dict["hand calculated"])
    assert aggregated_dict[
        (aggregated_dict["reactant_1&smiles"] == r1) & (aggregated_dict["reactant_2&smiles"]== r2)
    ]["suzuki_product_CAD_yield"].values[0] - hand_dict["hand calculated"] < 0.001
    # Checking plate sources are the same
    plate_vals = aggregated_dict[
        (aggregated_dict["reactant_1&smiles"] == r1) & (aggregated_dict["reactant_2&smiles"]== r2)
    ]["plate"].values[0][1:-1].split()
    plate_vals = [x[1:-1] for x in plate_vals] # removing the "'s
    assert sorted(plate_vals) == sorted(hand_dict["plates"])

# Test 2: comparing the 13 x 69 matrix
suzuki_data = SuzukiDataset(
    for_conventional_models = False,
    keepPhBr=False,
    from_notebook=False,
)
bb_list = suzuki_data.bb_smiles
core_list = suzuki_data.core_smiles
matrix_to_confirm = suzuki_data.yield_array

for i, core_smiles in enumerate(core_list):
    for j, bb_smiles in enumerate(bb_list) :
        assert aggregated_dict[
            (aggregated_dict["reactant_1&smiles"] == core_smiles) &
            (aggregated_dict["reactant_2&smiles"] == bb_smiles)
        ]["suzuki_product_CAD_yield"].values[0] - matrix_to_confirm[i, j] < 0.0001