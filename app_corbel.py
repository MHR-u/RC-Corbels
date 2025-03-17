import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Streamlit app interface
st.set_page_config(page_title="Ultimate Shear Strength of RC Corbel", layout="wide")

# Input Definitions
input_definitions = {
    "√(f_ck)": "Square root of concrete compressive strength (MPa)",
    "a_v": "Shear span (mm)",
    "b": "Width of the section (mm)",
    "w_c": "Width of the column (mm)",
    "d": "Effective depth (mm)",
    "h": "Total depth (mm)",
    "a/d": "Shear span-to-depth ratio",
    "ρ_f": "Longitudinal reinforcement ratio (%)",
    "ρ_h": "Horizontal reinforcement ratio (%)",
    "ρ_v": "Vertical reinforcement ratio (%)",
    "f_yt": "Yield strength of transverse reinforcement (MPa)",
    "f_yh": "Yield strength of horizontal reinforcement (MPa)",
    "f_yv": "Yield strength of vertical reinforcement (MPa)"
}

# User Input
st.title("Ultimate Shear Strength of RC Corbel")
st.write("### Enter the input parameters within their valid ranges:")

inputs = {}
inputs["√(f_ck)"] = st.number_input("√(f_ck) (MPa)", min_value=3.87, max_value=10.25)
inputs["a_v"] = st.number_input("a_v (mm)", min_value=53.0, max_value=870.0)
inputs["b"] = st.number_input("b (mm)", min_value=115.0, max_value=600.0)
inputs["w_c"] = st.number_input("w_c (mm)", min_value=80.0, max_value=1200.0)
inputs["d"] = st.number_input("d (mm)", min_value=92.0, max_value=1059.0)
inputs["h"] = st.number_input("h (mm)", min_value=120.0, max_value=1143.0)
inputs["a/d"] = st.number_input("a/d", min_value=0.11, max_value=1.69)
inputs["ρ_f"] = st.number_input("ρ_f (%)", min_value=0.21, max_value=4.93)
inputs["ρ_h"] = st.number_input("ρ_h (%)", min_value=0.0, max_value=2.33)
inputs["ρ_v"] = st.number_input("ρ_v (%)", min_value=0.0, max_value=1.10)
inputs["f_yt"] = st.number_input("f_yt (MPa)", min_value=298.0, max_value=1480.0)
inputs["f_yh"] = st.number_input("f_yh (MPa)", min_value=0.0, max_value=760.0)
inputs["f_yv"] = st.number_input("f_yv (MPa)", min_value=0.0, max_value=614.0)

# ANN Weights and Biases
W1 = np.array([[-0.857, -1.8555, -1.7843, 1.3438, -0.7501, -0.1004, 1.3384, -0.6314, 0.6611, -0.1014, -1.1344, 0.48, 0.2732],
                   [2.088, 0.893, -0.3714, 0.0288, 0.4978, 1.365, -0.1418, 0.3566, 1.1395, 0.4936, 0.9285, 0.0154, -1.4623],
                   [-1.0935, -0.5067, -0.6966, -0.8442, 1.1677, 0.5412, 1.1516, -0.0431, 0.2999, -0.2702, 1.8377, -0.8287, 0.2188],
                   [-0.0983, -0.0726, 0.4051, -0.8092, -0.2254, -1.0904, -0.1668, 0.9218, -2.378, -1.2443, -1.108, -2.0322, -0.3192],
                   [0.0417, 0.9866, 1.0395, 0.483, 0.9853, 0.4549, -0.5149, 2.1451, 0.3961, -0.6155, -0.3212, -1.3314, -0.5588],
                   [0.5504, 1.0666, 1.0758, -0.3431, 0.2757, -1.5934, 0.4455, -2.6158, -0.9969, 1.3799, 1.284, -0.1074, -1.952],
                   [0.429, 0.767, 1.3301, -0.3974, -0.2283, 1.8197, -1.6468, -0.3421, -1.3205, -0.3855, 0.5053, 0.2671, 1.0721],
                   [0.3467, -0.7774, -0.2492, -1.8511, -0.6145, -2.1708, 0.641, -1.9226, -0.4238, -0.934, 0.1103, 0.0587, 0.681],
                   [1.3998, 0.0039, 2.0116, 0.1473, -0.3997, 0.2102, -1.3228, -0.1925, 1.2611, -1.5971, 0.0281, 0.8871, -1.4558],
                   [-0.5919, 0.5765, 0.755, 1.0143, -2.1026, -0.0925, 0.6412, 0.2713, 0.6718, 1.7243, -0.1475, 0.8652, 1.598]])  # Replace with your W1 matrix
B1 = np.array([1.9628, -4.7316, 2.6417, 3.3912, 0.1861, -0.369, 0.2887, 2.9229, -1.2502, -3.4438])  # Replace with your B1 array
W2 = np.array([
        [0.8128, 0.5497, -0.3654, 0.0523, -0.4269, -0.3089, -1.0734, -0.1305, 0.6526, 0.6145],
        [-0.9653, 0.5534, -0.5805, -0.3921, -0.6196, 0.6622, -0.6217, -0.2844, -0.7128, 0.1383],
        [-0.6277, 1.2147, -0.8423, 0.1506, 0.135, 0.6732, -0.3792, 0.5254, 0.362, 1.0536],
        [0.2601, -0.7762, 0.989, -0.7996, -0.5325, 1.114, -0.1785, -0.5904, -0.0641, 0.0926],
        [0.5478, -1.0013, 0.5577, -1.0751, -0.4083, 1.2723, -0.1499, -0.9048, 0.1191, -0.0497],
        [-1.4048, -0.2243, -0.5184, -0.5061, 0.5336, -1.1036, 0.8138, -0.1046, -0.1222, 1.0099],
        [-0.5414, 0.957, -0.8039, -0.4263, 0.1208, 0.4027, -0.5892, -0.7088, -0.3365, -0.1322],
        [2.1116, 0.1318, 0.8281, 0.086, -0.508, 1.4086, -0.9856, -1.8136, -0.4433, -0.4073],
        [-0.1352, -0.5088, -0.3798, -0.4733, -0.651, 0.8054, -0.8273, -0.5509, -0.4711, -0.0384]
    ])  # Replace with your W2 matrix
B2 = np.array([-1.7159, 1.3111, 0.5415, -0.3307, 0.2794, -0.6069, -0.9185, 1.9076, -1.8871])  # Replace with your B2 array
W3 = np.array([-0.2839, 0.2487, 0.0105, -0.6244, 0.5289, 0.3096, -0.3237, -1.389, -0.4037])  # Replace with your W3 array
B3 = np.array([0.8576])  # Replace with your B3 array

# ANN Calculation (Tansig activation function)
def tansig(x):
    return 2 / (1 + np.exp(-2 * x)) - 1

# Forward pass through the ANN
def predict(inputs):
    X = np.array(list(inputs.values())).reshape(-1, 1)
    A1 = tansig(np.dot(W1, X) + B1.reshape(-1, 1))
    A2 = tansig(np.dot(W2, A1) + B2.reshape(-1, 1))
    Vn = np.dot(W3, A2) + B3  # Output layer (Linear activation)
    return Vn.item()

# Predict Vn
if st.button("Calculate Vn"):
    Vn = predict(inputs)
    st.write(f"### Predicted Vn: {Vn:.2f} kN")

# Plot Generation
st.write("## Plot Vu vs. Selected Input")
selected_input = st.selectbox("Select an input to plot against Vu:", list(inputs.keys()))

if selected_input:
    input_range = np.linspace(st.session_state[selected_input + "_min"], 
                              st.session_state[selected_input + "_max"], 100)
    Vn_values = []

    for value in input_range:
        temp_inputs = inputs.copy()
        temp_inputs[selected_input] = value
        Vn_values.append(predict(temp_inputs))

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(input_range, Vn_values, label=f"Vu vs. {selected_input}", color='blue')
    ax.set_title(f"Vu vs. {selected_input}")
    ax.set_xlabel(f"{selected_input} ({input_definitions[selected_input]})")
    ax.set_ylabel("Vu (kN)")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# Displaying Definitions
st.write("## Input Definitions")
for key, definition in input_definitions.items():
    st.write(f"**{key}:** {definition}")

