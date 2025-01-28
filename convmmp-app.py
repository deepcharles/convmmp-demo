import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd

from mpcsc import multipathcsc  # our library


st.title("Convolutional Sparse Coding with Multipath Orthogonal Matching Pursuit")

st.write("""Implementation of the algorithm described in (Gomes et al., 2024):

- Gomes, Y., Truong, C., Saut, J.-P., Hafid, F., Prieur, P., & Oudre, L. (2025). Convolutional Sparse Coding with Multipath Orthogonal Matching Pursuit. Proceedings of the IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).

The code is available at this address: https://github.com/deepcharles/convmmp

Sample data can be downloaded here: https://kiwi.cmla.ens-cachan.fr/index.php/s/9mf3gCb3aPeGTtZ
""")

# Sidebar instructions
st.sidebar.title("Instructions")
st.sidebar.write(
    "1. Upload multiple univariate signals (`.txt` or `.csv` files).\n"
    "2. Upload one dictionary (`.txt` or `.csv` file).\n\n"
    "The app computes the optimal decomposition of the signals using the dictionary atoms.\n\n The user can modify the parameters below. See article for a detailled description."
)

# Sidebar for parameter customization
st.sidebar.title("Parameters")
n_atoms_to_find = st.sidebar.number_input(
    "Number of atoms to find", min_value=1, value=3, step=1
)
distance = st.sidebar.number_input(
    "Temporal distance (in samples) between two atoms of same depth", min_value=1, value=20, step=1
)
width = st.sidebar.number_input(
    "Breadth of the multipath tree", min_value=1, value=3, step=1
)
n_paths = st.sidebar.number_input(
    "Number of paths ", min_value=1, value=5, step=1
)

# Initialize variables
all_signals = []
dictionary = None

# File uploader for all_signals
st.header("Upload signal files")
signal_files = st.file_uploader(
    "Upload multiple signal files (.txt, .csv). Each file must contain a univariate signal.", 
    type=["txt", "csv"], 
    accept_multiple_files=True
)

if signal_files:
    st.write("**Uploaded signals:**")
    for signal_file in signal_files:
        try:
            # Use comma delimiter for CSV files and whitespace for TXT files
            delimiter = "," if signal_file.name.endswith(".csv") else None
            signal_data = np.loadtxt(signal_file, delimiter=delimiter)

            # Ensure signal_data is at least 2D
            if signal_data.ndim == 1:
                signal_data = signal_data[:, np.newaxis]  # Convert 1D to 2D by adding a dimension at the end

            all_signals.append(signal_data)
            n_samples, n_dims = signal_data.shape
            # Display file details
            st.write(f"- {signal_file.name} ({n_samples} samples)")
        except Exception as e:
            st.error(f"Error processing file {signal_file.name}: {e}")

# File uploader for dictionary
st.header("Upload dictionary file")
dictionary_file = st.file_uploader(
    "Upload a single dictionary file (.txt, .csv). The code expects an array of shape (`n_atoms`, `n_samples_atom`) where `n_atoms` is the number of atoms and `n_samples_atom` is the atom's length.",
    type=["txt", "csv"]
)

if dictionary_file:
    try:
        # Use comma delimiter for CSV files and whitespace for TXT files
        delimiter = "," if dictionary_file.name.endswith(".csv") else None
        dictionary_data = np.loadtxt(dictionary_file, delimiter=delimiter)

        # Ensure dictionary_data is 3D
        if dictionary_data.ndim == 2:
            dictionary_data = dictionary_data[:, :, np.newaxis]  # Add a new dimension at the end
        elif dictionary_data.ndim == 1:
            st.error("Dictionary must have at least 2 dimensions (e.g., matrix or tensor).")
        dictionary = dictionary_data

        n_atoms, n_samples_atom, n_dims = dictionary.shape
        # Display file details
        st.write(f"**Uploaded dictionary:** {dictionary_file.name} ({n_atoms} atoms of length {n_samples_atom} samples)")
    except Exception as e:
        st.error(f"Error processing dictionary file {dictionary_file.name}: {e}")
else:
    st.info("Please upload a `.txt` or `.csv` file for `dictionary`.")

# Dimension checks
if all_signals and dictionary is not None:
    valid = True

    # Check if dictionary is now 3D
    if dictionary.ndim != 3:
        st.error("The dictionary must have exactly 3 dimensions after preprocessing!")
        valid = False

    # Check if the last dimension of dictionary matches the last dimension of all signals
    last_dim_dict = dictionary.shape[-1]
    for i, signal in enumerate(all_signals):
        if signal.shape[-1] != last_dim_dict:
            st.error(
                f"The last dimension of dictionary ({last_dim_dict}) does not match the last dimension of signal {i+1} "
                f"({signal.shape[-1]})."
            )
            valid = False

    # Check if each atom is shorter than the signals
    n_samples_atom = dictionary.shape[1]
    for i, signal in enumerate(all_signals):
        if signal.shape[0] < n_samples_atom:
            st.error(f"Signal {i+1} is shorter than the atoms.")
            valid = False

    if valid:
        st.success("The dictionary and all signals are compatible!")

        # Run `multipathcsc` on each signal
        st.header("Results from Multipath CSC")
        for i, signal in enumerate(all_signals):
            st.write(f"### Signal {i+1} Plot")
            try:
                # Call multipathcsc with required parameters
                approx, time_idxs, atom_idxs, vals, path = multipathcsc(
                    signal=signal,
                    dictionary=dictionary,
                    n_atoms_to_find=n_atoms_to_find,
                    distance=distance,
                    n_paths=n_paths,
                    width=width,
                    n_jobs=-2
                )

                # Plotting
                fig1, ax = plt.subplots(figsize=(10, 4))
                ax.plot(signal, label="Original", color="blue")
                ax.plot(approx, label="Smoothed", color="orange")
                ax.legend()
                ax.set_xmargin(0)
                ax.set_title(f"Signal {i+1}: Original vs Smoothed (path: {np.array(path) + 1})")
                ax.set_xlabel("Time")
                ax.set_ylabel("Amplitude")
                # Display the plot in Streamlit
                st.pyplot(fig1)

                # Second plot: Atom contributions
                fig2, ax2 = plt.subplots(figsize=(10, 4))
                ax2.set_xmargin(0)
                n_samples_atom = dictionary.shape[1]
                for k_atom in range(time_idxs.size):
                    approx_single_atom = np.zeros_like(signal)
                    start = time_idxs[k_atom]
                    end = start + n_samples_atom
                    atom = dictionary[atom_idxs[k_atom]]
                    val = vals[k_atom]
                    approx_single_atom[start:end] = val * atom
                    ax2.plot(approx_single_atom, label=f"Atom {k_atom+1}")
                ax2.legend()
                ax2.set_title(f"Signal {i+1}: Atom Contributions")
                ax2.set_xlabel("Time")
                ax2.set_ylabel("Amplitude")
                # Display the plot in Streamlit
                st.pyplot(fig2)

                # Concatenate results and prepare CSV download
                concatenated = np.column_stack((time_idxs, atom_idxs, vals))
                csv_buffer = StringIO()
                pd.DataFrame(concatenated, columns=["time_indexes", "atom_indexes", "activation_values"]).to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label=f"Download Results for Signal {i+1} as CSV",
                    data=csv_data,
                    file_name=f"signal_{i+1}_results.csv",
                    mime="text/csv"
                )

            except Exception as e:
                st.error(f"Error processing signal {i+1}: {e}")
