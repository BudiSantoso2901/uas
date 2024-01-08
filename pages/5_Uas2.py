import streamlit as st
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

def moora_normalization(matrix):
    matrix = matrix.transpose()
    row_values = []
    norm_matrix = []

    for i in range(matrix.shape[0]):
        sum_row = sum([pow(x, 2) for x in matrix[i]])

        for j in range(matrix[i].shape[0]):
            r_value = matrix[i][j] / math.sqrt(sum_row)
            row_values.append(r_value)

        norm_matrix.append(row_values)
        row_values = []

    norm_matrix = np.asarray(norm_matrix)

    return norm_matrix.transpose()

def moora_weighted_normalization(n_matrix, c_weights):
    norm_weighted = n_matrix.transpose()

    for i in range(c_weights.shape[0]):
        norm_weighted[i] = [r * c_weights[i] for r in norm_weighted[i]]

    norm_weighted = np.asarray(norm_weighted)

    return norm_weighted.transpose()

# Implementasi Menghitung Nilai Optimasi
def optimize_value(w_matrix, label):
    y_values = []

    for i in range(w_matrix.shape[0]):
        max_val = []
        min_val = []

        for j in range(w_matrix[i].shape[0]):
            # Hitung benefit
            if label[j] == 1:
                max_val.append(w_matrix[i][j])
            # Hitung cost
            else:
                min_val.append(w_matrix[i][j])

        y = sum(max_val) - sum(min_val)
        y_values.append(y)

    return np.asarray(y_values)

def run_moora_app():
    st.set_page_config(
        page_title="Implementasi MOORA dengan Streamlit",
        page_icon="📊",
    )

    st.write("# Implementasi Metode MOORA menggunakan Streamlit")

    st.markdown(
        """
        Metode Multi-Objective Optimization by Ratio Analysis (MOORA) adalah salah satu metode
        yang digunakan dalam Sistem Pendukung Keputusan. Metode ini digunakan untuk mengambil keputusan
        berdasarkan beberapa kriteria dengan menimbang rasio antara nilai benefit dan cost.
        """
    )

    st.divider()

    st.write("## Input Nilai Keputusan")

    # Mendefinisikan Bobot Kriteria
    c_weights = st.text_input("Masukkan Bobot Kriteria (pisahkan dengan koma)", "0.3,0.2,0.2,0.15,0.15")
    c_weights = np.array([float(weight) for weight in c_weights.split(',')])

    # Mendefinisikan Label Kriteria (benefit dan cost)
    label = st.multiselect("Pilih Label Kriteria (benefit dan cost)", ["benefit", "cost","benefit", "cost","benefit", "cost","benefit", "cost","benefit", "cost","benefit", "cost",])

    # Mendefinisikan Jumlah Alternatif
    num_alternatives = st.number_input("Masukkan Jumlah Alternatif", min_value=1, step=1, value=5)

    # Mendefinisikan Jumlah Kriteria
    num_criteria = st.number_input("Masukkan Jumlah Kriteria", min_value=1, step=1, value=5)

    # Menggunakan Matrix Keputusan Sederhana sebagai contoh
    decision_matrix = np.zeros((num_criteria, num_alternatives))

    for i in range(num_criteria):
        for j in range(num_alternatives):
            decision_matrix[i][j] = st.number_input(f"Masukkan nilai untuk Kriteria-{i+1} dan Alternatif-{j+1}")

    if st.button("Proses"):
        # Normalisasi Matrix Keputusan
        norm_matrix = moora_normalization(decision_matrix)

        # Normalisasi Matrix Terbobot
        weighted_matrix = moora_weighted_normalization(norm_matrix, c_weights)

        # Menghitung Nilai Optimasi (y_values)
        result = optimize_value(weighted_matrix, label)

        # Mendapatkan Peringkat
        result_with_index = list(enumerate(result, 1))
        sorted_result = sorted(result_with_index, key=lambda x: x[1], reverse=True)
        ranking = [x[0] for x in sorted_result]

        st.write("Matrix Keputusan:")
        st.table(pd.DataFrame(decision_matrix, columns=[f'Alternatif {i+1}' for i in range(num_alternatives)]))

        st.write("Normalisasi Matrix:")
        st.table(pd.DataFrame(norm_matrix, columns=[f'Alternatif {i+1}' for i in range(num_alternatives)]))

        st.write("Normalisasi Matrix Terbobot:")
        st.table(pd.DataFrame(weighted_matrix, columns=[f'Alternatif {i+1}' for i in range(num_alternatives)]))

        st.write("Hasil Nilai Optimasi:")
        st.table(pd.DataFrame({'Alternatif': [f'Alternatif {i+1}' for i in range(len(result))], 'Nilai Optimasi': result}))

        st.write("Peringkat:")
        st.table(pd.DataFrame({'Peringkat': ranking}))

        # Display Visualization
        fig, ax = plt.subplots()
        ax.bar([f'Alternatif {i}' for i in ranking], result)
        ax.set_ylabel('Nilai Optimasi')
        ax.set_title('Peringkat Hasil Nilai Optimasi')
        st.pyplot(fig)
if __name__ == "__main__":
    run_moora_app()
