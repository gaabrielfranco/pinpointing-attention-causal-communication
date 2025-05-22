# Pythia-160M
for (( layer=0; layer < 12; ++layer ))
do
    python3 compute_all_omega_svd_pythia.py -l $layer
done

# Gemma-2 2B
for (( layer=0; layer < 26; ++layer ))
do
    python3 compute_all_omega_svd_gemma.py -l $layer
done
