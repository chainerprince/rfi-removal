# RFI Removal/Protecting the radio sky using ML

Caltech CS163 · v1.0

Open [the notebook](notebooks/01_first_rfi_experiment.ipynb).
The [HTML copy](notebooks/01_first_rfi_experiment.html) has all plots and equations saved.

## Google Colab

1. Open [Colab](https://colab.research.google.com/).
2. Select **File → Upload notebook** and upload `01_first_rfi_experiment.ipynb`.
3. Open the Files panel. Upload `cs163-rfi-starter-v1.0.zip` and wait for the upload to finish.
4. Use a CPU runtime for the tutorial. Run the cells in order.
5. Download your edited notebook before ending the session.

Both files are needed. The first cell extracts the ZIP and installs the included
library. Colab files can disappear when the session ends. Use a GPU for U-Net training.

## Local Jupyter

Extract the ZIP. Open a terminal in `cs163-rfi-starter-v1.0`.
Use Python 3.11 or newer.

```sh
python -m pip install -e ".[notebook]"
python -m jupyter lab notebooks/01_first_rfi_experiment.ipynb
```

Installation needs internet or installed dependencies. The experiments and HTML
then work offline. Reading links need internet.

## Read one input and its mask

```python
import pandas as pd
from cs163_rfi import load_sample, flag_rfi, score_mask
from cs163_rfi.plotting import plot_sample

data = "data/starter-v1.0"
catalog = pd.read_csv(f"{data}/dataset.csv")
display(catalog.head())

sample = load_sample(data, sample_id=18, antenna=0)
power = sample["power"]
labels = sample["rfi_mask"]
flags = flag_rfi(power, method="sk-bandpass")
plot_sample(sample, flags, factor=16)
print(score_mask(flags, labels))
```

Both arrays have shape **(4096, 128)**: time × frequency.
Power is float32; labels and flags are Boolean. A true label means injected RFI
power is at least 1% of mean receiver-noise power. Weaker contributions count as
negative labels. A true detector flag means remove that sample. Neither mask
changes the stored power. Display averaging does not change the data or scores.

Each recording lasts **134.218 ms**, with **32.768 µs** samples and **128 channels**
spanning **3.91 MHz** near 1350 MHz. It contains ten antennas, A00–A09, and one
linear feed, X. The full power array has shape `(4096, 10, 128, 1)`.

## Data and splits

`data/starter-v1.0/dataset.csv` lists **200 recordings**:
100 train, 60 validation and 40 test. Ten antennas per recording give
**1,000 training pairs, 600 validation pairs and 400 test pairs**.
Keep all antennas and windows from one recording in the same split.

The CSV gives paths, RFI types, strength and duty cycle. Each numbered folder has
`power.npy`, `rfi_mask.npy`, `rfi_power.npy` and a short `recording.json`.
Use the loader: it converts stored power to float32 and unpacks the Boolean mask.
Injected power and transmitter settings are for analysis, not network inputs.

The set contains 180 RFI recordings, ten noise-only controls and ten
astronomical-pulse controls. The astronomical controls contain no RFI.

## Baselines and first model

Call `flag_rfi` with `method="sk"`, `"bandpass"`, or `"sk-bandpass"`.
The combined method also expands channel clusters. These use the DSA-110
spectral-kurtosis and bandpass routines. A flagged channel removes its full
134 ms block, on that antenna only. Keep the defaults for the first comparison.
Later, `m_values=(64,)` tests one SK accumulation length.

Start with a per-antenna U-Net. A batch has shape `(batch, 1, 4096, 128)`.
Predict one RFI probability per pixel, then choose a threshold on validation data.

For normalization, pool power across time, frequency and antennas from **all
noise-only training recordings**. Calculate **one mean and one standard deviation**.
Use `(power - mean) / std` for every model input in train, validation and test.
Do not normalize each recording separately. Leave labels unchanged and give
classical detectors the original power. Keep chosen settings fixed for the test.

Run the three baselines. Report precision, recall, false-positive rate and removal
by RFI family. Explain one miss and one false flag. Test the explanation with
controlled data. The notebook shows how to generate and save new recordings.
Each student reads one ML RFI paper; any ML RFI mitigation paper is welcome.

## Further reading

- [Buch et al.: Real-time RFI Mitigation Techniques in Radio Astronomy and their Practical Limitations](https://arxiv.org/html/2512.01954v2).
  Start with Table 1 for methods, benefits and limitations. Gregory Hellbourg is a coauthor.
- [Smith, Lynch and Pisano: Simulating Spectral Kurtosis Mitigation Against Realistic RFI Signals](https://arxiv.org/abs/2207.07642).
- [NRAO: interferometry](https://www.cv.nrao.edu/~sransom/web/Ch3.html#S7).

The examples use an ideal receiver and simplified transmitter waveforms.
They are a first experiment, not a model of every effect in real data.
Image credits are in `assets/CREDITS.md`.
