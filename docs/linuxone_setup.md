# IBM LinuxONE setup (what was actually done, and why)

All steps below were executed on the target VM (Ubuntu 24.04.5, `s390x`, big-endian, z/VM 7.3, 2 vCPU, ~4 GB RAM, ~45 GB disk, `nnpa` CPU flag present, passwordless `sudo`).
Nothing was run on a different machine except a one-off *file-format* export of the dataset tables (see "Data").

## 0. Connection hygiene
* The private key stays in a `chmod 600` file outside the repository; it is never printed, copied into notebooks/sources/logs, or committed (`.gitignore` excludes `*.pem`, `.env`).
* Host-key checking is **never disabled**: the host keys are pinned in a dedicated `known_hosts` file and every connection uses `StrictHostKeyChecking=yes`.
  Compare the pinned fingerprints with the ones your IBM console shows (`ssh-keygen -lf /etc/ssh/ssh_host_*_key.pub` on the VM).
* Jupyter/inference endpoints are bound to `127.0.0.1` and reached through SSH port forwarding (below), never exposed publicly.

## 1. Hardware inspection (recorded by the notebook, section 2)
`uname -m` (= `s390x`), `lscpu` (Machine type 9175, Hypervisor z/VM 7.3, flags `vx vxd vxe vxe2 vxp nnpa ...`), `free -h`, `df -h`, `nproc`, `/etc/os-release`, Python and package inventory,
Docker/Podman availability (**neither installed**).

## 2. Environment decision: native Python, not the community container
The reference environment <https://github.com/linuxone-community-cloud/jupyter-lab-ml> ships a Docker image
(`registry.linuxone.cloud.marist.edu/l1cc/jupyterlab-image-s390x:latest`). It was **investigated but not used**:
* the registry answers `401 Unauthorized` to anonymous manifest requests and no credentials were provided, so the image and its layers could not be verified;
* the VM has neither Docker nor Podman (installing a container runtime is a system-level change for no functional gain here);
* its published `packages.txt` (TensorFlow, `ibm-zdnn-plugin`, JAX, PyTorch ...) is larger than what MedRelay needs, which is scikit-learn/XGBoost/NetworkX/llama.cpp - all available natively.
The native route below is fully scripted and was verified end to end.

## 3. System packages (apt, one time)
```bash
sudo apt-get install -y python3-pip python3-venv python3-dev build-essential cmake ninja-build pkg-config libffi-dev libssl-dev unzip libzdnn-dev
sudo apt-get install -y python3-numpy python3-scipy python3-pandas python3-sklearn python3-matplotlib python3-networkx python3-joblib python3-yaml python3-psutil \
     python3-nbclient python3-nbformat python3-ipykernel python3-ipywidgets python3-plotly python3-xgboost python3-onnx python3-fastapi python3-uvicorn python3-requests python3-tqdm python3-httpx
```
Why apt for the scientific stack: PyPI has **no `s390x` wheels for NumPy/pandas/scikit-learn/pyarrow**; `pip install numpy` starts a multi-hour source build (tried and aborted). Ubuntu's precompiled `s390x` packages
(NumPy 1.26.4, pandas 2.1.4, scikit-learn 1.4.1, XGBoost 2.0.3, NetworkX 2.8.8) work immediately. `libzdnn-dev` (zDNN 1.0.1) provides the NNPA device test in section 16.

## 4. Python environment
```bash
python3 -m venv --system-site-packages .venv          # re-use the apt packages
.venv/bin/pip install --only-binary=:all: pydantic jupyterlab huggingface_hub gguf     # pure/binary wheels that exist for s390x
.venv/bin/pip install diskcache jinja2 typing_extensions
```
`pyarrow` has no `s390x` wheel, so **Parquet cannot be read on this host**; the dataset tables are consumed as gzip CSV (see "Data").

## 5. llama.cpp (GGUF inference) built for s390x
```bash
export CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_VXE=ON -DGGML_CPU_REPACK=ON -DCMAKE_C_FLAGS=-march=z16 -DCMAKE_CXX_FLAGS=-march=z16"
export CMAKE_BUILD_PARALLEL_LEVEL=2
.venv/bin/pip install llama-cpp-python --no-binary llama-cpp-python --no-cache-dir --no-deps
```
Pitfall found: with `GGML_NATIVE=ON` the build detects this machine as `arch15` and fails (`gcc: unrecognized argument in option '-march=arch15'`, GCC 13.3 knows up to z16); with `GGML_NATIVE=OFF` and no `-march` the vector intrinsics
are rejected (`requires z14 or higher`). Targeting **z16** works on this newer machine. The build reports `VXE = 1`.

## 6. Models
```bash
.venv/bin/python scripts/fetch_models.py            # IBM Granite 4.0 350M Q8_0 and 1B Q4_0 (Apache-2.0), pinned revisions, SHA-256 recorded in models/llm/MODELS.json
for m in granite-4.0-350m-Q8_0 granite-4.0-1b-Q4_0; do
  cp models/llm/original_le/$m.gguf models/llm/$m.be.gguf
  echo YES | .venv/bin/python -m gguf.scripts.gguf_convert_endian models/llm/$m.be.gguf big
done
```
**Endianness:** Hugging Face GGUF files are little-endian; this host is big-endian. `src/llm_backend.py` reads the GGUF header and refuses a mismatched file with this exact instruction.
Q8_0 and Q4_0 are used because they are supported by the endian converter. Correctness check: the converted 350M model reproduces the x86 reference output token for token.

## 7. Data
The dataset-engineering pipeline (`dataset_pipeline_src.tar.gz`, unpacked into `third_party/`) had already produced Parquet tables. Because `pyarrow` is unavailable here they were exported **once** to gzip CSV and uploaded to `data/source/`
(origin checksums in `data/source/reports/SOURCE_PARQUET_SHA256.json`). The replay stream is *rebuilt on the VM* (`src/data_loader.build_stream`) from the generator tables so every payload carries chain metadata and all patient encounters.
Kaggle credentials were not copied to the VM; the no-show table is the pipeline's already-cleaned export.

## 8. Run
```bash
.venv/bin/python scripts/build_notebook.py                    # regenerate the notebook from scripts/nb_cells_*.py
.venv/bin/python scripts/run_notebook.py --mode fast --reset   # clean-kernel execution (nbclient), development subset
.venv/bin/python scripts/run_notebook.py --mode full           # every record
.venv/bin/python -m pytest -q tests                            # unit/integration tests
```
Interactive use through an SSH tunnel (nothing is exposed publicly):
```bash
# on the VM
.venv/bin/jupyter lab --no-browser --ip 127.0.0.1 --port 8888
# on your laptop
ssh -i <key> -N -L 8888:127.0.0.1:8888 linux1@<vm-ip>      # then open http://127.0.0.1:8888 (token printed by jupyter)
```
## 9. Annotation conventions used for the developer-annotated extraction set
One obligation per distinct required action; task type per `src/ontology.py` (reviewing a result = `followup_review`; arranging/booking an appointment, referral or "follow up with <provider>" = `appointment_referral`;
revisit timing without a scheduling verb = `followup_review`; instructions to the patient = `patient_communication`); a deadline only when a relative deadline is explicitly stated for *that* action (durations such as "for ten days" are not
deadlines; ranges use the upper bound); a role only when the text names who must act; dependency = the action needs an earlier one (review <- test, notification <- review).
