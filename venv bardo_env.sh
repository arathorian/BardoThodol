# Crear entorno virtual
python3.11 -m venv bardo_env
source bardo_env/bin/activate

# Instalar dependencias Python
pip install --upgrade pip
pip install numpy scipy matplotlib seaborn
pip install qiskit plotly pandas scikit-learn
pip install jupyter jupyterlab ipykernel

# Instalar proyecto localmente
pip install -e .