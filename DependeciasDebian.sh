sudo apt update
sudo apt upgrade -y

# Paquetes básicos de desarrollo y Python
sudo apt install -y build-essential cmake git pkg-config \
    python3 python3-pip python3-venv python3-dev

# Librerías numéricas y científicas (BLAS/LAPACK, FFT, HDF5, etc.)
sudo apt install -y libblas-dev liblapack-dev libopenblas-dev \
    libfftw3-dev libhdf5-dev libhdf5-serial-dev hdf5-tools

# Compresión y utilidades para wheels y dependencias autóctonas
sudo apt install -y zlib1g-dev libbz2-dev liblzma-dev libssl-dev \
    libffi-dev libxml2-dev libxslt1-dev

# MPI (opcional, para dask-jobqueue / entornos HPC)
sudo apt install -y openmpi-bin libopenmpi-dev

# Herramientas adicionales útiles
sudo apt install -y curl unzip ca-certificates

