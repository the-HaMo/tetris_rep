#!/bin/bash
# Script de configuración del entorno virtual para Tetris 3D

echo "=================================================="
echo "  Tetris 3D - Configuración de Entorno Virtual"
echo "=================================================="
echo ""

# Detectar Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo "Error: Python no encontrado. Instala Python 3.8 o superior."
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "Python encontrado: $PYTHON_VERSION"
echo ""

# Preguntar si tiene GPU NVIDIA
read -p "¿Tienes GPU NVIDIA? (s/N): " gpu_response
USE_GPU=false
if [[ "$gpu_response" =~ ^[sS]$ ]]; then
    USE_GPU=true
    echo "Modo GPU activado (requirements-gpu.txt)"
else
    echo "Modo CPU (requirements.txt)"
fi
echo ""

# Crear entorno virtual
VENV_DIR=".venv"

if [ -d "$VENV_DIR" ]; then
    echo "El entorno virtual ya existe en '$VENV_DIR'"
    read -p "¿Deseas recrearlo? (s/N): " response
    if [[ "$response" =~ ^[sS]$ ]]; then
        echo "Eliminando entorno virtual existente..."
        rm -rf "$VENV_DIR"
    else
        echo "Usando entorno virtual existente"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        if $USE_GPU; then
            pip install -r requirements-gpu.txt
        else
            pip install -r requirements.txt
        fi
        echo "Dependencias actualizadas"
        echo ""
        echo "Para activar el entorno virtual:"
        echo "  source $VENV_DIR/bin/activate"
        exit 0
    fi
fi

echo "Creando entorno virtual en '$VENV_DIR'..."
$PYTHON_CMD -m venv "$VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "Error: No se pudo crear el entorno virtual"
    exit 1
fi

source "$VENV_DIR/bin/activate"

echo "Actualizando pip..."
pip install --upgrade pip

echo ""
if $USE_GPU; then
    echo "Instalando dependencias GPU desde requirements-gpu.txt..."
    pip install -r requirements-gpu.txt
else
    echo "Instalando dependencias desde requirements.txt..."
    pip install -r requirements.txt
fi

if [ $? -ne 0 ]; then
    echo ""
    echo "Error durante la instalación de dependencias"
    exit 1
fi

# Configurar variables de entorno CUDA en el activate (solo modo GPU)
if $USE_GPU; then
    VENV_ABS=$(realpath "$VENV_DIR")
    CUDA_NVCC_PATH="$VENV_ABS/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cuda_nvcc"
    CUDA_RUNTIME_INCLUDE="$VENV_ABS/lib/python$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')/site-packages/nvidia/cuda_runtime/include"

    # Evitar duplicados
    if ! grep -q "CUDA_PATH" "$VENV_DIR/bin/activate"; then
        cat >> "$VENV_DIR/bin/activate" << EOF

# CUDA paths for CuPy (added by setup_env.sh)
export CUDA_PATH="$CUDA_NVCC_PATH"
export CPATH="$CUDA_RUNTIME_INCLUDE"
EOF
        echo "Variables CUDA añadidas al activate del venv"
    fi
fi

echo ""
echo "=================================================="
echo "  Instalacion completada con exito"
echo "=================================================="
echo ""
echo "Para activar el entorno virtual:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Para desactivar:"
echo "  deactivate"
echo ""
