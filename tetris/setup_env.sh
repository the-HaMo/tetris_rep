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
    echo "❌ Error: Python no encontrado. Instala Python 3.8 o superior."
    exit 1
fi

# Verificar versión de Python
PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "✓ Python encontrado: $PYTHON_VERSION"
echo ""

# Crear entorno virtual
VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    echo "⚠ El entorno virtual ya existe en '$VENV_DIR'"
    read -p "¿Deseas recrearlo? (s/N): " response
    if [[ "$response" =~ ^[sS]$ ]]; then
        echo "🗑 Eliminando entorno virtual existente..."
        rm -rf "$VENV_DIR"
    else
        echo "👍 Usando entorno virtual existente"
        source "$VENV_DIR/bin/activate"
        pip install --upgrade pip
        pip install -r requirements.txt
        echo ""
        echo "✅ Dependencias actualizadas"
        echo ""
        echo "Para activar el entorno virtual:"
        echo "  source venv/bin/activate"
        exit 0
    fi
fi

echo "📦 Creando entorno virtual..."
$PYTHON_CMD -m venv "$VENV_DIR"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Error: No se pudo crear el entorno virtual"
    exit 1
fi

echo "✓ Entorno virtual creado en '$VENV_DIR'"
echo ""

# Activar entorno virtual
echo "🔧 Activando entorno virtual..."
source "$VENV_DIR/bin/activate"

# Actualizar pip
echo "📦 Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo ""
echo "📥 Instalando dependencias desde requirements.txt..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "  ✅ Instalación completada con éxito"
    echo "=================================================="
    echo ""
    echo "Para activar el entorno virtual en el futuro:"
    echo "  source venv/bin/activate"
    echo ""
    echo "Para desactivar el entorno virtual:"
    echo "  deactivate"
    echo ""
    echo "Para ejecutar el algoritmo Tetris 3D:"
    echo "  cd src"
    echo "  python tetris.py"
    echo ""
else
    echo ""
    echo "❌ Error durante la instalación de dependencias"
    exit 1
fi
