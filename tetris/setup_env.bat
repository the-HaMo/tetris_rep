@echo off
REM Script de configuración del entorno virtual para Tetris 3D (Windows)

echo ==================================================
echo   Tetris 3D - Configuracion de Entorno Virtual
echo ==================================================
echo.

REM Verificar Python
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Python no encontrado. Instala Python 3.8 o superior.
    pause
    exit /b 1
)

REM Mostrar version de Python
python --version
echo.

REM Crear entorno virtual
set VENV_DIR=venv

if exist %VENV_DIR% (
    echo El entorno virtual ya existe en '%VENV_DIR%'
    set /p response="Deseas recrearlo? (s/N): "
    if /i "%response%"=="s" (
        echo Eliminando entorno virtual existente...
        rmdir /s /q %VENV_DIR%
    ) else (
        echo Usando entorno virtual existente
        call %VENV_DIR%\Scripts\activate.bat
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        echo.
        echo Dependencias actualizadas
        echo.
        echo Para activar el entorno virtual:
        echo   venv\Scripts\activate
        pause
        exit /b 0
    )
)

echo Creando entorno virtual...
python -m venv %VENV_DIR%

if not exist %VENV_DIR% (
    echo Error: No se pudo crear el entorno virtual
    pause
    exit /b 1
)

echo Entorno virtual creado en '%VENV_DIR%'
echo.

REM Activar entorno virtual
echo Activando entorno virtual...
call %VENV_DIR%\Scripts\activate.bat

REM Actualizar pip
echo Actualizando pip...
python -m pip install --upgrade pip

REM Instalar dependencias
echo.
echo Instalando dependencias desde requirements.txt...
pip install -r requirements.txt

if %errorlevel% equ 0 (
    echo.
    echo ==================================================
    echo   Instalacion completada con exito
    echo ==================================================
    echo.
    echo Para activar el entorno virtual en el futuro:
    echo   venv\Scripts\activate
    echo.
    echo Para desactivar el entorno virtual:
    echo   deactivate
    echo.
    echo Para ejecutar el algoritmo Tetris 3D:
    echo   cd src
    echo   python tetris.py
    echo.
) else (
    echo.
    echo Error durante la instalacion de dependencias
)

pause
