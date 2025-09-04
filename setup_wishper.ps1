# 🛠️ Скрипт для настройки окружения OpenAI Whisper на Windows 🛠️
#
# Этот PowerShell-скрипт автоматизирует полную установку и настройку программного
# окружения, необходимого для работы системы распознавания речи OpenAI Whisper
# с использованием GPU от NVIDIA.
#
# Основные задачи:
# - Проверка и установка необходимых компонентов (Python, CUDA, драйверы)
# - Создание виртуального окружения Python
# - Установка PyTorch с поддержкой GPU
# - Установка библиотеки openai-whisper и зависимостей
# - Тестирование установки
#
# Порядок использования:
# 1. Запустите PowerShell от имени администратора
# 2. Разрешите выполнение скриптов: Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# 3. Запустите скрипт: .\setup_whisper.ps1
#
# Автор: Адаптация для Windows
# На основе оригинального скрипта: Михаил Шардин https://shardin.name/
# Версия: 1.2 Windows
#

Write-Host "🚀 Установка окружения для OpenAI Whisper на Windows" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green

# Проверка прав администратора
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Host "⚠️  Запустите скрипт от имени администратора!" -ForegroundColor Red
    exit 1
}

# Информация о системе
Write-Host "📋 Информация о системе:" -ForegroundColor Yellow
systeminfo | Select-String "OS Name", "OS Version", "System Type"
Write-Host ""

# Проверка и установка Python
Write-Host "🐍 Проверка установки Python..." -ForegroundColor Yellow
if (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonVersion = python --version
    Write-Host "✅ Python установлен: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "⚠️  Python не найден. Установите Python 3.8+ с официального сайта" -ForegroundColor Red
    Write-Host "Скачайте с: https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "Не забудьте отметить 'Add Python to PATH' при установке" -ForegroundColor Yellow
    exit 1
}

# Проверка NVIDIA драйверов
Write-Host "🎮 Проверка NVIDIA драйверов..." -ForegroundColor Yellow
try {
    $nvidiaInfo = nvidia-smi 2>$null
    if ($nvidiaInfo) {
        Write-Host "✅ NVIDIA драйверы установлены" -ForegroundColor Green
        $gpuName = nvidia-smi --query-gpu=name --format=csv,noheader,nounits
        $gpuMemory = nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
        Write-Host "🎮 GPU: $gpuName ($gpuMemory MB)" -ForegroundColor Cyan
    } else {
        Write-Host "⚠️  NVIDIA драйверы не найдены" -ForegroundColor Red
        Write-Host "Установите драйверы с: https://www.nvidia.com/Download/index.aspx" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  NVIDIA драйверы не найдены" -ForegroundColor Red
}

# Улучшенная проверка CUDA
Write-Host "🔧 Проверка CUDA..." -ForegroundColor Yellow
$cudaDetected = $false
$cudaVersion = ""

# Проверка различных путей установки CUDA
$possibleCudaPaths = @(
    $env:CUDA_PATH,
    $env:CUDA_PATH_V13_0,
    $env:CUDA_PATH_V12_0, 
    $env:CUDA_PATH_V11_8,
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8"
)

foreach ($path in $possibleCudaPaths) {
    if ($path -and (Test-Path $path)) {
        $cudaDetected = $true
        $cudaVersion = $path -replace '.*\\v(\d+\.\d+).*', '$1'
        Write-Host "✅ CUDA toolkit обнаружен: $path" -ForegroundColor Green
        Write-Host "📋 Версия CUDA: $cudaVersion" -ForegroundColor Cyan
        break
    }
}

if (-not $cudaDetected) {
    Write-Host "⚠️  CUDA toolkit не найден" -ForegroundColor Yellow
    Write-Host "Рекомендуется установить CUDA 11.8 или новее с официального сайта NVIDIA" -ForegroundColor Yellow
}

# Создание виртуального окружения
Write-Host "🏠 Создание виртуального окружения..." -ForegroundColor Yellow
python -m venv .venv

# Активация виртуального окружения
Write-Host "🔓 Активация виртуального окружения..." -ForegroundColor Yellow
.\.venv\Scripts\Activate.ps1

# Обновление pip
Write-Host "⬆️  Обновление pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Установка PyTorch с поддержкой CUDA
Write-Host "🔥 Установка PyTorch с поддержкой GPU..." -ForegroundColor Yellow

# Определяем правильный индекс для PyTorch на основе версии CUDA
if ($cudaDetected) {
    switch -Regex ($cudaVersion) {
        "13\.\d+" {
            Write-Host "📦 Установка PyTorch для CUDA 13.x..." -ForegroundColor Cyan
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
        "12\.\d+" {
            Write-Host "📦 Установка PyTorch для CUDA 12.x..." -ForegroundColor Cyan
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
        "11\.\d+" {
            Write-Host "📦 Установка PyTorch для CUDA 11.x..." -ForegroundColor Cyan
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        }
        default {
            Write-Host "📦 Установка PyTorch для последней версии CUDA..." -ForegroundColor Cyan
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
        }
    }
} else {
    Write-Host "📦 Установка CPU версии PyTorch..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Установка OpenAI Whisper с игнорированием конфликтов зависимостей
Write-Host "🎙️  Установка OpenAI Whisper..." -ForegroundColor Yellow
pip install openai-whisper --no-deps
pip install openai-whisper  # Повторная установка для зависимостей

# Дополнительные библиотеки с обработкой конфликтов
Write-Host "📚 Установка дополнительных библиотек..." -ForegroundColor Yellow

# Установка основных зависимостей с игнорированием конфликтов
$packages = @("numpy", "scipy", "librosa", "soundfile", "pydub")
foreach ($package in $packages) {
    Write-Host "📦 Установка $package..." -ForegroundColor Cyan
    pip install $package --no-deps
    pip install $package  # Повторная установка для зависимостей
}

# Установка недостающих зависимостей для разрешения конфликтов
Write-Host "🔧 Разрешение конфликтов зависимостей..." -ForegroundColor Yellow
$missingDeps = @("h11>=0.8", "fastapi<1,>=0", "pydantic<2.9,>=2.4.1")
foreach ($dep in $missingDeps) {
    Write-Host "📦 Установка недостающей зависимости: $dep" -ForegroundColor Cyan
    pip install $dep
}

# Установка FFmpeg (если не установлен)
Write-Host "🎵 Проверка FFmpeg..." -ForegroundColor Yellow
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "⚠️  FFmpeg не найден. Установите вручную:" -ForegroundColor Yellow
    Write-Host "1. Скачайте с: https://www.gyan.dev/ffmpeg/builds/" -ForegroundColor Yellow
    Write-Host "2. Распакуйте и добавьте в PATH" -ForegroundColor Yellow
    Write-Host "3. Или используйте: winget install ffmpeg" -ForegroundColor Yellow
} else {
    Write-Host "✅ FFmpeg установлен" -ForegroundColor Green
}

# Проверка установленных пакетов
Write-Host "📊 Проверка установленных пакетов..." -ForegroundColor Yellow
pip list | Select-String "torch", "whisper", "numpy", "scipy", "librosa"

# Тестирование установки
Write-Host "🧪 Тестирование установки..." -ForegroundColor Yellow
python -c "
import torch
import whisper
print(f'PyTorch версия: {torch.__version__}')
print(f'CUDA доступна: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        print(f'GPU: {gpu_name}')
        print(f'CUDA версия: {torch.version.cuda}')
        print(f'GPU устройств: {torch.cuda.device_count()}')
        
        # Тест совместимости
        test_tensor = torch.zeros(10, 10).cuda()
        result = test_tensor + 1
        print('✅ GPU совместим с PyTorch')
        
    except Exception as e:
        print(f'⚠️  Ошибка GPU: {e}')
        print('🔄 Будет использоваться CPU режим')
else:
    print('💻 Будет использоваться CPU')

print('✅ Whisper импортирован успешно')

# Дополнительная проверка основных зависимостей
try:
    import numpy
    import scipy
    import librosa
    import soundfile
    import pydub
    print('✅ Все основные зависимости загружены успешно')
except ImportError as e:
    print(f'⚠️  Ошибка импорта: {e}')
"

Write-Host ""
Write-Host "🎉 Установка завершена!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "Примечание: Некоторые предупреждения о конфликтах зависимостей нормальны" -ForegroundColor Yellow
Write-Host "и не должны мешать работе Whisper" -ForegroundColor Yellow
Write-Host ""
Write-Host "Для активации окружения используйте:" -ForegroundColor Yellow
Write-Host ".\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host ""
Write-Host "Для запуска транскрибации:" -ForegroundColor Yellow
Write-Host "python whisper_transcribe.py [папка_с_аудио] [модель] [выходная_папка]" -ForegroundColor Cyan
Write-Host ""
Write-Host "Примеры:" -ForegroundColor Yellow
Write-Host "python whisper_transcribe.py ./audio" -ForegroundColor Cyan
Write-Host "python whisper_transcribe.py ./audio large ./results" -ForegroundColor Cyan
Write-Host ""
Write-Host "Доступные модели (от быстрой к точной):" -ForegroundColor Yellow
Write-Host "tiny, base, small, medium, large" -ForegroundColor Cyan