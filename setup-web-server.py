#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🛠️ Установка и запуск веб-сервера для Whisper
"""

import os
import sys
import subprocess

def check_venv():
    """Проверка виртуального окружения"""
    venv_path = ".venv"
    if os.path.exists(venv_path):
        print("✅ Виртуальное окружение найдено")
        return True
    else:
        print("❌ Виртуальное окружение не найдено")
        return False

def install_dependencies():
    """Установка зависимостей Flask"""
    print("\n📦 Установка зависимостей...")
    
    # Определяем путь к pip
    if os.name == 'nt':  # Windows
        pip_path = ".venv\\Scripts\\pip.exe"
    else:  # Linux/Mac
        pip_path = ".venv/bin/pip"
    
    if os.path.exists(pip_path):
        dependencies = [
            "flask==2.3.3",
            "werkzeug==2.3.7", 
            "python-multipart==0.0.6"
        ]
        
        for dep in dependencies:
            print(f"Устанавливаю: {dep}")
            try:
                subprocess.run([pip_path, "install", dep], check=True)
                print(f"✅ {dep} установлен")
            except subprocess.CalledProcessError as e:
                print(f"❌ Ошибка установки {dep}: {e}")
    else:
        print("❌ pip не найден в виртуальном окружении")

def create_app_files():
    """Создание необходимых файлов для веб-сервера"""
    print("\n📁 Создание файлов веб-сервера...")
    
    # Создаем папку templates если её нет
    os.makedirs("templates", exist_ok=True)
    
    # Создаем app.py
    app_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 Веб-интерфейс для OpenAI Whisper транскрибации
"""

import os
import sys
import glob
import zipfile
import subprocess
import threading
import time
from pathlib import Path

# Исправление кодировки для Windows
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

try:
    from flask import Flask, render_template, request, send_file, jsonify, flash, redirect, url_for
    from werkzeug.utils import secure_filename
    print("✅ Flask импортирован успешно")
except ImportError as e:
    print(f"❌ Ошибка импорта Flask: {e}")
    print("Установите: pip install flask werkzeug python-multipart")
    sys.exit(1)

# Настройки приложения
app = Flask(__name__)
app.secret_key = 'whisper-web-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024

# Папки
UPLOAD_FOLDER = 'audio'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a'}

# Создаем папки если их нет
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Глобальная переменная для отслеживания статуса обработки
processing_status = {
    'is_processing': False,
    'start_time': None,
    'current_file': None
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_folders():
    """Очистка папок перед обработкой"""
    try:
        for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
            for item in os.listdir(folder):
                item_path = os.path.join(folder, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
        print("✅ Папки очищены")
        return True
    except Exception as e:
        print(f"❌ Ошибка очистки папок: {e}")
        return False

def create_zip_archive():
    """Создание ZIP архива с результатами"""
    try:
        result_files = []
        for ext in ['.txt', '.srt', '.vtt', '.json']:
            result_files.extend(glob.glob(os.path.join(RESULTS_FOLDER, f'*{ext}')))
        
        if not result_files:
            print("❌ Файлы результатов не найдены")
            return None
        
        zip_filename = "results.zip"
        zip_path = os.path.join(RESULTS_FOLDER, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for result_file in result_files:
                zipf.write(result_file, os.path.basename(result_file))
        
        print(f"✅ ZIP архив создан: {zip_filename}")
        return zip_filename
        
    except Exception as e:
        print(f"❌ Ошибка создания ZIP архива: {e}")
        return None

def process_audio():
    """Обработка всех аудиофайлов в папке"""
    global processing_status
    
    try:
        processing_status['is_processing'] = True
        processing_status['start_time'] = time.time()
        
        print("🚀 Запуск обработки всех файлов в папке audio...")
        
        cmd = [
            sys.executable, 'whisper_transcribe.py',
            UPLOAD_FOLDER,
            'large',
            RESULTS_FOLDER
        ]
        
        print(f"Выполняем команду: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd='.', encoding='utf-8')
        
        if result.returncode == 0:
            print("✅ Обработка завершена успешно")
            zip_filename = create_zip_archive()
            
            processing_status['is_processing'] = False
            processing_status['current_file'] = None
            
            if zip_filename:
                return True, "Обработка завершена успешно"
            else:
                return False, "Ошибка создания архива"
        else:
            print(f"❌ Ошибка обработки: {result.stderr}")
            processing_status['is_processing'] = False
            return False, f"Ошибка: {result.stderr}"
            
    except Exception as e:
        print(f"❌ Исключение при обработке: {e}")
        processing_status['is_processing'] = False
        return False, f"Исключение: {str(e)}"

@app.route('/')
def index():
    """Главная страница"""
    processed_files = []
    
    zip_files = glob.glob(os.path.join(RESULTS_FOLDER, '*.zip'))
    for zip_file in zip_files:
        processed_files.append({
            'name': os.path.splitext(os.path.basename(zip_file))[0],
            'zip_path': os.path.basename(zip_file),
            'size': os.path.getsize(zip_file)
        })
    
    if not processed_files:
        for result_file in glob.glob(os.path.join(RESULTS_FOLDER, '*')):
            if not result_file.endswith('.zip'):
                processed_files.append({
                    'name': os.path.splitext(os.path.basename(result_file))[0],
                    'zip_path': os.path.basename(result_file),
                    'size': os.path.getsize(result_file)
                })
    
    return render_template('index.html', 
                         processed_files=processed_files,
                         is_processing=processing_status['is_processing'])

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загрузки файлов"""
    global processing_status
    
    if 'files' not in request.files:
        flash('Файлы не выбраны', 'error')
        return redirect('/')
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        flash('Файлы не выбраны', 'error')
        return redirect('/')
    
    if processing_status['is_processing']:
        flash('Обработка уже выполняется. Дождитесь завершения.', 'warning')
        return redirect('/')
    
    if not cleanup_folders():
        flash('Ошибка очистки папки', 'error')
        return redirect('/')
    
    saved_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            saved_files.append(filename)
            print(f"📁 Файл сохранен: {filename}")
    
    if not saved_files:
        flash('Нет подходящих файлов для загрузки', 'error')
        return redirect('/')
    
    def process_task():
        success, message = process_audio()
        if success:
            print(f"✅ Обработка завершена: {message}")
        else:
            print(f"❌ Ошибка обработки: {message}")
    
    threading.Thread(target=process_task, daemon=True).start()
    
    flash(f'Загружено {len(saved_files)} файлов. Обработка начата. Это может занять несколько минут.', 'success')
    return redirect('/')

@app.route('/download/<filename>')
def download_file(filename):
    """Скачивание файла"""
    file_path = os.path.join(RESULTS_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    flash('Файл не найден', 'error')
    return redirect('/')

@app.route('/cleanup', methods=['POST'])
def cleanup_files():
    """Очистка файлов"""
    global processing_status
    
    if processing_status['is_processing']:
        flash('Нельзя очищать во время обработки', 'warning')
        return redirect('/')
    
    if cleanup_folders():
        flash('Все файлы очищены', 'success')
    else:
        flash('Ошибка при очистке файлов', 'error')
    return redirect('/')

@app.route('/status')
def status():
    """Статус обработки"""
    audio_files = len([f for f in os.listdir(UPLOAD_FOLDER) if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))])
    result_files = len([f for f in os.listdir(RESULTS_FOLDER) if os.path.isfile(os.path.join(RESULTS_FOLDER, f)) and not f.endswith('.zip')])
    zip_files = len([f for f in os.listdir(RESULTS_FOLDER) if f.endswith('.zip')])
    
    return jsonify({
        'audio_files': audio_files,
        'result_files': result_files,
        'zip_files': zip_files,
        'is_processing': processing_status['is_processing'],
        'processing_time': time.time() - processing_status['start_time'] if processing_status['start_time'] else 0,
        'ready': result_files > 0 or zip_files > 0
    })

if __name__ == '__main__':
    print("🌐 Запуск веб-сервера Whisper...")
    print("📁 Загрузки:", os.path.abspath(UPLOAD_FOLDER))
    print("📁 Результаты:", os.path.abspath(RESULTS_FOLDER))
    print("🚀 Сервер: http://localhost:5000")
    print("⏹️  Остановка: Ctrl+C")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
'''

    with open("app.py", "w", encoding="utf-8") as f:
        f.write(app_content)
    print("✅ app.py создан")

    # Создаем index.html
    index_content = '''<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Whisper Web Interface</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background: #f8f9fa; }
        .card { transition: transform 0.2s; }
        .card:hover { transform: translateY(-2px); }
        .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .processing { animation: pulse 2s infinite; }
        .file-list { max-height: 200px; overflow-y: auto; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.7; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header text-white py-4 mb-4">
        <div class="container">
            <h1 class="display-4">🎙️ Whisper Web Interface</h1>
            <p class="lead">Загрузите аудиофайлы для автоматической транскрибации</p>
            <small>Используется модель Large для максимальной точности</small>
        </div>
    </div>

    <div class="container">
        <div class="card shadow-lg mb-4">
            <div class="card-header bg-primary text-white">
                <h5 class="mb-0">📤 Загрузка аудиофайлов</h5>
            </div>
            <div class="card-body">
                <form action="/upload" method="post" enctype="multipart/form-data" id="uploadForm">
                    <div class="mb-3">
                        <label class="form-label">Аудиофайлы (можно несколько):</label>
                        <input class="form-control" type="file" name="files" accept=".mp3,.wav,.m4a" multiple required>
                        <div class="form-text">Поддерживаемые форматы: MP3, WAV, M4A (макс. 200MB всего)</div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary btn-lg w-100 mt-3" id="submitBtn">
                        🚀 Начать обработку
                    </button>
                </form>
            </div>
        </div>

        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                    <div class="alert alert-{{ 'danger' if category == 'error' else 'success' if category == 'success' else 'warning' }} alert-dismissible fade show">
                        {{ message }}
                        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                    </div>
                {% endfor %}
            {% endif %}
        {% endwith %}

        <div class="card shadow-lg mb-4" id="statusCard" style="display: none;">
            <div class="card-header bg-warning text-dark">
                <h5 class="mb-0">⏳ Обработка...</h5>
            </div>
            <div class="card-body text-center">
                <div class="spinner-border text-primary mb-3" role="status">
                    <span class="visually-hidden">Loading...</span>
                </div>
                <p id="statusText">Обработка началась. Это может занять несколько минут.</p>
                <div class="progress mb-2">
                    <div class="progress-bar progress-bar-striped progress-bar-animated" style="width: 100%"></div>
                </div>
            </div>
        </div>

        {% if processed_files %}
        <div class="card shadow-lg">
            <div class="card-header bg-success text-white">
                <h5 class="mb-0">✅ Готовые файлы</h5>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-hover">
                        <thead>
                            <tr>
                                <th>Имя файла</th>
                                <th>Размер</th>
                                <th>Тип</th>
                                <th>Действия</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for file in processed_files %}
                            <tr>
                                <td>{{ file.name }}</td>
                                <td>{{ (file.size / 1024)|round(2) }} KB</td>
                                <td>
                                    {% if file.zip_path.endswith('.zip') %}
                                    <span class="badge bg-primary">ZIP архив</span>
                                    {% else %}
                                    <span class="badge bg-secondary">Текстовый файл</span>
                                    {% endif %}
                                </td>
                                <td>
                                    <a href="/download/{{ file.zip_path }}" class="btn btn-success btn-sm">
                                        📥 Скачать
                                    </a>
                                </td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
        {% endif %}

        <div class="text-center mt-4">
            <form action="/cleanup" method="post" onsubmit="return confirm('Удалить ВСЕ файлы?')">
                <button type="submit" class="btn btn-outline-danger">
                    🗑️ Очистить все файлы
                </button>
            </form>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        let checkInterval;
        
        async function checkProcessingStatus() {
            try {
                const response = await fetch('/status');
                                const data = await response.json();
                
                console.log('Статус обработки:', data);
                
                if (data.is_processing) {
                    const minutes = Math.floor(data.processing_time / 60);
                    const seconds = Math.floor(data.processing_time % 60);
                    document.getElementById('statusText').textContent = 
                        `Обработка выполняется: ${minutes}м ${seconds}с`;
                }
                
                if (!data.is_processing && data.ready) {
                    clearInterval(checkInterval);
                    console.log('Обработка завершена, обновляем страницу...');
                    location.reload();
                }
                
                if (!data.is_processing && !data.ready && data.processing_time > 120) {
                    clearInterval(checkInterval);
                    console.log('Возможно ошибка обработки, обновляем страницу...');
                    location.reload();
                }
                
            } catch (error) {
                console.log('Ошибка проверки статуса:', error);
            }
        }

        document.getElementById('uploadForm').addEventListener('submit', function() {
            document.getElementById('statusCard').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            
            checkInterval = setInterval(checkProcessingStatus, 3000);
            setTimeout(checkProcessingStatus, 1000);
        });

        document.addEventListener('DOMContentLoaded', function() {
            {% if is_processing %}
            document.getElementById('statusCard').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            
            checkInterval = setInterval(checkProcessingStatus, 3000);
            checkProcessingStatus();
            {% endif %}
        });

        setInterval(() => {
            if (document.getElementById('statusCard').style.display === 'block') {
                checkProcessingStatus();
            }
        }, 30000);

        document.querySelector('input[type="file"]').addEventListener('change', function(e) {
            const files = e.target.files;
            let totalSize = 0;
            
            for (let i = 0; i < files.length; i++) {
                totalSize += files[i].size;
            }
            
            if (totalSize > 100 * 1024 * 1024) {
                if (!confirm(`Общий размер файлов: ${(totalSize / 1024 / 1024).toFixed(1)}MB. Это может занять много времени. Продолжить?`)) {
                    e.target.value = '';
                }
            }
            
            if (files.length > 0) {
                let fileList = 'Выбрано файлов: ' + files.length;
                if (files.length <= 5) {
                    fileList += ' (';
                    for (let i = 0; i < files.length; i++) {
                        if (i > 0) fileList += ', ';
                        fileList += files[i].name;
                    }
                    fileList += ')';
                }
                console.log(fileList);
            }
        });
    </script>
</body>
</html>'''

    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(index_content)
    print("✅ templates/index.html создан")

def start_server():
    """Запуск веб-сервера"""
    print("\n🚀 Запуск веб-сервера...")
    
    if os.name == 'nt':  # Windows
        python_path = ".venv\\Scripts\\python.exe"
    else:  # Linux/Mac
        python_path = ".venv/bin/python"
    
    if os.path.exists(python_path):
        print("🌐 Сервер запускается...")
        os.system(f'"{python_path}" app.py')
    else:
        print("❌ Python в виртуальном окружении не найден")
        print("Запустите: python app.py")

def main():
    """Основная функция"""
    print("🛠️  УСТАНОВКА И ЗАПУСК WEB-СЕРВЕРА WHISPER")
    print("="*60)
    
    if not check_venv():
        print("❌ Сначала создайте виртуальное окружение:")
        print("python -m venv .venv")
        print(".venv\\Scripts\\activate")
        return
    
    install_dependencies()
    create_app_files()
    start_server()

if __name__ == "__main__":
    main()
