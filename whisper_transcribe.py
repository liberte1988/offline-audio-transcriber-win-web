#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎙️ Массовое распознавание аудио с помощью OpenAI Whisper 🎙️

Этот Python-скрипт предназначен для пакетной обработки аудиофайлов (mp3, wav, m4a)
в указанной директории, используя модель OpenAI Whisper для транскрибации речи.
Скрипт оптимизирован для работы на GPU NVIDIA для значительного ускорения.

Напоминание: Для максимальной производительности убедитесь, что у вас установлены
совместимые драйверы NVIDIA, CUDA и PyTorch с поддержкой CUDA.

Основные задачи:
- Автоматическое определение и использование GPU, если он доступен.
- Поиск всех поддерживаемых аудиофайлов в заданной директории.
- Последовательная обработка каждого файла с отображением прогресса.
- Сохранение результатов в нескольких форматах для удобства:
  - .txt: чистый текст для каждого файла.
  - .srt: файл субтитров с временными метками.
  - all_transcripts.txt: общий файл со всеми текстами.
- Вывод итоговой статистики по окончании работы.

Порядок использования:
1. Активируйте виртуальное окружение: source .venv/bin/activate
2. Запустите скрипт, указав параметры в командной строке:
   python whisper_transcribe.py <путь_к_аудио> <модель> <папка_результатов>
3. Если параметры не указаны, будут использованы значения по умолчанию.

Автор: Михаил Шардин https://shardin.name/
Дата создания: 29.08.2025
Версия: 1.0

Актуальная версия скрипта всегда здесь: https://github.com/empenoso/offline-audio-transcriber

"""

import os
import sys
import glob
import json
import time
from pathlib import Path
import whisper
import torch
import io

# Устанавливаем UTF-8 кодировку для вывода
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


def check_gpu():
    """Проверка доступности CUDA и GPU с тестированием совместимости"""
    if not torch.cuda.is_available():
        print("❌ CUDA недоступна. Будет использоваться CPU")
        return False
    
    try:
        gpu_name = torch.cuda.get_device_name(0)
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"🎮 Найден GPU: {gpu_name}")
        print(f"💾 Память GPU: {memory_gb:.1f} GB")
        
        # Тест совместимости GPU - создаем небольшой тензор
        test_tensor = torch.zeros(10, 10).cuda()
        _ = test_tensor + 1  # Простая операция
        test_tensor = test_tensor.cpu()  # Освобождаем память
        del test_tensor
        torch.cuda.empty_cache()
        
        print("✅ GPU совместим с PyTorch")
        return True
        
    except Exception as e:
        print(f"⚠️  GPU найден, но несовместим с текущим PyTorch: {str(e)}")
        print("🔄 Переключение на CPU режим")
        return False

def load_whisper_model(model_size="medium", use_gpu=True):
    """Загрузка модели Whisper с обработкой ошибок GPU"""
    print(f"🔄 Загрузка модели Whisper ({model_size})...")
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    try:
        model = whisper.load_model(model_size, device=device)
        print(f"✅ Модель загружена на {device}")
        return model, device
    except Exception as e:
        if device == "cuda":
            print(f"⚠️  Ошибка загрузки на GPU: {str(e)}")
            print("🔄 Переключение на CPU...")
            model = whisper.load_model(model_size, device="cpu")
            print(f"✅ Модель загружена на CPU")
            return model, "cpu"
        else:
            raise e

def get_audio_files(directory):
    """Поиск аудиофайлов в директории"""
    audio_extensions = ['*.wav', '*.mp3', '*.m4a', '*.WAV', '*.MP3', '*.M4A']
    files = []
    
    for ext in audio_extensions:
        files.extend(glob.glob(os.path.join(directory, ext)))
    
    return sorted(files)

def transcribe_audio(model, file_path, device="cpu", language="ru"):
    """Распознавание одного аудиофайла"""
    print(f"🎵 Обрабатываю: {os.path.basename(file_path)}")
    
    try:
        # Засекаем время обработки
        start_time = time.time()
        
        # Распознавание с указанием языка
        result = model.transcribe(
            file_path, 
            language=language,
            verbose=False,
            fp16=device == "cuda"  # Используем fp16 только для GPU
        )
        
        processing_time = time.time() - start_time
        
        # Извлекаем текст и сегменты
        text = result["text"].strip()
        segments = result.get("segments", [])
        
        print(f"✅ Готово за {processing_time:.1f}с")
        
        return {
            "file": file_path,
            "text": text,
            "segments": segments,
            "language": result.get("language", language),
            "processing_time": processing_time
        }
        
    except Exception as e:
        print(f"❌ Ошибка при обработке {file_path}: {e}")
        return None

def save_single_result(result, output_dir):
    """Сохранение результата одного файла сразу после обработки"""
    if not result:
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(result['file']))[0]
    
    # Текстовый файл
    individual_txt = os.path.join(output_dir, f"{base_name}.txt")
    with open(individual_txt, 'w', encoding='utf-8') as f:
        f.write(result['text'])
    
    # SRT субтитры (если есть сегменты)
    if result['segments']:
        srt_path = os.path.join(output_dir, f"{base_name}.srt")
        with open(srt_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(result['segments'], 1):
                start = format_timestamp(segment['start'])
                end = format_timestamp(segment['end'])
                text = segment['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    
    # Добавляем в общий файл
    all_txt_path = os.path.join(output_dir, "all_transcripts.txt")
    with open(all_txt_path, 'a', encoding='utf-8') as f:
        f.write(f"=== {os.path.basename(result['file'])} ===\n")
        f.write(f"{result['text']}\n\n")
    
    print(f"💾 Файл сохранен: {base_name}.txt, {base_name}.srt")

def save_final_json(results, output_dir):
    """Сохранение финального JSON файла со всеми результатами"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохранение JSON с детальной информацией
    json_path = os.path.join(output_dir, "transcripts_detailed.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def format_timestamp(seconds):
    """Форматирование времени для SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def print_statistics(results):
    """Вывод статистики обработки"""
    successful = [r for r in results if r is not None]
    failed = len(results) - len(successful)
    
    if successful:
        total_time = sum(r['processing_time'] for r in successful)
        avg_time = total_time / len(successful)
        total_text = sum(len(r['text']) for r in successful)
        
        print(f"\n📊 Статистика:")
        print(f"✅ Успешно обработано: {len(successful)} файлов")
        print(f"❌ Ошибок: {failed}")
        print(f"⏱️  Общее время: {total_time:.1f}с")
        print(f"⚡ Среднее время на файл: {avg_time:.1f}с")
        print(f"📝 Всего символов распознано: {total_text}")

def main():
    """Основная функция"""
    print("🎙️  Скрипт распознавания русской речи с OpenAI Whisper\n")
    
    # Параметры (можно изменить)
    input_directory = "."  # Текущая директория
    output_directory = "transcripts"
    model_size = "large"  # tiny, base, small, medium, large
    language = "ru"  # Русский язык
    
    # Получение параметров из аргументов командной строки
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
    if len(sys.argv) > 2:
        model_size = sys.argv[2]
    if len(sys.argv) > 3:
        output_directory = sys.argv[3]
    
    print(f"📁 Директория с аудио: {input_directory}")
    print(f"🎯 Модель: {model_size}")
    print(f"💾 Выходная директория: {output_directory}")
    print(f"🌍 Язык: {language}\n")
    
    # Проверка GPU
    use_gpu = check_gpu()
    print()
    
    # Поиск аудиофайлов
    audio_files = get_audio_files(input_directory)
    
    if not audio_files:
        print(f"❌ Аудиофайлы не найдены в {input_directory}")
        print("Поддерживаемые форматы: wav, mp3, m4a")
        return
    
    print(f"🎵 Найдено {len(audio_files)} аудиофайлов:")
    for file in audio_files:
        size_mb = os.path.getsize(file) / (1024 * 1024)
        print(f"  - {os.path.basename(file)} ({size_mb:.1f} MB)")
    print()
    
    # Загрузка модели
    model, actual_device = load_whisper_model(model_size, use_gpu)
    print()
    
    # Создаем выходную директорию и очищаем общий файл
    os.makedirs(output_directory, exist_ok=True)
    
    # Очищаем общий файл в начале
    all_txt_path = os.path.join(output_directory, "all_transcripts.txt")
    with open(all_txt_path, 'w', encoding='utf-8') as f:
        f.write("")  # Очищаем файл
    
    # Обработка файлов с немедленным сохранением
    results = []
    total_files = len(audio_files)
    
    print(f"🚀 Начинаю обработку {total_files} файлов на {actual_device.upper()}...\n")
    
    for i, file_path in enumerate(audio_files, 1):
        print(f"[{i}/{total_files}] ", end="")
        result = transcribe_audio(model, file_path, actual_device, language)
        
        if result:
            results.append(result)
            # Сохраняем результат сразу после обработки
            save_single_result(result, output_directory)
            
            # Показываем превью текста
            if result['text']:
                preview = result['text'][:100] + "..." if len(result['text']) > 100 else result['text']
                print(f"📝 Превью: {preview}")
        else:
            results.append(None)
        print()
    
    # Сохранение финального JSON файла
    print("💾 Сохраняю итоговый JSON...")
    save_final_json(results, output_directory)
    
    # Статистика
    print_statistics(results)
    
    print(f"\n🎉 Готово! Результаты сохранены в {output_directory}/")
    print(f"📄 Файлы:")
    print(f"  - all_transcripts.txt (весь текст)")
    print(f"  - transcripts_detailed.json (JSON с деталями)")
    print(f"  - [имя_файла].txt (отдельные текстовые файлы)")
    print(f"  - [имя_файла].srt (субтитры)")

if __name__ == "__main__":
    # Справка по использованию
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Использование:")
        print("  python whisper_transcribe.py [директория] [модель] [выходная_папка]")
        print("\nПримеры:")
        print("  python whisper_transcribe.py")
        print("  python whisper_transcribe.py ./audio")
        print("  python whisper_transcribe.py ./audio large ./results")
        print("\nМодели: tiny, base, small, medium, large")
        print("Чем больше модель, тем точнее, но медленнее")
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        sys.exit(1)