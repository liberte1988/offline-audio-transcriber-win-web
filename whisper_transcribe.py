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
Версия: 2.0

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

# Добавьте эти импорты после существующих
import librosa
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    """Поиск аудиофайлов в директории без дубликатов"""
    audio_extensions = ['.wav', '.mp3', '.m4a']
    files = []
    
    # Проверяем все файлы в директории
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_ext = os.path.splitext(filename)[1].lower()
            if file_ext in audio_extensions:
                files.append(file_path)
    
    return sorted(files)

def transcribe_audio(model, file_path, device="cpu", language="ru"):
    """Распознавание одного аудиофайла"""
    print(f"🎵 Обрабатываю: {os.path.basename(file_path)}")
    
    try:
        start_time = time.time()
        
        result = model.transcribe(
            file_path, 
            language=language,
            verbose=False,
            fp16=device == "cuda"
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

def get_audio_duration(file_path):
    """Получить длительность аудиофайла"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = len(y) / sr
        return duration
    except:
        return 0

def fast_kmeans_diarization(audio_path, n_speakers=2, min_segment_duration=1.0):
    """
    Быстрая диаризация на основе K-Means и MFCC
    """
    try:
        print("🔊 Быстрая диаризация (K-Means)...")
        
        # Загрузка аудио
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
        
        # Извлечение MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512)
        mfccs = mfccs.T  # Транспонируем для кластеризации по времени
        
        # Нормализация
        scaler = StandardScaler()
        X = scaler.fit_transform(mfccs)
        
        # K-Means кластеризация (быстрее чем Spectral)
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Создание сегментов
        hop_duration = 512 / sr  # Длительность одного фрейма в секундах
        segments = []
        current_speaker = None
        current_start = 0
        
        for i, label in enumerate(labels):
            current_time = i * hop_duration
            
            if current_speaker is None:
                current_speaker = f"spk_{label + 1}"
                current_start = current_time
            elif f"spk_{label + 1}" != current_speaker:
                segment_duration = current_time - current_start
                if segment_duration >= min_segment_duration:
                    segments.append({
                        'speaker': current_speaker,
                        'start': current_start,
                        'end': current_time,
                        'duration': segment_duration
                    })
                current_speaker = f"spk_{label + 1}"
                current_start = current_time
        
        # Последний сегмент
        if current_speaker is not None:
            segment_duration = duration - current_start
            if segment_duration >= min_segment_duration:
                segments.append({
                    'speaker': current_speaker,
                    'start': current_start,
                    'end': duration,
                    'duration': segment_duration
                })
        
        print(f"✅ Найдено {len(segments)} сегментов за {duration:.1f}с аудио")
        return segments
        
    except Exception as e:
        print(f"❌ Ошибка быстрой диаризации: {e}")
        return []

def energy_pitch_diarization(audio_path, n_speakers=2, min_segment_duration=1.0):
    """
    Диаризация на основе энергии и высоты тона (очень быстрая)
    """
    try:
        print("🔊 Диаризация по энергии и тону...")
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Детекция речи по энергии
        energy = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
        energy_threshold = np.percentile(energy, 25)
        speech_frames = energy > energy_threshold
        
        # Извлечение высоты тона только для речевых фреймов
        pitches = []
        times = []
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech:
                start = i * 512
                end = start + 2048
                if end < len(y):
                    frame = y[start:end]
                    # Быстрая оценка высоты тона
                    f0 = librosa.yin(frame, fmin=80, fmax=400, sr=sr)
                    if not np.isnan(f0[0]):
                        pitches.append(f0[0])
                        times.append(i * 512 / sr)
        
        if len(pitches) < 10:
            return []
        
        # Кластеризация по высоте тона
        pitches = np.array(pitches).reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=5)
        pitch_labels = kmeans.fit_predict(pitches)
        
        # Создание сегментов
        segments = []
        current_speaker = None
        current_start = 0
        
        for i, (time, label) in enumerate(zip(times, pitch_labels)):
            speaker_id = f"spk_{label + 1}"
            
            if current_speaker is None:
                current_speaker = speaker_id
                current_start = time
            elif speaker_id != current_speaker:
                if time - current_start >= min_segment_duration:
                    segments.append({
                        'speaker': current_speaker,
                        'start': current_start,
                        'end': time,
                        'duration': time - current_start
                    })
                current_speaker = speaker_id
                current_start = time
        
        print(f"✅ Диаризация по тону: {len(segments)} сегментов")
        return segments
        
    except Exception as e:
        print(f"❌ Ошибка диаризации по тону: {e}")
        return []

def smart_diarization(audio_path, n_speakers=2, method="auto"):
    """
    Умный выбор метода диаризации в зависимости от ситуации
    """
    # Определяем длительность аудио
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr
    
    if method == "auto":
        if duration > 300:  # > 5 минут - используем самый быстрый метод
            return energy_pitch_diarization(audio_path, n_speakers)
        elif duration > 60:  # 1-5 минут - быстрый K-Means
            return fast_kmeans_diarization(audio_path, n_speakers)
        else:  # < 1 минуты - точный метод
            return fast_kmeans_diarization(audio_path, n_speakers)  # Используем K-Means для точности
    elif method == "fast":
        return energy_pitch_diarization(audio_path, n_speakers)
    elif method == "balanced":
        return fast_kmeans_diarization(audio_path, n_speakers)
    else:
        return fast_kmeans_diarization(audio_path, n_speakers)

def transcribe_with_diarization(model, audio_path, output_dir, language="ru", n_speakers=2):
    """
    Транскрибация с диаризацией
    """
    try:
        print(f"🎤 Диаризация для {n_speakers} говорящих...")
        
        # Выбор метода диаризации
        segments = smart_diarization(audio_path, n_speakers, "balanced")
        
        if not segments:
            print("⚠️  Диаризация не дала результатов")
            return None
        
        # Загружаем аудио для транскрибации
        y, sr = librosa.load(audio_path, sr=16000)
        
        results = []
        total_segments = len(segments)
        
        for i, segment in enumerate(segments, 1):
            print(f"🔄 Обработка сегмента {i}/{total_segments}...", end="", flush=True)
            
            start_sample = int(segment['start'] * sr)
            end_sample = int(segment['end'] * sr)
            
            # Проверяем границы
            if start_sample >= len(y) or end_sample > len(y):
                continue
                
            segment_audio = y[start_sample:end_sample]
            
            if len(segment_audio) == 0:
                continue
            
            # Сохраняем временный файл для транскрибации
            temp_path = os.path.join(output_dir, f"temp_segment_{i}.wav")
            sf.write(temp_path, segment_audio, sr)
            
            # Транскрибируем сегмент
            try:
                result = model.transcribe(
                    temp_path,
                    language=language,
                    verbose=False,
                    fp16=torch.cuda.is_available()
                )
                
                results.append({
                    'speaker': segment['speaker'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': result['text'].strip(),
                    'segments': result.get('segments', [])
                })
                
                print("✅")
                
            except Exception as e:
                print(f"❌: {e}")
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        return results
        
    except Exception as e:
        print(f"❌ Ошибка в transcribe_with_diarization: {e}")
        return None

def save_diarized_results(results, output_dir, base_name):
    """
    Сохранение результатов с диаризацией
    """
    if not results:
        return None, None, None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Текстовый файл с метками говорящих
    txt_path = os.path.join(output_dir, f"{base_name}_diarized.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Результаты диаризации для {base_name}\n")
        f.write("=" * 50 + "\n\n")
        
        for result in results:
            start_min = int(result['start'] // 60)
            start_sec = int(result['start'] % 60)
            end_min = int(result['end'] // 60)
            end_sec = int(result['end'] % 60)
            
            f.write(f"[{result['speaker']}] {start_min:02d}:{start_sec:02d} - {end_min:02d}:{end_sec:02d}\n")
            f.write(f"{result['text']}\n\n")
    
    # 2. JSON с детальной информацией
    json_path = os.path.join(output_dir, f"{base_name}_diarized.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 3. SRT с метками говорящих
    srt_path = os.path.join(output_dir, f"{base_name}_diarized.srt")
    with open(srt_path, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results, 1):
            start = format_timestamp(result['start'])
            end = format_timestamp(result['end'])
            speaker = result['speaker']
            text = result['text']
            
            f.write(f"{i}\n{start} --> {end}\n[{speaker}] {text}\n\n")
    
    print(f"💾 Файлы диаризации сохранены: {base_name}_diarized.*")
    return txt_path, json_path, srt_path

def save_single_result(result, output_dir, enable_diarization=False, model=None, n_speakers=2):
    """
    Сохранение результата одного файла сразу после обработки
    """
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
    
    # Диаризация (если включена и есть модель)
    if enable_diarization and model:
        try:
            diarized_results = transcribe_with_diarization(
                model, result['file'], output_dir, language="ru", n_speakers=n_speakers
            )
            if diarized_results:
                save_diarized_results(diarized_results, output_dir, base_name)
        except Exception as e:
            print(f"⚠️  Ошибка диаризации для {base_name}: {e}")
    
    print(f"💾 Файл сохранен: {base_name}.txt, {base_name}.srt")

def format_timestamp(seconds):
    """Форматирование времени для SRT"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def save_final_json(results, output_dir):
    """Сохранение финального JSON файла со всеми результатами"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Сохранение JSON с детальной информацией
    json_path = os.path.join(output_dir, "transcripts_detailed.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

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
    
    # Параметры по умолчанию
    input_directory = "."  # Текущая директория
    output_directory = "transcripts"
    model_size = "large"  # tiny, base, small, medium, large
    language = "ru"  # Русский язык
    enable_diarization = False  # Диаризация по умолчанию выключена
    n_speakers = 2  # Количество говорящих по умолчанию
    
    # Получение параметров из аргументов командной строки
    if len(sys.argv) > 1:
        input_directory = sys.argv[1]
    if len(sys.argv) > 2:
        model_size = sys.argv[2]
    if len(sys.argv) > 3:
        output_directory = sys.argv[3]
    if len(sys.argv) > 4:
        enable_diarization = sys.argv[4].lower() in ['true', '1', 'yes', 'y']
    if len(sys.argv) > 5:
        try:
            n_speakers = int(sys.argv[5])
        except ValueError:
            print("⚠️  Неверное количество говорящих, используется значение по умолчанию: 2")
            n_speakers = 2
    
    print(f"📁 Директория с аудио: {input_directory}")
    print(f"🎯 Модель: {model_size}")
    print(f"💾 Выходная директория: {output_directory}")
    print(f"🌍 Язык: {language}")
    print(f"🎤 Диаризация: {'ВКЛЮЧЕНА' if enable_diarization else 'ВЫКЛЮЧЕНА'}")
    if enable_diarization:
        print(f"👥 Количество говорящих: {n_speakers}")
    print()
    
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
        duration = get_audio_duration(file)
        print(f"  - {os.path.basename(file)} ({size_mb:.1f} MB, {duration:.1f} сек)")
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
            # Сохраняем результат сразу после обработки с диаризацией
            save_single_result(result, output_directory, enable_diarization, model, n_speakers)
            
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
    
    if enable_diarization:
        print(f"  - [имя_файла]_diarized.* (файлы с диаризацией)")

if __name__ == "__main__":
    # Справка по использованию
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("Использование:")
        print("  python whisper_transcribe.py [директория] [модель] [выходная_папка] [диаризация] [говорящие]")
        print("\nПримеры:")
        print("  python whisper_transcribe.py")
        print("  python whisper_transcribe.py ./audio")
        print("  python whisper_transcribe.py ./audio large ./results")
        print("  python whisper_transcribe.py ./audio large ./results true")
        print("  python whisper_transcribe.py ./audio large ./results true 3")
        print("\nМодели: tiny, base, small, medium, large")
        print("Диаризация: true/false (по умолчанию false)")
        print("Говорящие: количество говорящих (по умолчанию 2)")
        print("Чем больше модель, тем точнее, но медленнее")
        sys.exit(0)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Прервано пользователем")
    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
