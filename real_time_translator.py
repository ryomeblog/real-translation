import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
import queue
import threading
import os
import time
import tempfile
import hashlib
from dotenv import load_dotenv
from translate import Translator
from gtts import gTTS
from concurrent.futures import ThreadPoolExecutor

# 環境変数の読み込み
load_dotenv()

class RealTimeTranslator:
    def __init__(self, model_size="base", source_lang="en", target_lang="ja", use_gpu=False):
        # デバイスとcompute_typeの設定
        if use_gpu:
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"
        
        # Whisperモデルの初期化
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type
        )
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        
        # 音声バッファの設定
        self.audio_queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        
        # 録音パラメータ
        self.sample_rate = 16000
        self.block_size = 4000
        self.input_device = None
        self.output_device = None
        
        # 状態管理
        self.is_recording = False
        self.buffer_ready = threading.Event()
        
        # スレッド管理
        self.stream = None
        self.stream_thread = None
        self.process_thread = None
        self.tts_thread = None
        self.tts_queue = queue.Queue()
        
        # 同期オブジェクト
        self.lock = threading.Lock()
        
        # 音声キャッシュの設定
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.temp_dir, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Translatorの初期化
        self.translators = [
            Translator(to_lang=target_lang, from_lang=source_lang, provider='mymemory'),
            Translator(to_lang=target_lang, from_lang=source_lang, provider='libre')
        ]
        self.current_translator_index = 0
        print("翻訳エンジンを初期化しました")

    def find_stereo_mix_device(self):
        """ステレオミキサーまたは同等のデバイスを検索"""
        devices = sd.query_devices()
        stereo_mix_keywords = ['stereo mix', 'ステレオ ミキサー', 'ステレオミキサー', 'what u hear',
                             'wave out', 'アナログミックス', 'analog mix', 'オーディオ出力']
        
        print("\n利用可能な入力デバイス:")
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")
                # ステレオミキサーっぽいデバイスを探す
                if any(keyword.lower() in device['name'].lower() for keyword in stereo_mix_keywords):
                    print(f"推奨入力デバイスが見つかりました: {device['name']}")
                    return i
        return None

    def select_audio_devices(self):
        """入力デバイスと出力デバイスの選択"""
        devices = sd.query_devices()
        
        # 入力デバイスの選択
        stereo_mix_id = self.find_stereo_mix_device()
        
        if stereo_mix_id is not None:
            print(f"\nステレオミキサーを使用します: {devices[stereo_mix_id]['name']}")
            self.input_device = stereo_mix_id
        else:
            print("\n警告: ステレオミキサーが見つかりません。手動で選択してください。")
            while True:
                try:
                    device_id = input("使用する入力デバイスの番号を入力してください: ")
                    device_id = int(device_id)
                    if 0 <= device_id < len(devices) and devices[device_id]['max_input_channels'] > 0:
                        print(f"\n入力デバイス: {devices[device_id]['name']}")
                        self.input_device = device_id
                        break
                    else:
                        print("無効なデバイス番号です。")
                except ValueError:
                    print("数値を入力してください。")
                except Exception as e:
                    print(f"エラー: {e}")
        
        # 出力デバイスの選択
        print("\n利用可能な出力デバイス:")
        for i, device in enumerate(devices):
            if device['max_output_channels'] > 0:
                print(f"{i}: {device['name']}")
        
        while True:
            try:
                device_id = input("\n使用する出力デバイスの番号を入力してください: ")
                device_id = int(device_id)
                if 0 <= device_id < len(devices) and devices[device_id]['max_output_channels'] > 0:
                    print(f"出力デバイス: {devices[device_id]['name']}")
                    self.output_device = device_id
                    break
                else:
                    print("無効なデバイス番号です。")
            except ValueError:
                print("数値を入力してください。")
            except Exception as e:
                print(f"エラー: {e}")

    def translate_text(self, text):
        """複数の翻訳プロバイダーを試行して翻訳"""
        for attempt in range(len(self.translators)):
            try:
                translator = self.translators[self.current_translator_index]
                result = translator.translate(text)
                
                # 翻訳結果が英語の場合は次のプロバイダーを試す
                if any(char in result for char in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'):
                    raise Exception("Translation not in Japanese")
                
                return result
            except Exception as e:
                print(f"翻訳エラー (プロバイダー {self.current_translator_index}): {e}")
                # 次のプロバイダーに切り替え
                self.current_translator_index = (self.current_translator_index + 1) % len(self.translators)
        
        return f"翻訳エラー: {text}"  # すべてのプロバイダーが失敗した場合

    def audio_callback(self, indata, frames, time, status):
        """音声データを受け取るコールバック関数"""
        if status:
            print(status)
        if self.is_recording:
            self.audio_queue.put(indata.copy())

    def generate_speech(self, text):
        """音声ファイルを生成"""
        try:
            # テキストから一意のファイル名を生成
            filename = os.path.join(self.cache_dir, f"{hashlib.md5(text.encode()).hexdigest()}.wav")
            
            # キャッシュをチェック
            if os.path.exists(filename):
                return filename
            
            # 音声ファイルを生成
            tts = gTTS(text=text, lang='ja')
            mp3_filename = filename.replace('.wav', '.mp3')
            tts.save(mp3_filename)
            
            # MP3をWAVに変換（sounddeviceで再生するため）
            data, sr = sf.read(mp3_filename)
            sf.write(filename, data, sr)
            
            # MP3ファイルを削除
            os.remove(mp3_filename)
            
            return filename
        except Exception as e:
            print(f"音声生成エラー: {e}")
            return None

    def play_audio(self, filename):
        """音声ファイルを再生"""
        try:
            data, sr = sf.read(filename)
            sd.play(data, sr, device=self.output_device)
            sd.wait()
        except Exception as e:
            print(f"音声再生エラー: {e}")

    def tts_worker(self):
        """音声合成ワーカースレッド"""
        while self.is_recording:
            try:
                text = self.tts_queue.get(timeout=1)
                if text is not None:
                    # 音声ファイルを生成
                    audio_file = self.generate_speech(text)
                    if audio_file:
                        self.play_audio(audio_file)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"音声合成エラー: {e}")

    def process_audio(self):
        """音声認識と翻訳を実行するメインループ"""
        while self.is_recording:
            try:
                # バッファが準備できるまで待機
                if not self.buffer_ready.wait(timeout=1):
                    continue

                with self.lock:
                    if len(self.buffer) < self.sample_rate:  # 1秒未満のデータは処理しない
                        continue

                    # 音声認識を実行
                    segments, _ = self.model.transcribe(
                        self.buffer,
                        language=self.source_lang,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500)
                    )

                    # バッファをクリア
                    self.buffer = np.array([], dtype=np.float32)
                    self.buffer_ready.clear()

                # 認識結果を処理
                for segment in segments:
                    recognized_text = segment.text.strip()
                    if recognized_text:
                        print(f"\n認識テキスト: {recognized_text}")
                        
                        # テキストを翻訳
                        translated_text = self.translate_text(recognized_text)
                        if translated_text:
                            print(f"翻訳結果: {translated_text}")
                            # 翻訳結果を音声合成キューに追加
                            self.tts_queue.put(translated_text)

            except Exception as e:
                print(f"音声処理エラー: {e}")

    def stream_audio(self):
        """音声ストリーム処理"""
        while self.is_recording:
            try:
                # キューから音声データを取得
                audio_data = self.audio_queue.get(timeout=1)
                
                with self.lock:
                    # バッファに追加
                    self.buffer = np.append(self.buffer, audio_data.flatten())
                    
                    # バッファが2秒分たまったらフラグを設定
                    if len(self.buffer) >= self.sample_rate * 2:
                        self.buffer_ready.set()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"ストリーム処理エラー: {e}")

    def start_translation(self):
        """翻訳処理を開始"""
        self.is_recording = True
        
        # デバイスの選択
        self.select_audio_devices()

        # スレッドの開始
        self.process_thread = threading.Thread(target=self.process_audio)
        self.stream_thread = threading.Thread(target=self.stream_audio)
        self.tts_thread = threading.Thread(target=self.tts_worker)
        
        self.process_thread.start()
        self.stream_thread.start()
        self.tts_thread.start()

        # 音声入力ストリームを開始
        with sd.InputStream(
            device=self.input_device,
            callback=self.audio_callback,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            dtype=np.float32
        ):
            print("\nリアルタイム翻訳を開始しました。Ctrl+Cで終了します。")
            try:
                while self.is_recording:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                print("\n終了します...")
            finally:
                self.stop_translation()

    def stop_translation(self):
        """翻訳処理を停止"""
        self.is_recording = False
        
        # スレッドの終了を待機
        for thread in [self.process_thread, self.stream_thread, self.tts_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=2)
        
        # Executorのシャットダウンとクリーンアップ
        self.executor.shutdown(wait=False)
        
        # キャッシュディレクトリの削除
        try:
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"クリーンアップエラー: {e}")
        
        print("翻訳を終了しました。")

def main():
    # トランスレーターの初期化（英語→日本語）
    translator = RealTimeTranslator(
        model_size="base",
        source_lang="en",
        target_lang="ja",
        use_gpu=False  # GPUを使用する場合はTrueに設定
    )

    try:
        translator.start_translation()
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()