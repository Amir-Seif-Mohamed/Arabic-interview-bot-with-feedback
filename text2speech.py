from klaam import TextToSpeech
from pyarabic import araby
from pyarabic.number import vocalize_number
import winsound
import mishkal.tashkeel


root_path = "C:/Users/LEGION/Desktop/gpt3-interview-bot-main/"
prepare_tts_model_path = "cfgs/FastSpeech2/config/Arabic/preprocess.yaml"
model_config_path = "cfgs/FastSpeech2/config/Arabic/model.yaml"
train_config_path = "cfgs/FastSpeech2/config/Arabic/train.yaml"
vocoder_config_path = "cfgs/FastSpeech2/model_config/hifigan/config.json"
speaker_pre_trained_path = "data/model_weights/hifigan/generator_universal.pth.tar"

text2speech = TextToSpeech(prepare_tts_model_path, model_config_path, train_config_path, vocoder_config_path, speaker_pre_trained_path,root_path)
def text_speech(text:str):
    vocalizer = mishkal.tashkeel.TashkeelClass()
    txt = vocalizer.tashkeel(text)
    wordlist = araby.tokenize(txt)
    vocalized =  vocalize_number(wordlist)
    print(vocalized)
    text2speech.synthesize(u" ".join(vocalized))
    winsound.PlaySound("sample.wav", winsound.SND_FILENAME)
