
ollama show jinyong:latest --modelfile

# 先删除旧的
ollama rm jinyong:latest

# 用正确Modelfile重建
ollama create jinyong -f ./models/Modelfile

# 验证
ollama show jinyong --modelfile

ollama run jinyong "写一个峨眉派弟子以金庸风格与蒙古侦察兵对峙的场景，字数800左右"

## 直接用base qwen，不用你的fine-tune
# ollama pull qwen2.5:7b-instruct
#ollama run qwen2.5:7b-instruct "以金庸武侠风格，写一个峨眉派弟子与蒙古侦察兵对峙的场景，约800字"