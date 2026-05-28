import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

# Permite que o frontend converse com o backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicia o cliente do Groq buscando a chave de segurança nas variáveis do sistema
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Rota principal para carregar a página HTML
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

# Rota de inteligência que recebe o áudio
@app.post("/speak")
async def speak_endpoint(audio: UploadFile = File(...)):
    # 1. Salvar o arquivo de áudio temporariamente
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    try:
        # 2. Ouvidos: Groq Whisper (Transforma Áudio em Texto)
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en" # Força o modelo a entender/traduzir para o inglês
            )
        user_text = transcription.text

        # 3. Cérebro: Groq Llama 3.3 (O Professor de Inglês)
        system_prompt = """
        You are a friendly, enthusiastic, and encouraging American English teacher. 
        We are doing a speaking practice. 
        If I make a grammar mistake, gently correct me before replying. 
        Keep your answers brief, conversational, and ONLY in English. Do not use emojis.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
        )
        teacher_response = chat_completion.choices[0].message.content

        # Devolve para o site o que o aluno falou e a resposta do professor
        return {
            "user_text": user_text, 
            "teacher_response": teacher_response
        }

    finally:
        # Limpa o arquivo temporário do servidor para não lotar a memória
        os.remove(temp_audio_path)