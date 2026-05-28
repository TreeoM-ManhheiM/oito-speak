import os
import tempfile
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/speak")
async def speak_endpoint(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    try:
        # 1. Ouvidos: Groq Whisper
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="pt"
            )
        user_text = transcription.text

        # 2. Cérebro: Focado estritamente no estilo clássico e limpo de papel de prova
        system_prompt = """
        Você é uma banca examinadora e um professor tutor especializado no formato ENEM.
        O usuário informará um assunto (Ex: Termodinâmica, Função Quadrática, Ecologia). 
        Sua resposta deve ser estruturada estritamente em duas divisões utilizando as tags [TEXTO_CHAT] e [AUDIO_PROFESSOR].

        [TEXTO_CHAT]
        Apresente uma questão limpa no modelo exato das provas impressas do ENEM:
        - Inicie com o texto de apoio e o enunciado da questão de forma contínua.
        - NUNCA desenhe gráficos usando barras, traços ou caracteres. Se a questão envolver dados gráficos, descreva-os diretamente no texto de forma clara (Ex: "Sabendo que o gráfico de pressão por volume apresenta uma linha reta horizontal constante...").
        - Liste as alternativas de forma organizada:
          A) 
          B) 
          C) 
          D) 
          E)
        - Abaixo das alternativas, adicione uma seção clara chamada "RESOLUÇÃO DA QUESTÃO", demonstrando os passos e as fórmulas aplicadas (Ex: delta = b² - 4ac) de modo direto e legível. Termine indicando claramente o Gabarito Correto.

        [AUDIO_PROFESSOR]
        Aqui você escreve EXCLUSIVAMENTE as falas explicativas do professor para leitura em áudio (Voz).
        - O tom deve ser de um professor lendo e comentando a questão para o aluno de forma contínua e estimulante.
        - NUNCA soe robótico ou leia equações termo por termo como códigos. Diga de forma natural: "Olhando para essa questão do ENEM, vemos que o segredo está em..."
        - Não use barras, marcadores textuais, asteriscos ou emojis nesta seção.

        Sincronize perfeitamente ambas as seções sobre o mesmo tópico solicitado.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
        )
        full_response = chat_completion.choices[0].message.content

        chat_text = "Erro ao carregar a questão impressa."
        audio_text = "Erro ao carregar áudio explicativo."

        if "[TEXTO_CHAT]" in full_response and "[AUDIO_PROFESSOR]" in full_response:
            parts = full_response.split("[AUDIO_PROFESSOR]")
            audio_text = parts[1].strip()
            chat_text = parts[0].replace("[TEXTO_CHAT]", "").strip()
        else:
            chat_text = full_response 

        return {
            "user_text": user_text, 
            "teacher_response_chat": chat_text,
            "teacher_response_audio": audio_text
        }

    finally:
        os.remove(temp_audio_path)
