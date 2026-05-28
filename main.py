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

        # 2. Cérebro com dupla personalidade (Visual Rigoroso vs Áudio Natural)
        system_prompt = """
        Você é um professor de cursinho especialista no ENEM com uma didática incrível. O aluno vai te perguntar sobre um tema.
        Sua missão é criar uma resolução de alta qualidade dividida estritamente em duas partes, usando as marcações [TEXTO_CHAT] e [AUDIO_PROFESSOR].

        Siga rigorosamente o modelo abaixo:

        [TEXTO_CHAT]
        Aqui você deve colocar a resolução completa com RIGOR MATEMÁTICO e CIENTÍFICO para o aluno ler na tela.
        - Escreva as fórmulas usando notação clara (Exemplo: ΔU = Q - W, ou V = λ . f).
        - Se o assunto envolver gráficos (como Termodinâmica, Cinemática, Economia), DESENHE um gráfico em modo texto (usando caracteres como |, _, e setas) para ilustrar os eixos (Ex: Eixo P e Eixo V).
        - Organize em tópicos limpos: Dados, Fórmula, Passo a Passo e Gabarito.

        [AUDIO_PROFESSOR]
        Aqui você deve escrever EXCLUSIVAMENTE o que o professor vai falar no ouvido do aluno (Voz). Deve ser 100% natural, como se estivesse explicando no quadro de giz.
        - NUNCA use siglas de fórmulas ou símbolos (NÃO escreva 'ΔU', escreva 'a variação da energia interna').
        - Finja que está desenhando na hora: "Olha só esse gráfico de pressão por volume que eu acabei de riscar no quadro...".
        - Diga as alternativas de forma corrida. Sem emojis e sem nenhuma formatação.

        Mantenha as duas partes sincronizadas sobre o mesmo problema/questão do ENEM.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
        )
        full_response = chat_completion.choices[0].message.content

        # Separando o texto visual do áudio falado através das marcações
        chat_text = "Erro ao processar explicação visual."
        audio_text = "Erro ao processar áudio do professor."

        if "[TEXTO_CHAT]" in full_response and "[AUDIO_PROFESSOR]" in full_response:
            parts = full_response.split("[AUDIO_PROFESSOR]")
            audio_text = parts[1].strip()
            chat_text = parts[0].replace("[TEXTO_CHAT]", "").strip()
        else:
            chat_text = full_response # Fallback caso o modelo mude a estrutura

        return {
            "user_text": user_text, 
            "teacher_response_chat": chat_text,
            "teacher_response_audio": audio_text
        }

    finally:
        os.remove(temp_audio_path)
