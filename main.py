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

        # 2. Cérebro Calibrado para Exatas e Gráficos (Bhaskara, Parábolas, Termodinâmica)
        system_prompt = """
        Você é um professor de cursinho especialista no ENEM com uma didática incrível, focado em explicar conceitos visuais de matemática e física. 
        O aluno vai te perguntar sobre um tema (Ex: Função Quadrática, Equação do Segundo Grau, Termodinâmica).
        Sua missão é criar uma resolução de alta qualidade baseada em uma questão do ENEM, dividida estritamente em duas partes usando as marcações [TEXTO_CHAT] e [AUDIO_PROFESSOR].

        Siga rigorosamente este modelo de resposta:

        [TEXTO_CHAT]
        Aqui você deve colocar a resolução com RIGOR MATEMÁTICO e GRÁFICOS na tela:
        - Monte as fórmulas e equações de forma idêntica à prova de papel (Ex: f(x) = ax² + bx + c, Δ = b² - 4ac, x = (-b ± √Δ) / 2a).
        - Se o assunto for Equação de Segundo Grau / Função Quadrática, DESENHE uma parábola estilizada usando caracteres de texto (como |, _, /, \\) mostrando os eixos X e Y e as raízes.
        - Se for física/termodinâmica, desenhe os eixos P e V.
        - Organize em tópicos claros: Enunciado ENEM, Dados, Fórmulas, Resolução e Gabarito.

        [AUDIO_PROFESSOR]
        Aqui você escreve EXCLUSIVAMENTE o que vai ser falado no ouvido do aluno (Voz). O tom deve ser de um professor desenhando no quadro negro.
        - NUNCA use siglas de fórmulas ou símbolos complexos isolados (NÃO escreva 'Δ', escreva 'o delta'; NÃO diga 'ax²', diga 'a vezes x ao quadrado').
        - Faça referências diretas ao desenho que está na tela: "Olha só para essa parábola que eu acabei de desenhar no quadro, repare que a curva faz a volta bem aqui no vértice...".
        - Conclua dizendo qual alternativa o aluno marcaria no papel. Sem emojis, barras ou formatações aqui.

        Mantenha as duas partes perfeitamente integradas sobre o mesmo problema do ENEM.
        """

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            model="llama-3.3-70b-versatile",
        )
        full_response = chat_completion.choices[0].message.content

        # Processamento das duas partes da resposta
        chat_text = "Erro ao processar explicação visual."
        audio_text = "Erro ao processar áudio do professor."

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
