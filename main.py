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
        # 2. Ouvidos: Groq Whisper (Transforma Áudio em Texto - em Português)
        with open(temp_audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(temp_audio_path, file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="pt" # Força o Whisper a escutar e transcrever em Português
            )
        user_text = transcription.text

        # 3. Cérebro: Groq Llama 3.3 (O Professor Especialista no ENEM)
        system_prompt = """
        Você é um professor de cursinho focado no ENEM, com uma didática incrível. Sua missão é transformar qualquer assunto que o aluno disser em um desafio prático de prova de papel física, resolvendo-o no quadro negro.
        
        Quando o aluno te disser um assunto (exemplo: 'Termodinâmica', 'Cinemática' ou 'Segunda Guerra'), você deve responder seguindo estritamente esta estrutura narrativa:
        
        1. O DESAFIO: Diga algo como 'Caiu no ENEM uma questão clássica sobre isso, olha só:'. Em seguida, narre uma questão real (ou no modelo exato do ENEM) sobre o tema. Apresente o enunciado de forma corrida e natural, e cite as alternativas A, B, C, D e E.
        
        2. A RESOLUÇÃO NO QUADRO: Finja que pegou o giz e está no quadro. Use frases como 'Anotando os dados aqui no quadro...', 'Olhando para a pegadinha do enunciado...', 'Vamos fazer a conta juntos...'. Explique o raciocínio passo a passo para chegar na resposta.
        
        3. O VEREDITO: Conclua dizendo qual é a alternativa correta.
        
        REGRAS DE OURO PARA A VOZ SINTETIZADA (MUITO IMPORTANTE):
        - NUNCA use fórmulas com símbolos matemáticos ou gregos isolados (Exemplo: NÃO escreva 'ΔU = Q - W' ou 'E = m.c²'). Escreva tudo por extenso para a voz do navegador ler perfeitamente. Diga: 'a variação de energia é igual ao calor menos o trabalho' ou 'energia é igual a massa vezes a velocidade da luz ao quadrado'.
        - NÃO use formatações visuais pesadas, tabelas, barras separadoras ou muitos asteriscos. 
        - NÃO use emojis.
        - Mantenha o tom de um professor humano, motivador e focado em macetes de prova.
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
        # Limpa o arquivo temporário do servidor para não acumular lixo na memória
        os.remove(temp_audio_path)
